import cv2
import json
import math
import os
import subprocess
from collections import defaultdict, deque

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "videos/mixkit-open-office-space-914-hd-ready.mp4"

MODEL_PATH = "yolo11s-pose.pt"

OUTPUT_VIDEO = "activity_output.mp4"
TEMP_VIDEO = "activity_output_temp.avi"

OUTPUT_JSON = "activity_results.json"

TRACKER_CONFIG = "botsort_activity.yaml"

CONFIDENCE = 0.05
IMAGE_SIZE = 1280
MAX_DETECTIONS = 300

# Maximum persistent Person IDs
MAX_PERSON_IDS = 30

# How long a lost person's ID can be reused
MAX_LOST_FRAMES = 120

# Maximum distance for reconnecting a lost person
MAX_REIDENTIFICATION_DISTANCE = 180

# Activity must remain stable for this many frames
MIN_ACTIVITY_FRAMES = 24


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded.")
print()


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 24.0


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


print("========================================")
print("VIDEO INFORMATION")
print("========================================")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Resolution: {width} x {height}")
print("========================================")
print()


# ============================================================
# TEMPORARY VIDEO WRITER
#
# We first create AVI using OpenCV.
# After processing, FFmpeg converts it to H.264 MP4.
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"XVID"
)

writer = cv2.VideoWriter(
    TEMP_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create temporary video."
    )


# ============================================================
# PERSON ID MANAGEMENT
# ============================================================

# YOLO tracker ID -> persistent Person ID
tracker_to_person = {}

# Person ID -> last known center
person_registry = {}

# Person ID -> last frame seen
person_last_seen = {}

# Persons currently visible
active_person_ids = set()

# Persons temporarily lost
lost_person_ids = set()

# All IDs that have been created
all_person_ids = set()

# Fixed pool of possible IDs
available_person_ids = set(
    range(1, MAX_PERSON_IDS + 1)
)


# ============================================================
# POSITION HISTORY
# ============================================================

position_history = defaultdict(
    lambda: deque(maxlen=10)
)


# ============================================================
# ACTIVITY STATE
# ============================================================

# Confirmed activity
current_activity = {}

# Frame at which confirmed activity started
activity_start_frame = {}

# Activity candidate waiting for confirmation
pending_activity = {}

# Frame at which candidate activity started
pending_activity_start = {}


# ============================================================
# ACTIVITY RESULTS
# ============================================================

activity_events = []

activity_counts = {
    "standing": 0,
    "sitting": 0,
    "walking": 0
}


# ============================================================
# COUNTERS
# ============================================================

frame_number = 0

max_person_count = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_center(
    x1,
    y1,
    x2,
    y2
):
    """
    Calculate center of bounding box.
    """

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def calculate_distance(
    point1,
    point2
):
    """
    Calculate Euclidean distance.
    """

    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        +
        (point1[1] - point2[1]) ** 2
    )


def calculate_angle(
    point_a,
    point_b,
    point_c
):
    """
    Calculate angle ABC.
    """

    vector1 = (
        point_a[0] - point_b[0],
        point_a[1] - point_b[1]
    )

    vector2 = (
        point_c[0] - point_b[0],
        point_c[1] - point_b[1]
    )

    magnitude1 = math.sqrt(
        vector1[0] ** 2 +
        vector1[1] ** 2
    )

    magnitude2 = math.sqrt(
        vector2[0] ** 2 +
        vector2[1] ** 2
    )

    if magnitude1 == 0 or magnitude2 == 0:
        return 180.0

    dot_product = (
        vector1[0] * vector2[0]
        +
        vector1[1] * vector2[1]
    )

    cosine = (
        dot_product /
        (magnitude1 * magnitude2)
    )

    cosine = max(
        -1.0,
        min(1.0, cosine)
    )

    return math.degrees(
        math.acos(cosine)
    )


# ============================================================
# FIND REUSABLE PERSON ID
# ============================================================

def find_reusable_person(
    center,
    frame_number
):

    best_person_id = None

    best_distance = float("inf")


    for person_id in list(
        lost_person_ids
    ):

        if person_id not in person_registry:
            continue


        last_seen = person_last_seen.get(
            person_id,
            -999999
        )


        frames_lost = (
            frame_number -
            last_seen
        )


        if frames_lost > MAX_LOST_FRAMES:
            continue


        old_center = person_registry[
            person_id
        ]


        distance = calculate_distance(
            center,
            old_center
        )


        if (
            distance
            <= MAX_REIDENTIFICATION_DISTANCE
        ):

            if distance < best_distance:

                best_distance = distance

                best_person_id = (
                    person_id
                )


    return best_person_id


# ============================================================
# ASSIGN PERSISTENT PERSON ID
# ============================================================

def assign_person_id(
    tracker_id,
    center,
    frame_number
):

    # --------------------------------------------------------
    # Existing tracker ID
    # --------------------------------------------------------

    if tracker_id in tracker_to_person:

        person_id = tracker_to_person[
            tracker_id
        ]

        active_person_ids.add(
            person_id
        )

        lost_person_ids.discard(
            person_id
        )

        person_registry[
            person_id
        ] = center

        person_last_seen[
            person_id
        ] = frame_number

        return person_id


    # --------------------------------------------------------
    # Try to reconnect lost person
    # --------------------------------------------------------

    person_id = find_reusable_person(
        center,
        frame_number
    )


    if person_id is not None:

        tracker_to_person[
            tracker_id
        ] = person_id

        lost_person_ids.discard(
            person_id
        )

        active_person_ids.add(
            person_id
        )

        person_registry[
            person_id
        ] = center

        person_last_seen[
            person_id
        ] = frame_number

        return person_id


    # --------------------------------------------------------
    # Get unused ID
    # --------------------------------------------------------

    unused_ids = (
        available_person_ids
        -
        all_person_ids
    )


    if unused_ids:

        person_id = min(
            unused_ids
        )

        tracker_to_person[
            tracker_id
        ] = person_id

        all_person_ids.add(
            person_id
        )

        active_person_ids.add(
            person_id
        )

        person_registry[
            person_id
        ] = center

        person_last_seen[
            person_id
        ] = frame_number

        return person_id


    # --------------------------------------------------------
    # If all 30 IDs are already created,
    # reuse a lost ID.
    # --------------------------------------------------------

    if lost_person_ids:

        person_id = min(
            lost_person_ids
        )

        lost_person_ids.discard(
            person_id
        )

        tracker_to_person[
            tracker_id
        ] = person_id

        active_person_ids.add(
            person_id
        )

        person_registry[
            person_id
        ] = center

        person_last_seen[
            person_id
        ] = frame_number

        return person_id


    # All 30 IDs are currently active.
    return None


# ============================================================
# ACTIVITY CLASSIFICATION
# ============================================================

def classify_activity(
    keypoints,
    x1,
    y1,
    x2,
    y2,
    person_id
):
    """
    Classify person as walking, sitting or standing.
    """

    # ========================================================
    # PERSON CENTER
    # ========================================================

    center = calculate_center(
        x1,
        y1,
        x2,
        y2
    )


    position_history[
        person_id
    ].append(center)


    # ========================================================
    # WALKING DETECTION
    # ========================================================

    # Walking is checked first.
    # This prevents a moving person from being
    # incorrectly classified as sitting.

    if len(
        position_history[person_id]
    ) >= 8:

        old_position = (
            position_history[
                person_id
            ][0]
        )


        movement = calculate_distance(
            old_position,
            center
        )


        if movement > 20:

            return "walking"


    # ========================================================
    # SITTING DETECTION
    # ========================================================

    sitting_detected = False


    if keypoints is not None:

        try:

            points = (
                keypoints.xy
                .cpu()
                .numpy()
            )


            if len(points) > 0:

                points = points[0]


                # ------------------------------------------------
                # COCO KEYPOINTS
                # ------------------------------------------------

                left_shoulder = points[5]
                right_shoulder = points[6]

                left_hip = points[11]
                right_hip = points[12]

                left_knee = points[13]
                right_knee = points[14]

                left_ankle = points[15]
                right_ankle = points[16]


                # ------------------------------------------------
                # Valid keypoints
                # ------------------------------------------------

                left_valid = (
                    left_hip[0] > 0
                    and
                    left_knee[0] > 0
                    and
                    left_ankle[0] > 0
                )


                right_valid = (
                    right_hip[0] > 0
                    and
                    right_knee[0] > 0
                    and
                    right_ankle[0] > 0
                )


                # ------------------------------------------------
                # Knee angles
                # ------------------------------------------------

                left_angle = None

                right_angle = None


                if left_valid:

                    left_angle = calculate_angle(
                        left_hip,
                        left_knee,
                        left_ankle
                    )


                if right_valid:

                    right_angle = calculate_angle(
                        right_hip,
                        right_knee,
                        right_ankle
                    )


                # ------------------------------------------------
                # Sitting based on bent knee
                #
                # Use a conservative threshold.
                # ------------------------------------------------

                if (
                    left_angle is not None
                    and
                    right_angle is not None
                ):

                    average_angle = (
                        left_angle +
                        right_angle
                    ) / 2


                    if average_angle < 125:

                        sitting_detected = True


                elif left_angle is not None:

                    if left_angle < 120:

                        sitting_detected = True


                elif right_angle is not None:

                    if right_angle < 120:

                        sitting_detected = True


                # ------------------------------------------------
                # Additional body geometry check
                # ------------------------------------------------

                if (
                    left_valid
                    or
                    right_valid
                ):

                    shoulder_y = (
                        left_shoulder[1]
                        +
                        right_shoulder[1]
                    ) / 2


                    hip_y = (
                        left_hip[1]
                        +
                        right_hip[1]
                    ) / 2


                    knee_y = (
                        left_knee[1]
                        +
                        right_knee[1]
                    ) / 2


                    body_height = (
                        y2 - y1
                    )


                    hip_knee_distance = abs(
                        knee_y -
                        hip_y
                    )


                    if body_height > 0:

                        ratio = (
                            hip_knee_distance /
                            body_height
                        )


                        # Only use this as an
                        # additional sitting signal.
                        if ratio < 0.20:

                            sitting_detected = True


        except Exception:

            sitting_detected = False


    # ========================================================
    # SITTING
    # ========================================================

    if sitting_detected:

        return "sitting"


    # ========================================================
    # DEFAULT = STANDING
    # ========================================================

    return "standing"


# ============================================================
# UPDATE ACTIVITY
# ============================================================

def update_activity(
    person_id,
    detected_activity,
    frame_number
):

    # --------------------------------------------------------
    # First activity
    # --------------------------------------------------------

    if person_id not in current_activity:

        current_activity[
            person_id
        ] = detected_activity

        activity_start_frame[
            person_id
        ] = frame_number

        return


    current = current_activity[
        person_id
    ]


    # --------------------------------------------------------
    # Same activity
    # --------------------------------------------------------

    if detected_activity == current:

        pending_activity.pop(
            person_id,
            None
        )

        pending_activity_start.pop(
            person_id,
            None
        )

        return


    # --------------------------------------------------------
    # New candidate activity
    # --------------------------------------------------------

    if (
        person_id not in pending_activity
        or
        pending_activity[person_id]
        != detected_activity
    ):

        pending_activity[
            person_id
        ] = detected_activity

        pending_activity_start[
            person_id
        ] = frame_number

        return


    # --------------------------------------------------------
    # Check whether candidate is stable
    # --------------------------------------------------------

    pending_start = (
        pending_activity_start[
            person_id
        ]
    )


    stable_frames = (
        frame_number -
        pending_start
    )


    if stable_frames < MIN_ACTIVITY_FRAMES:

        return


    # ========================================================
    # ACTIVITY CHANGE CONFIRMED
    # ========================================================

    old_activity = current_activity[
        person_id
    ]


    old_start_frame = (
        activity_start_frame[
            person_id
        ]
    )


    duration_frames = (
        frame_number -
        old_start_frame
    )


    duration_seconds = (
        duration_frames /
        fps
    )


    if duration_seconds > 0:

        event = {

            "person_id":
                person_id,

            "activity":
                old_activity,

            "start_time":
                round(
                    old_start_frame /
                    fps,
                    3
                ),

            "end_time":
                round(
                    frame_number /
                    fps,
                    3
                ),

            "duration_seconds":
                round(
                    duration_seconds,
                    3
                ),

            "start_frame":
                old_start_frame,

            "end_frame":
                frame_number
        }


        activity_events.append(
            event
        )


        activity_counts[
            old_activity
        ] += 1


    # --------------------------------------------------------
    # Update confirmed activity
    # --------------------------------------------------------

    current_activity[
        person_id
    ] = detected_activity


    activity_start_frame[
        person_id
    ] = pending_start


    pending_activity.pop(
        person_id,
        None
    )


    pending_activity_start.pop(
        person_id,
        None
    )


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

print(
    "Starting activity detection..."
)

print()


while True:

    success, frame = cap.read()


    if not success:
        break


    frame_number += 1


    # ========================================================
    # YOLO TRACK
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        max_det=MAX_DETECTIONS,
        verbose=False
    )


    result = results[0]


    current_frame_person_ids = set()


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if (
        result.boxes is not None
        and
        result.boxes.id is not None
    ):

        boxes = result.boxes


        tracker_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )


        class_ids = (
            boxes.cls
            .int()
            .cpu()
            .tolist()
        )


        if result.keypoints is not None:

            keypoints_data = (
                result.keypoints
            )

        else:

            keypoints_data = None


        # ----------------------------------------------------
        # Process each detection
        # ----------------------------------------------------

        for index, tracker_id in enumerate(
            tracker_ids
        ):

            # COCO person class
            if class_ids[index] != 0:
                continue


            x1, y1, x2, y2 = (
                boxes.xyxy[index]
                .cpu()
                .tolist()
            )


            center = calculate_center(
                x1,
                y1,
                x2,
                y2
            )


            # ------------------------------------------------
            # Persistent Person ID
            # ------------------------------------------------

            person_id = assign_person_id(
                tracker_id,
                center,
                frame_number
            )


            if person_id is None:
                continue


            current_frame_person_ids.add(
                person_id
            )


            # ------------------------------------------------
            # Keypoints
            # ------------------------------------------------

            keypoints = None


            if keypoints_data is not None:

                try:

                    keypoints = (
                        keypoints_data[
                            index
                        ]
                    )

                except Exception:

                    keypoints = None


            # ------------------------------------------------
            # Activity
            # ------------------------------------------------

            detected_activity = (
                classify_activity(
                    keypoints,
                    x1,
                    y1,
                    x2,
                    y2,
                    person_id
                )
            )


            # ------------------------------------------------
            # Stable activity
            # ------------------------------------------------

            update_activity(
                person_id,
                detected_activity,
                frame_number
            )


            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (
                    int(x1),
                    int(y1)
                ),
                (
                    int(x2),
                    int(y2)
                ),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # Display activity
            # ------------------------------------------------

            display_activity = (
                current_activity.get(
                    person_id,
                    detected_activity
                )
            )


            label = (
                f"Person {person_id} | "
                f"{display_activity}"
            )


            cv2.putText(
                frame,
                label,
                (
                    int(x1),
                    max(
                        20,
                        int(y1) - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


    # ========================================================
    # CURRENT PERSON COUNT
    # ========================================================

    visible_count = len(
        current_frame_person_ids
    )


    if visible_count > max_person_count:

        max_person_count = (
            visible_count
        )


    # ========================================================
    # LOST PERSON MANAGEMENT
    # ========================================================

    disappeared_persons = (
        active_person_ids
        -
        current_frame_person_ids
    )


    for person_id in disappeared_persons:

        if person_id in person_last_seen:

            frames_lost = (
                frame_number
                -
                person_last_seen[
                    person_id
                ]
            )


            if (
                frames_lost
                <= MAX_LOST_FRAMES
            ):

                lost_person_ids.add(
                    person_id
                )


    active_person_ids = (
        current_frame_person_ids
    )


    # ========================================================
    # DISPLAY COUNTERS
    # ========================================================

    cv2.putText(
        frame,
        f"Current Persons: {visible_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Maximum Persons: {max_person_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    writer.write(frame)


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 50 == 0:

        print(
            f"Processed frame "
            f"{frame_number}/{total_frames} | "
            f"Current persons: "
            f"{visible_count} | "
            f"Maximum persons: "
            f"{max_person_count} | "
            f"IDs used: "
            f"{len(all_person_ids)}"
        )


# ============================================================
# FINISH FINAL ACTIVITIES
# ============================================================

for person_id, activity in (
    current_activity.items()
):

    start_frame = (
        activity_start_frame.get(
            person_id
        )
    )


    if start_frame is None:
        continue


    duration_frames = (
        frame_number -
        start_frame
    )


    duration_seconds = (
        duration_frames /
        fps
    )


    if duration_seconds > 0:

        event = {

            "person_id":
                person_id,

            "activity":
                activity,

            "start_time":
                round(
                    start_frame /
                    fps,
                    3
                ),

            "end_time":
                round(
                    frame_number /
                    fps,
                    3
                ),

            "duration_seconds":
                round(
                    duration_seconds,
                    3
                ),

            "start_frame":
                start_frame,

            "end_frame":
                frame_number
        }


        activity_events.append(
            event
        )


        activity_counts[
            activity
        ] += 1


# ============================================================
# CLOSE OPENCV
# ============================================================

cap.release()

writer.release()


# ============================================================
# CONVERT VIDEO TO H.264 MP4
# ============================================================

print()
print("Converting output video to H.264...")

ffmpeg_command = [
    "ffmpeg",
    "-y",
    "-i",
    TEMP_VIDEO,
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    OUTPUT_VIDEO
]


try:

    result = subprocess.run(
        ffmpeg_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


    if result.returncode != 0:

        print()
        print(
            "FFmpeg conversion failed."
        )

        print(
            result.stderr
        )

    else:

        print(
            "H.264 video created successfully."
        )

        # Remove temporary AVI
        if os.path.exists(TEMP_VIDEO):

            os.remove(
                TEMP_VIDEO
            )


except FileNotFoundError:

    print()
    print(
        "FFmpeg was not found."
    )

    print(
        "The temporary video is available as:"
    )

    print(
        TEMP_VIDEO
    )


# ============================================================
# CREATE JSON
# ============================================================

person_ids = sorted(all_person_ids)

# ------------------------------------------------------------
# Group activities by Person ID
# ------------------------------------------------------------

persons = {}

for person_id in person_ids:
    persons[person_id] = {
        "person_id": person_id,
        "activities": []
    }


for event in activity_events:

    person_id = event["person_id"]

    if person_id not in persons:
        persons[person_id] = {
            "person_id": person_id,
            "activities": []
        }

    persons[person_id]["activities"].append(
        {
            "activity": event["activity"],
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "duration_seconds": event["duration_seconds"]
        }
    )


# Convert dictionary to list
persons_list = list(persons.values())


# ------------------------------------------------------------
# Calculate video duration
# ------------------------------------------------------------

video_duration_seconds = round(
    total_frames / fps,
    3
)


# ------------------------------------------------------------
# Create final JSON
# ------------------------------------------------------------

activity_result = {

    "video_name":
        os.path.basename(VIDEO_PATH),

    "video_duration_seconds":
        video_duration_seconds,

    "summary": {

        "total_unique_persons":
            len(person_ids),

        "walking":
            activity_counts["walking"],

        "standing":
            activity_counts["standing"],

        "sitting":
            activity_counts["sitting"]
    },

    "persons":
        persons_list
}


# ------------------------------------------------------------
# Save JSON
# ------------------------------------------------------------

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        activity_result,
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("========================================")
print("PROCESSING COMPLETED")
print("========================================")

print(
    f"Maximum persons visible at one time: "
    f"{max_person_count}"
)

print(
    f"Total Person IDs: "
    f"{len(person_ids)}"
)

print(
    f"Person IDs: "
    f"{person_ids}"
)

print(
    f"Standing events: "
    f"{activity_counts['standing']}"
)

print(
    f"Sitting events: "
    f"{activity_counts['sitting']}"
)

print(
    f"Walking events: "
    f"{activity_counts['walking']}"
)

print(
    f"Total activity events: "
    f"{len(activity_events)}"
)

print()
print(
    f"JSON: {OUTPUT_JSON}"
)

print(
    f"Video: {OUTPUT_VIDEO}"
)

print("========================================")