import json
import math


INPUT_PATH = "person_tracking.json"
OUTPUT_PATH = "activity_features.json"


# COCO / YOLO pose keypoint indexes
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

LEFT_HIP = 11
RIGHT_HIP = 12

LEFT_KNEE = 13
RIGHT_KNEE = 14

LEFT_ANKLE = 15
RIGHT_ANKLE = 16


KEYPOINT_CONFIDENCE_THRESHOLD = 0.30


def distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
    )


def midpoint(point1, point2):
    """Calculate midpoint between two points."""
    return [
        (point1[0] + point2[0]) / 2,
        (point1[1] + point2[1]) / 2
    ]


def valid_keypoint(keypoint):
    """Check whether a keypoint has sufficient confidence."""
    return (
        keypoint is not None
        and len(keypoint) >= 3
        and keypoint[2] is not None
        and keypoint[2] >= KEYPOINT_CONFIDENCE_THRESHOLD
    )


def get_keypoint(keypoints, index):
    """Return a keypoint only if it is valid."""
    if index >= len(keypoints):
        return None

    point = keypoints[index]

    if not valid_keypoint(point):
        return None

    return [float(point[0]), float(point[1])]


def calculate_features(person):
    """Calculate pose features for one person in one frame."""

    keypoints = person["keypoints"]

    left_shoulder = get_keypoint(keypoints, LEFT_SHOULDER)
    right_shoulder = get_keypoint(keypoints, RIGHT_SHOULDER)

    left_hip = get_keypoint(keypoints, LEFT_HIP)
    right_hip = get_keypoint(keypoints, RIGHT_HIP)

    left_knee = get_keypoint(keypoints, LEFT_KNEE)
    right_knee = get_keypoint(keypoints, RIGHT_KNEE)

    left_ankle = get_keypoint(keypoints, LEFT_ANKLE)
    right_ankle = get_keypoint(keypoints, RIGHT_ANKLE)

    features = {
        "person_id": person["person_id"],
        "confidence": person["confidence"],
        "bbox": person["bbox"],
        "shoulder_center": None,
        "hip_center": None,
        "knee_center": None,
        "ankle_center": None,
        "torso_length": None,
        "leg_length": None,
        "hip_knee_distance": None,
        "knee_ankle_distance": None
    }

    # Shoulder center
    if left_shoulder and right_shoulder:
        features["shoulder_center"] = midpoint(
            left_shoulder,
            right_shoulder
        )

    # Hip center
    if left_hip and right_hip:
        features["hip_center"] = midpoint(
            left_hip,
            right_hip
        )

    # Knee center
    if left_knee and right_knee:
        features["knee_center"] = midpoint(
            left_knee,
            right_knee
        )

    # Ankle center
    if left_ankle and right_ankle:
        features["ankle_center"] = midpoint(
            left_ankle,
            right_ankle
        )

    # Torso length
    if features["shoulder_center"] and features["hip_center"]:
        features["torso_length"] = distance(
            features["shoulder_center"],
            features["hip_center"]
        )

    # Average leg length
    leg_lengths = []

    if left_hip and left_knee and left_ankle:
        leg_lengths.append(
            distance(left_hip, left_knee)
            + distance(left_knee, left_ankle)
        )

    if right_hip and right_knee and right_ankle:
        leg_lengths.append(
            distance(right_hip, right_knee)
            + distance(right_knee, right_ankle)
        )

    if leg_lengths:
        features["leg_length"] = sum(leg_lengths) / len(leg_lengths)

    # Hip → knee distance
    hip_knee_distances = []

    if left_hip and left_knee:
        hip_knee_distances.append(
            distance(left_hip, left_knee)
        )

    if right_hip and right_knee:
        hip_knee_distances.append(
            distance(right_hip, right_knee)
        )

    if hip_knee_distances:
        features["hip_knee_distance"] = (
            sum(hip_knee_distances)
            / len(hip_knee_distances)
        )

    # Knee → ankle distance
    knee_ankle_distances = []

    if left_knee and left_ankle:
        knee_ankle_distances.append(
            distance(left_knee, left_ankle)
        )

    if right_knee and right_ankle:
        knee_ankle_distances.append(
            distance(right_knee, right_ankle)
        )

    if knee_ankle_distances:
        features["knee_ankle_distance"] = (
            sum(knee_ankle_distances)
            / len(knee_ankle_distances)
        )

    return features


# Load tracking data
with open(INPUT_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)


feature_frames = []

for frame in data["frames"]:

    frame_features = {
        "frame_number": frame["frame_number"],
        "timestamp_seconds": frame["timestamp_seconds"],
        "persons": []
    }

    for person in frame["persons"]:

        features = calculate_features(person)

        frame_features["persons"].append(features)

    feature_frames.append(frame_features)


# Final output
output = {
    "video_name": data["video_name"],
    "model_name": data["model_name"],
    "fps": data["fps"],
    "total_frames": data["total_frames"],
    "unique_person_ids": data["unique_person_ids"],
    "unique_person_count": data["unique_person_count"],
    "frames": feature_frames
}


# Save feature data
with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(output, file, indent=2)


print("Activity feature analysis completed.")
print(f"Frames analyzed: {len(feature_frames)}")
print(f"Unique tracking IDs: {data['unique_person_count']}")
print(f"Output saved to: {OUTPUT_PATH}")