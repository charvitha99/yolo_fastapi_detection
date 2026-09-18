from ultralytics import YOLO
import json
import requests
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VIDEO_FOLDER = Path("videos")
OUTPUT_FOLDER = Path("json_results")

API_URL = "http://127.0.0.1:8000/api/v1/detections"

MODEL_PATH = "yolo11n.pt"


# Create output folder
OUTPUT_FOLDER.mkdir(exist_ok=True)


# Load YOLO model once
model = YOLO(MODEL_PATH)


# Get all MP4 videos
videos = sorted(VIDEO_FOLDER.glob("*.mp4"))


# Skip the video already processed
videos = [
    video for video in videos
    if video.name != "test_video.mp4"
]


print(f"Remaining videos found: {len(videos)}")


# --------------------------------------------------
# Process videos
# --------------------------------------------------

for video_path in videos:

    print("\n" + "=" * 60)
    print(f"Processing video: {video_path.name}")
    print("=" * 60)

    all_frames = []

    # Run YOLO
    results = model(str(video_path), stream=True)

    for frame_number, result in enumerate(results, start=1):

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bounding_box": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })

        all_frames.append({
            "frame_number": frame_number,
            "detections": detections
        })

    # Complete JSON
    output_data = {
        "video": video_path.name,
        "model": "yolo11n",
        "total_frames": len(all_frames),
        "frames": all_frames
    }

    # Save JSON
    json_path = OUTPUT_FOLDER / f"{video_path.stem}_detections.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=4)

    print(f"JSON created: {json_path}")
    print(f"Total frames: {len(all_frames)}")

    # --------------------------------------------------
    # Send JSON to FastAPI
    # --------------------------------------------------

    print("Sending detection data to FastAPI...")

    response = requests.post(
        API_URL,
        json=output_data,
        timeout=600
    )

    print(f"FastAPI status: {response.status_code}")

    try:
        print("FastAPI response:")
        print(response.json())
    except Exception:
        print(response.text)


print("\n" + "=" * 60)
print("ALL REMAINING VIDEOS COMPLETED")
print("=" * 60)
