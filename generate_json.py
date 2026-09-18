from ultralytics import YOLO
import json
from pathlib import Path


# Load YOLO model
model = YOLO("yolo11n.pt")

# Input video
video_path = "videos/test_video.mp4"

# Output JSON
output_path = "detection_results.json"

# Store all frame detections
all_results = []

# Run YOLO on the video
results = model(video_path, stream=True)

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

    frame_data = {
        "frame_number": frame_number,
        "detections": detections
    }

    all_results.append(frame_data)

# Complete JSON response
output_data = {
    "video": Path(video_path).name,
    "model": "yolo11n",
    "total_frames": len(all_results),
    "frames": all_results
}

# Save JSON
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=4)

print(f"JSON generated successfully: {output_path}")
print(f"Total frames processed: {len(all_results)}")