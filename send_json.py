import json
import requests


# Load YOLO detection JSON
with open("detection_results.json", "r", encoding="utf-8") as file:
    detection_data = json.load(file)


# FastAPI endpoint
url = "http://127.0.0.1:8000/api/v1/detections"


# Send JSON to FastAPI
response = requests.post(
    url,
    json=detection_data
)


print("Status code:", response.status_code)
print("Response:")
print(response.json())