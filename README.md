# YOLO Object Detection

## 📌 Project Overview

This project is a video-based object detection system using **YOLO11** and **OpenCV**.

The system processes video files, detects objects frame by frame, generates detection results in **JSON format**, and sends the results to a **FastAPI** backend for processing and storage in **PostgreSQL**.

---

## 🛠️ Technologies Used

- Python
- YOLO11
- Ultralytics
- OpenCV
- FastAPI
- PostgreSQL
- Psycopg
- Uvicorn
- JSON

---

## 📂 Project Structure

```text
yolo_fastapi_detection/
│
├── main.py
├── database.py
├── generate_json.py
├── send_json.py
├── process_remaining_videos.py
├── test_db.py
├── requirements.txt
├── README.md
├── .gitignore
└── ...