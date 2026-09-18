from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
import json

from database import get_connection


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="YOLO Detection API",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Pydantic Models
# --------------------------------------------------

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox


class FrameDetection(BaseModel):
    frame_number: int
    detections: List[Detection]


class DetectionRequest(BaseModel):
    video: str
    model: str
    total_frames: int
    frames: List[FrameDetection]


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():
    logger.info("Home endpoint accessed")

    return {
        "message": "YOLO Detection API is running"
    }


# --------------------------------------------------
# Detection API
# --------------------------------------------------

@app.post("/api/v1/detections")
def receive_detections(data: DetectionRequest):

    logger.info(
        "Detection data received: video=%s, model=%s, frames=%s",
        data.video,
        data.model,
        data.total_frames
    )

    connection = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        insert_query = """
            INSERT INTO detection_events (
                video_name,
                model_name,
                frame_number,
                class_id,
                class_name,
                confidence,
                x1,
                y1,
                x2,
                y2,
                json_data
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
        """

        rows = []
        events_inserted = 0

        for frame in data.frames:

            # Store only this frame's JSON,
            # instead of the entire 10 MB video JSON.
            frame_json = json.dumps({
                "video": data.video,
                "model": data.model,
                "frame_number": frame.frame_number,
                "detections": [
                    detection.model_dump()
                    for detection in frame.detections
                ]
            })

            for detection in frame.detections:

                rows.append(
                    (
                        data.video,
                        data.model,
                        frame.frame_number,
                        detection.class_id,
                        detection.class_name,
                        detection.confidence,
                        detection.bounding_box.x1,
                        detection.bounding_box.y1,
                        detection.bounding_box.x2,
                        detection.bounding_box.y2,
                        frame_json
                    )
                )

                events_inserted += 1

        logger.info(
            "Prepared %s detection events",
            events_inserted
        )

        if rows:
            cursor.executemany(
                insert_query,
                rows
            )

        connection.commit()

        cursor.close()
        connection.close()

        logger.info(
            "Detection events stored successfully: %s events",
            events_inserted
        )

        return {
            "message": "Detection data received and stored successfully",
            "video": data.video,
            "model": data.model,
            "total_frames": data.total_frames,
            "events_inserted": events_inserted
        }

    except Exception as e:

        if connection:
            connection.rollback()
            connection.close()

        logger.error(
            "Failed to store detection events: %s",
            str(e)
        )

        return {
            "message": "Failed to store detection data",
            "error": str(e)
        }