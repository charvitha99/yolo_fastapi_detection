from fastapi import FastAPI
from typing import Dict, Any
import os
import json
import psycopg
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="YOLO Detection and Activity API",
    version="1.0.0"
)


# ============================================================
# POSTGRESQL CONNECTION
# ============================================================

def get_postgres_connection():

    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


# ============================================================
# ROOT API
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "YOLO Detection and Activity API is running"
    }


# ============================================================
# DETECTION API
# ============================================================

@app.post("/api/v1/detections")
def receive_detection(
    data: Dict[str, Any]
):

    print(
        f"Detection data received: "
        f"video={data.get('video')}, "
        f"model={data.get('model')}, "
        f"frames={data.get('total_frames')}"
    )

    return {
        "message":
            "Detection data received successfully",

        "video":
            data.get("video"),

        "model":
            data.get("model"),

        "total_frames":
            data.get("total_frames")
    }


# ============================================================
# ACTIVITY API
# ============================================================

@app.post("/api/v1/activity")
def receive_activity(
    data: Dict[str, Any]
):

    video_name = data.get(
        "video_name"
    )

    persons = data.get(
        "persons",
        []
    )

    model_name = "yolo11s-pose"

    connection = get_postgres_connection()

    events_inserted = 0

    try:

        with connection.cursor() as cursor:

            # ====================================================
            # READ PERSONS
            # ====================================================

            for person in persons:

                person_id = person.get(
                    "person_id"
                )

                activities = person.get(
                    "activities",
                    []
                )

                # =================================================
                # READ ACTIVITIES FOR EACH PERSON
                # =================================================

                for activity_event in activities:

                    activity = activity_event.get(
                        "activity"
                    )

                    start_time = activity_event.get(
                        "start_time"
                    )

                    end_time = activity_event.get(
                        "end_time"
                    )

                    duration_seconds = activity_event.get(
                        "duration_seconds"
                    )

                    # =================================================
                    # INSERT INTO POSTGRESQL
                    # =================================================

                    cursor.execute(
                        """
                        INSERT INTO person_activity_events
                        (
                            video_name,
                            person_id,
                            activity,
                            start_time,
                            end_time,
                            duration_seconds,
                            start_frame,
                            end_frame,
                            model_name,
                            json_data
                        )
                        VALUES
                        (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        )
                        """,
                        (
                            video_name,
                            person_id,
                            activity,
                            start_time,
                            end_time,
                            duration_seconds,
                            0,
                            0,
                            model_name,
                            json.dumps(
                                activity_event
                            )
                        )
                    )

                    events_inserted += 1

        # ========================================================
        # COMMIT TRANSACTION
        # ========================================================

        connection.commit()

    finally:

        connection.close()


    # ============================================================
    # LOG
    # ============================================================

    print(
        f"Activity data received: "
        f"video={video_name}, "
        f"persons={len(persons)}, "
        f"events={events_inserted}"
    )


    # ============================================================
    # RESPONSE
    # ============================================================

    return {

        "message":
            "Activity data received and stored successfully",

        "video":
            video_name,

        "model":
            model_name,

        "total_persons":
            data.get(
                "summary",
                {}
            ).get(
                "total_unique_persons",
                0
            ),

        "activity_counts":
            data.get(
                "summary",
                {}
            ),

        "events_inserted":
            events_inserted
    }