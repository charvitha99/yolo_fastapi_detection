# YOLO FastAPI Detection Pipeline

A video detection pipeline that uses YOLO for object detection, FastAPI for receiving and validating detection results, and PostgreSQL for storing validated detection events.

> Note: YOLO is currently used for the detection stage. The same pipeline can later be integrated with NVIDIA DeepStream for production video analytics.

---

## Project Overview

This project processes video files using YOLO object detection and generates structured JSON detection results.

The detection results are then:

1. Generated from video using YOLO.
2. Converted into JSON format.
3. Sent to a FastAPI REST API.
4. Validated using Pydantic models.
5. Stored in PostgreSQL.
6. Stored with both structured detection fields and JSON data.

### Pipeline

```text
Video Files
    ↓
YOLO Object Detection
    ↓
Detection Results
    ↓
JSON
    ↓
FastAPI
    ↓
Pydantic Validation
    ↓
PostgreSQL
    ↓
Detection Events