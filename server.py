import time
import os
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from detector import process_image


app = FastAPI(
    title="Indian Vehicle License Plate Detector API",
    version="1.0.0"
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Change in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Static folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")


# Serve static files
if os.path.exists(static_dir):
    app.mount(
        "/static",
        StaticFiles(directory=static_dir),
        name="static"
    )


# Home page
@app.get("/")
async def serve_index():

    index_path = os.path.join(static_dir, "index.html")

    if os.path.exists(index_path):
        return FileResponse(index_path)

    return {
        "message": "Indian Vehicle License Plate Recognition API is running."
    }


# Health check API
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "API is running"
    }


# License plate detection API
@app.post("/api/detect")
async def detect_license_plate(
    file: UploadFile = File(...)
):


    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )


    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded."
        )


    start_time = time.time()


    try:

        detections, annotated_b64 = process_image(contents)


    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"License plate detection failed: {str(e)}"
        )


    execution_time = round(
        time.time() - start_time,
        3
    )


    # Filter confidence if required
    CONFIDENCE_THRESHOLD = 0.50

    filtered_detections = []

    for detection in detections:

        confidence = detection.get(
            "confidence",
            1
        )

        if confidence >= CONFIDENCE_THRESHOLD:
            filtered_detections.append(detection)


    return {

        "success": True,

        "filename": file.filename,

        "execution_time_seconds": execution_time,

        "total_detections": len(filtered_detections),

        "detections": filtered_detections,

        "annotated_image": annotated_b64

    }