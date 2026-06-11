"""
fastapi_app.py
--------------
FastAPI backend for steel surface defect analysis.

Run with:
    uvicorn fastapi_app:app --host 0.0.0.0 --port 8000

Endpoints
---------
POST /analyse          Upload image → JSON analysis report
POST /analyse/overlay  Upload image → annotated JPEG image response
GET  /health           Health check
GET  /models           List available ONNX models
"""

import io
import os
import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from DefectAnalysis.pipeline import ONNXInferencePipeline

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Steel Defect Analysis API",
    description="ONNX-based semantic segmentation + defect analytics",
    version="1.0.0",
)

ONNX_MODELS = {
    "fpn_resnet34": "ONNX_models/fpn_resnet.onnx",
}

# Cached pipeline instances (one per model key)
_pipelines: dict[str, ONNXInferencePipeline] = {}


def get_pipeline(model_key: str, threshold: float, min_pixels: int) -> ONNXInferencePipeline:
    if model_key not in ONNX_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model_key}'. Available: {list(ONNX_MODELS.keys())}",
        )
    path = ONNX_MODELS[model_key]
    if not os.path.isfile(path):
        raise HTTPException(
            status_code=503,
            detail=f"ONNX model file not found: '{path}'. Export it first.",
        )
    # Create or reuse cached pipeline
    if model_key not in _pipelines:
        _pipelines[model_key] = ONNXInferencePipeline(path)
    pipe = _pipelines[model_key]
    pipe.threshold         = threshold
    pipe.min_defect_pixels = min_pixels
    return pipe


def decode_upload(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes → BGR numpy array."""
    arr    = np.frombuffer(file_bytes, dtype=np.uint8)
    img    = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Upload a valid JPG/PNG.")
    return img


# ── Response model ────────────────────────────────────────────────────────────
class AnalysisResponse(BaseModel):
    model_used:        str
    defect_percentage: float
    severity:          str
    risk_score:        float
    per_class_area:    dict
    rul_days:          int
    rul_method:        str
    recommendation:    str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    return {
        k: {"path": v, "available": os.path.isfile(v)}
        for k, v in ONNX_MODELS.items()
    }


@app.post("/analyse", response_model=AnalysisResponse)
async def analyse(
    file:       UploadFile = File(..., description="Steel surface image (JPG/PNG)"),
    model:      str        = Query("fpn_resnet34", description="Model key"),
    threshold:  float      = Query(0.5, ge=0.1, le=0.9),
    min_pixels: int        = Query(20,  ge=0),
):
    """
    Analyse an uploaded steel surface image.
    Returns JSON with defect metrics, severity, risk score, and RUL estimate.
    """
    img  = decode_upload(await file.read())
    pipe = get_pipeline(model, threshold, min_pixels)
    res  = pipe.run_from_array(img)

    report = res["report"]
    rul    = res["rul"]

    return AnalysisResponse(
        model_used        = model,
        defect_percentage = report["defect_percentage"],
        severity          = report["severity"],
        risk_score        = report["risk_score"],
        per_class_area    = report["per_class_area"],
        rul_days          = rul["rul_days"],
        rul_method        = rul["method"],
        recommendation    = rul["rul_recommendation"],
    )


@app.post("/analyse/overlay")
async def analyse_overlay(
    file:       UploadFile = File(...),
    model:      str        = Query("fpn_resnet34"),
    threshold:  float      = Query(0.5, ge=0.1, le=0.9),
    min_pixels: int        = Query(20,  ge=0),
    output:     str        = Query("overlay", description="overlay | heatmap"),
):
    """
    Analyse an uploaded image and return the annotated image as JPEG.
    """
    img  = decode_upload(await file.read())
    pipe = get_pipeline(model, threshold, min_pixels)
    res  = pipe.run_from_array(img)

    out_img = res["overlay_image"] if output == "overlay" else res["heatmap_image"]

    _, buf = cv2.imencode(".jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type="image/jpeg",
        headers={"Content-Disposition": f'inline; filename="result_{output}.jpg"'},
    )