from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from ml.predictor import predict_age
import uvicorn
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lightweight Age Prediction API",
    description="An API that takes an image and returns the estimated age of the person's face.",
    version="1.0.0"
)

@app.post("/predict-age/")
async def predict_age_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid content type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_bytes = await file.read()
        logger.info(f"Received image: {file.filename}, size: {len(image_bytes)} bytes")
        
        result = predict_age(image_bytes)
        
        if "error" in result:
            return JSONResponse(status_code=400, content=result)
            
        return result
        
    except ValueError as ve:
        logger.error(f"Value error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fnfe:
        logger.error(f"File not found error: {str(fnfe)}")
        raise HTTPException(status_code=500, detail="Models missing. Please run download_models.py.")
    except Exception as e:
        logger.exception("Internal Server Error")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/")
def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "UI file not found. Ensure frontend/index.html exists."})

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running. Send a POST request with an image to /predict-age/"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
