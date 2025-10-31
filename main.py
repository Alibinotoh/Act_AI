"""
Image Classification API using Pre-trained MobileNet Model
Model: google/mobilenet_v2_1.0_224 (optimized for low memory)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
from PIL import Image
import io
import uvicorn
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="AI-powered image classification using Google's Vision Transformer",
    version="1.0.0"
)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the model
classifier = None

@app.on_event("startup")
async def load_model():
    """Load the pre-trained model on startup"""
    global classifier
    print("🔄 Loading pre-trained model...")
    print("⏳ This may take 1-2 minutes on first run (downloading model)...")
    
    # Use a lighter model that works with 512MB RAM (MobileNet)
    # This model is smaller (~14MB) and more memory efficient
    classifier = pipeline("image-classification", model="google/mobilenet_v2_1.0_224")
    
    print("✅ Model loaded successfully!")
    print("🚀 API is ready to classify images!")


@app.get("/")
async def root():
    """Serve the custom UI"""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    else:
        # Fallback to API info if HTML not found
        return {
            "message": "Image Classification API",
            "model": "google/mobilenet_v2_1.0_224",
            "status": "running",
            "endpoints": {
                "/classify": "POST - Upload an image to classify",
                "/health": "GET - Check API health status",
                "/docs": "GET - Interactive API documentation"
            }
        }


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Image Classification API",
        "model": "google/mobilenet_v2_1.0_224",
        "status": "running",
        "endpoints": {
            "/classify": "POST - Upload an image to classify",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify an uploaded image
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with top predictions and confidence scores
    """
    
    # Check if model is loaded
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait...")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary (handles PNG with alpha channel, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Classify the image (returns top 5 predictions by default)
        predictions = classifier(image)
        
        # Format response
        return {
            "success": True,
            "filename": file.filename,
            "predictions": [
                {
                    "label": pred["label"],
                    "confidence": round(pred["score"] * 100, 2),  # Convert to percentage
                    "score": round(pred["score"], 4)
                }
                for pred in predictions
            ],
            "top_prediction": {
                "label": predictions[0]["label"],
                "confidence": round(predictions[0]["score"] * 100, 2)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/classify-top")
async def classify_image_top_only(file: UploadFile = File(...)):
    """
    Classify an uploaded image and return only the top prediction
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with only the top prediction
    """
    
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait...")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        predictions = classifier(image, top_k=1)  # Get only top prediction
        
        return {
            "success": True,
            "filename": file.filename,
            "label": predictions[0]["label"],
            "confidence": round(predictions[0]["score"] * 100, 2),
            "score": round(predictions[0]["score"], 4)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Image Classification API")
    print("=" * 60)
    print("📦 Model: google/mobilenet_v2_1.0_224")
    print("🌐 Custom UI: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔧 API Info: http://localhost:8000/api")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
