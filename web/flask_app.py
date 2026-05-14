from fastapi import FastAPI
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.middleware.cors import CORSMiddleware
from api_routes import router

app = FastAPI(
    title="Workout Analytics API",
    description="REST API for workout tracking and analytics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
def root():
    return {"status": "ok", "message": "Workout Analytics API running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# Run: uvicorn web.fastapi_backend:app --reload --port 8000