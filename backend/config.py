"""
SMART AGRI AI - Backend Configuration
Defines paths, feature schemas, model artifacts, and environment settings.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
UPLOAD_DIR = BACKEND_DIR / "uploads"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Database Path
DB_PATH = BACKEND_DIR / "database.db"

# Crop Recommendation Configuration
CROP_DATA_PATH = DATA_DIR / "Crop_recommendation.csv"
CROP_MODEL_PATH = MODELS_DIR / "crop_model.joblib"
CROP_METRICS_PATH = MODELS_DIR / "crop_metrics.json"
CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Fertilizer Recommendation Configuration
FERTILIZER_DATA_PATH = DATA_DIR / "fertilizer_dataset" / "fertilizer_recommendation.csv"
FERTILIZER_MODEL_PATH = MODELS_DIR / "fertilizer_model.joblib"
FERTILIZER_METRICS_PATH = MODELS_DIR / "fertilizer_metrics.json"

# Plant Disease Detection Configuration
DISEASE_DATA_DIR = DATA_DIR / "disease_dataset"
DISEASE_MODEL_PATH = MODELS_DIR / "disease_model.h5"
DISEASE_CLASSES_PATH = MODELS_DIR / "disease_classes.json"
DISEASE_METRICS_PATH = MODELS_DIR / "disease_metrics.json"

# Allowed Upload Formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

# Server Defaults
HOST = "127.0.0.1"
PORT = 5000
DEBUG = True
