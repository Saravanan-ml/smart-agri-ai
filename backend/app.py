"""
SMART AGRI AI – AI-Based Smart Agriculture Decision Support System
Flask REST API & Static Server
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    FRONTEND_DIR,
    CROP_MODEL_PATH,
    CROP_METRICS_PATH,
    CROP_FEATURES,
    FERTILIZER_MODEL_PATH,
    FERTILIZER_METRICS_PATH,
    DISEASE_MODEL_PATH,
    DISEASE_CLASSES_PATH,
    DISEASE_METRICS_PATH,
    ALLOWED_EXTENSIONS,
    HOST,
    PORT,
    DEBUG
)
from backend.utils.preprocessing import (
    validate_crop_inputs,
    get_crop_metadata,
    allowed_file,
    preprocess_image,
    init_db,
    log_advisory,
    get_recent_advisories
)

# Initialize Flask application
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# Initialize SQLite database
init_db()

# Cached model references
_CROP_MODEL = None
_FERTILIZER_MODEL = None
_DISEASE_MODEL = None
_DISEASE_CLASSES = None

def get_crop_model():
    global _CROP_MODEL
    if _CROP_MODEL is None and CROP_MODEL_PATH.exists():
        try:
            _CROP_MODEL = joblib.load(CROP_MODEL_PATH)
        except Exception as e:
            app.logger.error(f"Failed to load crop model: {e}")
    return _CROP_MODEL

def get_fertilizer_model():
    global _FERTILIZER_MODEL
    if _FERTILIZER_MODEL is None and FERTILIZER_MODEL_PATH.exists():
        try:
            _FERTILIZER_MODEL = joblib.load(FERTILIZER_MODEL_PATH)
        except Exception as e:
            app.logger.error(f"Failed to load fertilizer model: {e}")
    return _FERTILIZER_MODEL

def get_disease_model():
    global _DISEASE_MODEL, _DISEASE_CLASSES
    if _DISEASE_MODEL is None and DISEASE_MODEL_PATH.exists():
        try:
            import tensorflow as tf
            _DISEASE_MODEL = tf.keras.models.load_model(str(DISEASE_MODEL_PATH))
            if DISEASE_CLASSES_PATH.exists():
                with open(DISEASE_CLASSES_PATH, "r") as f:
                    _DISEASE_CLASSES = json.load(f)
        except Exception as e:
            app.logger.error(f"Failed to load disease model: {e}")
    return _DISEASE_MODEL, _DISEASE_CLASSES


# --------------------------------------------------------------------------
# Frontend Page & Static File Routing
# --------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve home page."""
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def serve_static_page(path):
    """Serve HTML pages, CSS, JS, and assets."""
    file_path = FRONTEND_DIR / path
    if file_path.exists() and file_path.is_file():
        return send_from_directory(FRONTEND_DIR, path)
    
    # Try adding .html if not provided
    html_path = FRONTEND_DIR / f"{path}.html"
    if html_path.exists() and html_path.is_file():
        return send_from_directory(FRONTEND_DIR, f"{path}.html")

    return send_from_directory(FRONTEND_DIR, "index.html")


# --------------------------------------------------------------------------
# REST API Endpoints
# --------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """System and models health status."""
    crop_ready = CROP_MODEL_PATH.exists()
    fert_ready = FERTILIZER_MODEL_PATH.exists()
    disease_ready = DISEASE_MODEL_PATH.exists()

    return jsonify({
        "status": "healthy",
        "system": "SMART AGRI AI – Decision Support System",
        "version": "1.0.0",
        "models": {
            "crop_recommendation": {
                "status": "ready" if crop_ready else "not_trained",
                "model_file": str(CROP_MODEL_PATH.name),
                "ready": crop_ready
            },
            "fertilizer_recommendation": {
                "status": "ready" if fert_ready else "not_trained",
                "model_file": str(FERTILIZER_MODEL_PATH.name),
                "ready": fert_ready
            },
            "disease_detection": {
                "status": "ready" if disease_ready else "not_trained",
                "model_file": str(DISEASE_MODEL_PATH.name),
                "ready": disease_ready
            }
        }
    })


@app.route("/api/predict-crop", methods=["POST"])
def predict_crop():
    """
    Predict optimal crop given soil and environmental conditions.
    Accepts JSON: { N, P, K, temperature, humidity, ph, rainfall }
    """
    model = get_crop_model()
    if model is None:
        return jsonify({
            "success": false,
            "error": "Crop model is not trained yet. Please add the dataset and run the training script: 'python backend/training/train_crop_model.py'",
            "model_status": "not_trained"
        }), 200

    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({
                "success": False,
                "error": "Request body must be a valid JSON object containing soil and environmental parameters."
            }), 400

        # Validate physiological inputs
        validated = validate_crop_inputs(data)

        # Build feature vector DataFrame to maintain feature name alignment
        input_df = pd.DataFrame([{
            "N": validated["N"],
            "P": validated["P"],
            "K": validated["K"],
            "temperature": validated["temperature"],
            "humidity": validated["humidity"],
            "ph": validated["ph"],
            "rainfall": validated["rainfall"]
        }])[CROP_FEATURES]

        # Predict using trained model
        pred_label = model.predict(input_df)[0]
        classes = model.classes_

        # Calculate class probabilities
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(input_df)[0]
            top_indices = np.argsort(probas)[::-1][:5]
            
            top_predictions = []
            for idx in top_indices:
                crop_name = classes[idx]
                prob = float(probas[idx])
                top_predictions.append({
                    "crop": crop_name,
                    "crop_display": crop_name.title(),
                    "probability": round(prob, 4),
                    "percentage": round(prob * 100, 1),
                    "icon": get_crop_metadata(crop_name).get("icon", "🌱")
                })
            
            best_confidence = float(probas[classes == pred_label][0])
        else:
            best_confidence = 1.0
            top_predictions = [{
                "crop": pred_label,
                "crop_display": pred_label.title(),
                "probability": 1.0,
                "percentage": 100.0,
                "icon": get_crop_metadata(pred_label).get("icon", "🌱")
            }]

        crop_metadata = get_crop_metadata(pred_label)

        # Log to SQLite
        input_summary = f"N:{validated['N']}, P:{validated['P']}, K:{validated['K']}, T:{validated['temperature']}C, H:{validated['humidity']}%, pH:{validated['ph']}, Rain:{validated['rainfall']}mm"
        log_advisory("Crop Recommendation", input_summary, pred_label.title(), best_confidence)

        return jsonify({
            "success": True,
            "prediction": pred_label,
            "prediction_display": pred_label.title(),
            "confidence": round(best_confidence, 4),
            "confidence_pct": round(best_confidence * 100, 1),
            "metadata": crop_metadata,
            "top_predictions": top_predictions,
            "inputs": validated
        })

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        app.logger.exception("Error during crop prediction")
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/predict-fertilizer", methods=["POST"])
def predict_fertilizer():
    """
    Recommend optimal fertilizer based on crop and soil parameters.
    Accepts JSON: { crop, soil_type, N, P, K }
    """
    model = get_fertilizer_model()
    if model is None:
        return jsonify({
            "success": False,
            "error": "Fertilizer model is not trained yet. Please run: 'python backend/training/train_fertilizer_model.py'",
            "model_status": "not_trained"
        }), 200

    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Request body must be valid JSON."}), 400

        crop = str(data.get("crop", "")).strip().title()
        soil_type = str(data.get("soil_type", "")).strip().title()
        
        if not crop:
            return jsonify({"success": False, "error": "Crop type is required."}), 400
        if not soil_type:
            return jsonify({"success": False, "error": "Soil type is required."}), 400

        try:
            n_val = float(data.get("N", 0))
            p_val = float(data.get("P", 0))
            k_val = float(data.get("K", 0))
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": "N, P, and K must be valid numbers."}), 400

        input_df = pd.DataFrame([{
            "crop": crop,
            "soil_type": soil_type,
            "N": n_val,
            "P": p_val,
            "K": k_val
        }])

        pred_fertilizer = model.predict(input_df)[0]
        confidence = 0.85
        top_predictions = []

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(input_df)[0]
            top_idx = np.argsort(probas)[::-1][:3]
            for idx in top_idx:
                f_name = model.classes_[idx]
                p = float(probas[idx])
                top_predictions.append({
                    "fertilizer": f_name,
                    "probability": round(p, 4),
                    "percentage": round(p * 100, 1)
                })
            confidence = float(probas[model.classes_ == pred_fertilizer][0])

        # Agronomic advisory notes
        advisory_notes = {
            "Urea": "High nitrogen source (46% N). Promote leaf growth and chlorophyll formation. Apply in split doses during vegetative stages.",
            "DAP": "Di-Ammonium Phosphate (18% N, 46% P2O5). Excellent starter fertilizer for strong root establishment and early seedling vigor.",
            "MOP": "Muriate of Potash (60% K2O). Enhances disease resistance, water regulation, and improves grain/fruit filling quality.",
            "14-35-14": "High phosphorus balanced complex. Ideal for basal application in root crops and legumes.",
            "28-28": "High nitrogen & phosphorus complex. Accelerates tillering and canopy expansion in cereals.",
            "17-17-17": "Balanced NPK complex for all-round plant nutrition and sustained flowering/fruiting.",
            "20-20": "Equal proportion NP fertilizer suited for soils with medium potassium availability."
        }

        note = advisory_notes.get(pred_fertilizer, "Apply recommended dosage following agricultural extension guidelines.")

        # Log to SQLite
        summary = f"Crop:{crop}, Soil:{soil_type}, N:{n_val}, P:{p_val}, K:{k_val}"
        log_advisory("Fertilizer Recommendation", summary, pred_fertilizer, confidence)

        return jsonify({
            "success": True,
            "prediction": pred_fertilizer,
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100, 1),
            "advisory": note,
            "top_predictions": top_predictions
        })

    except Exception as e:
        app.logger.exception("Error during fertilizer prediction")
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/predict-disease", methods=["POST"])
def predict_disease():
    """
    Classify plant disease from uploaded leaf image using CNN.
    Accepts multipart/form-data with 'image' file.
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in upload request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No image selected for upload."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Invalid file extension. Please upload a valid image file (.jpg, .jpeg, or .png)."
        }), 400

    # Inspect image size and dimensions
    try:
        from PIL import Image
        img = Image.open(file)
        width, height = img.size
        img_format = img.format
        file.seek(0)  # reset stream position
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to parse uploaded image: {str(e)}"}), 400

    model, class_names = get_disease_model()

    # If CNN model has not been trained yet, return clear, transparent explanation
    if model is None or class_names is None:
        return jsonify({
            "success": False,
            "model_status": "not_trained",
            "error": "Plant Disease CNN model is not trained yet. Please place plant leaf datasets in 'data/disease_dataset/' and run 'python backend/training/train_disease_model.py'.",
            "file_info": {
                "filename": file.filename,
                "dimensions": f"{width}x{height} px",
                "format": img_format
            }
        }), 200

    try:
        processed_img = preprocess_image(file, target_size=(224, 224))
        preds = model.predict(processed_img)[0]
        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])
        disease_name = class_names[top_idx]

        clean_name = disease_name.replace("___", " - ").replace("_", " ")

        log_advisory("Plant Disease Detection", f"File: {file.filename} ({width}x{height})", clean_name, confidence)

        return jsonify({
            "success": True,
            "prediction": clean_name,
            "raw_class": disease_name,
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100, 1),
            "file_info": {
                "filename": file.filename,
                "dimensions": f"{width}x{height} px",
                "format": img_format
            }
        })
    except Exception as e:
        app.logger.exception("Error during leaf disease classification")
        return jsonify({"success": False, "error": f"CNN classification failed: {str(e)}"}), 500


@app.route("/api/dashboard", methods=["GET"])
def get_dashboard_data():
    """
    Returns authentic training metrics and active status for all models.
    Does NOT fabricate accuracy values.
    """
    dashboard = {
        "models": {
            "crop": {
                "name": "Crop Recommendation",
                "algorithm": "Random Forest Classifier",
                "ready": CROP_MODEL_PATH.exists(),
                "status": "Ready" if CROP_MODEL_PATH.exists() else "Not Trained",
                "metrics": None
            },
            "fertilizer": {
                "name": "Fertilizer Recommendation",
                "algorithm": "Random Forest Pipeline",
                "ready": FERTILIZER_MODEL_PATH.exists(),
                "status": "Ready" if FERTILIZER_MODEL_PATH.exists() else "Not Trained",
                "metrics": None
            },
            "disease": {
                "name": "Plant Disease Detection",
                "algorithm": "Deep Convolutional Neural Network (CNN)",
                "ready": DISEASE_MODEL_PATH.exists(),
                "status": "Ready" if DISEASE_MODEL_PATH.exists() else "Not Trained",
                "metrics": None
            }
        },
        "recent_advisories": get_recent_advisories(limit=10)
    }

    # Load authentic crop metrics if available
    if CROP_METRICS_PATH.exists():
        try:
            with open(CROP_METRICS_PATH, "r") as f:
                dashboard["models"]["crop"]["metrics"] = json.load(f)
        except Exception:
            pass

    # Load authentic fertilizer metrics if available
    if FERTILIZER_METRICS_PATH.exists():
        try:
            with open(FERTILIZER_METRICS_PATH, "r") as f:
                dashboard["models"]["fertilizer"]["metrics"] = json.load(f)
        except Exception:
            pass

    # Load authentic disease metrics if available
    if DISEASE_METRICS_PATH.exists():
        try:
            with open(DISEASE_METRICS_PATH, "r") as f:
                dashboard["models"]["disease"]["metrics"] = json.load(f)
        except Exception:
            pass

    return jsonify(dashboard)


@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns SQLite recent decision support logs."""
    limit = request.args.get("limit", 15, type=int)
    logs = get_recent_advisories(limit=limit)
    return jsonify({"success": True, "logs": logs})


# --------------------------------------------------------------------------
# Error Handlers
# --------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Endpoint not found"}), 404
    return send_from_directory(FRONTEND_DIR, "index.html"), 200

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error occurred"}), 500


if __name__ == "__main__":
    print(f"Starting SMART AGRI AI Server at http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
