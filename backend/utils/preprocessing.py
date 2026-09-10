"""
SMART AGRI AI - Preprocessing and Utility Functions
Provides input validation, crop metadata, image preprocessing, and SQLite logging.
"""

import sqlite3
import datetime
from pathlib import Path
from PIL import Image
import numpy as np
from backend.config import DB_PATH, ALLOWED_EXTENSIONS

# Crop agronomic metadata for rich advisory display
CROP_METADATA = {
    "rice": {
        "scientific_name": "Oryza sativa",
        "category": "Cereal / Food Grain",
        "ideal_soil": "Clayey or loamy soils with good water retention capacity",
        "growth_duration": "100 - 150 days",
        "advisory": "Maintain 2-5 cm standing water during vegetative stage. Apply nitrogen in split doses.",
        "icon": "🌾"
    },
    "maize": {
        "scientific_name": "Zea mays",
        "category": "Cereal / Coarse Grain",
        "ideal_soil": "Well-drained deep loamy or silt loam soils",
        "growth_duration": "90 - 120 days",
        "advisory": "Avoid waterlogging. Ensure adequate nitrogen and potassium during silking stage.",
        "icon": "🌽"
    },
    "cotton": {
        "scientific_name": "Gossypium hirsutum",
        "category": "Fiber / Cash Crop",
        "ideal_soil": "Deep black soils (Regur) or alluvial soils",
        "growth_duration": "150 - 180 days",
        "advisory": "Requires warm climate and bright sunshine during boll maturation. Manage bollworms effectively.",
        "icon": "☁️"
    },
    "jute": {
        "scientific_name": "Corchorus olitorius",
        "category": "Fiber Crop",
        "ideal_soil": "Alluvial soil of river deltas with high organic matter",
        "growth_duration": "120 - 150 days",
        "advisory": "Requires high humidity and abundant rainfall. Harvest at 50% flowering stage for best fiber strength.",
        "icon": "🌱"
    },
    "coffee": {
        "scientific_name": "Coffea arabica / canephora",
        "category": "Plantation / Beverage",
        "ideal_soil": "Rich, porous, volcanic or acidic loams (pH 5.5 - 6.5)",
        "growth_duration": "Perennial (3-4 years to first harvest)",
        "advisory": "Cultivate under shade canopy in hilly slopes to prevent soil erosion and sunburn.",
        "icon": "☕"
    },
    "apple": {
        "scientific_name": "Malus domestica",
        "category": "Fruit / Horticultural",
        "ideal_soil": "Deep loamy soils with pH 5.5 - 6.8",
        "growth_duration": "Perennial deciduous tree",
        "advisory": "Requires 800 - 1000 chill hours (<7°C). Prune trees during dormancy for optimum fruit set.",
        "icon": "🍎"
    },
    "banana": {
        "scientific_name": "Musa acuminata",
        "category": "Fruit / Commercial",
        "ideal_soil": "Rich, well-drained loamy soil with high organic content",
        "growth_duration": "10 - 12 months",
        "advisory": "Heavy consumer of potassium. Provide adequate irrigation and propping support for fruiting bunch.",
        "icon": "🍌"
    },
    "grapes": {
        "scientific_name": "Vitis vinifera",
        "category": "Fruit / Horticultural",
        "ideal_soil": "Well-drained sandy loam or clay loam",
        "growth_duration": "Perennial vine",
        "advisory": "Train vines on trellis or bower systems. Ensure proper fungicide schedule during monsoon.",
        "icon": "🍇"
    },
    "watermelon": {
        "scientific_name": "Citrullus lanatus",
        "category": "Cucurbit / Horticultural",
        "ideal_soil": "Warm, deep, well-drained sandy loams",
        "growth_duration": "75 - 90 days",
        "advisory": "Cultivate in raised beds with plastic mulching. Withhold irrigation 7 days before harvest for higher sweetness.",
        "icon": "🍉"
    },
    "muskmelon": {
        "scientific_name": "Cucumis melo",
        "category": "Cucurbit / Fruit",
        "ideal_soil": "Well-drained sandy loam with pH 6.0 - 7.0",
        "growth_duration": "80 - 100 days",
        "advisory": "Thrives in sunny, hot, and dry conditions during fruit ripening.",
        "icon": "🍈"
    },
    "orange": {
        "scientific_name": "Citrus sinensis",
        "category": "Citrus Fruit",
        "ideal_soil": "Light loamy or alluvial soils with good drainage",
        "growth_duration": "Perennial citrus",
        "advisory": "Avoid standing water around the root zone to prevent root rot (Phytophthora).",
        "icon": "🍊"
    },
    "papaya": {
        "scientific_name": "Carica papaya",
        "category": "Fruit / Quick Yielding",
        "ideal_soil": "Rich sandy loam with excellent drainage",
        "growth_duration": "9 - 11 months",
        "advisory": "Extremely susceptible to waterlogging and viral mosaic disease transmitted by aphids.",
        "icon": "🥭"
    },
    "coconut": {
        "scientific_name": "Cocos nucifera",
        "category": "Plantation / Oilseed",
        "ideal_soil": "Coastal sand, red sandy loam, or laterite soil",
        "growth_duration": "Perennial palm (5-6 years to yield)",
        "advisory": "Requires balanced NPK, magnesium, and adequate basin irrigation in summer.",
        "icon": "🥥"
    },
    "pomegranate": {
        "scientific_name": "Punica granatum",
        "category": "Fruit / Arid Zone",
        "ideal_soil": "Deep loamy or alluvial soils, tolerates slight salinity",
        "growth_duration": "Perennial shrub",
        "advisory": "Implement drip irrigation and fruit bagging to protect bolls against fruit borers.",
        "icon": "🍎"
    },
    "blackgram": {
        "scientific_name": "Vigna mungo",
        "category": "Pulse / Legume",
        "ideal_soil": "Heavier soils like black cotton soils and loams",
        "growth_duration": "70 - 85 days",
        "advisory": "Fixes atmospheric nitrogen through root nodule symbiosis. Ideal for crop rotation.",
        "icon": "🌱"
    },
    "chickpea": {
        "scientific_name": "Cicer arietinum",
        "category": "Pulse / Legume (Rabi)",
        "ideal_soil": "Well-drained clay loam with pH 6.0 - 7.5",
        "growth_duration": "90 - 110 days",
        "advisory": "Requires cool weather and moderate moisture. Highly susceptible to pod borer (Helicoverpa).",
        "icon": "🫘"
    },
    "kidneybeans": {
        "scientific_name": "Phaseolus vulgaris",
        "category": "Pulse / Legume",
        "ideal_soil": "Light rich loam with pH 5.5 - 6.5",
        "growth_duration": "90 - 120 days",
        "advisory": "Does not fix nitrogen as efficiently as other pulses; needs supplemental basal N.",
        "icon": "🫘"
    },
    "pigeonpeas": {
        "scientific_name": "Cajanus cajan",
        "category": "Pulse / Legume (Arhar/Tur)",
        "ideal_soil": "Deep sandy loam or clay loam with deep rooting profile",
        "growth_duration": "140 - 180 days",
        "advisory": "Excellent drought resistance. Intercropping with sorghum or pearl millet is recommended.",
        "icon": "🫘"
    },
    "mothbeans": {
        "scientific_name": "Vigna aconitifolia",
        "category": "Pulse / Arid Legume",
        "ideal_soil": "Sandy or light loamy soils",
        "growth_duration": "60 - 75 days",
        "advisory": "Exceptional drought-hardy cover crop that prevents wind erosion in arid tracts.",
        "icon": "🌱"
    },
    "mungbean": {
        "scientific_name": "Vigna radiata",
        "category": "Pulse / Green Gram",
        "ideal_soil": "Loam to sandy loam soil with good drainage",
        "growth_duration": "60 - 70 days",
        "advisory": "Short duration catch crop. Protect against Yellow Mosaic Virus using vector control.",
        "icon": "🫘"
    },
    "lentil": {
        "scientific_name": "Lens culinaris",
        "category": "Pulse / Legume",
        "ideal_soil": "Can grow on all soil types from light loams to heavy black soils",
        "growth_duration": "100 - 120 days",
        "advisory": "Rabi season crop. Inoculate seeds with Rhizobium culture before sowing.",
        "icon": "🫘"
    },
    "mango": {
        "scientific_name": "Mangifera indica",
        "category": "Fruit / King of Fruits",
        "ideal_soil": "Deep rich well-drained alluvial or red loams",
        "growth_duration": "Perennial orchard tree",
        "advisory": "Avoid irrigation during flowering stage to encourage abundant fruit setting.",
        "icon": "🥭"
    }
}

def get_crop_metadata(crop_name: str) -> dict:
    """Retrieve detailed agronomic profile for a predicted crop."""
    key = str(crop_name).strip().lower()
    return CROP_METADATA.get(key, {
        "scientific_name": "Agricultural Crop",
        "category": "Crop",
        "ideal_soil": "Standard fertile farm soil",
        "growth_duration": "Seasonal",
        "advisory": "Follow standard agronomic practices and maintain recommended nutrient management.",
        "icon": "🌱"
    })

def validate_crop_inputs(data: dict) -> dict:
    """
    Validates and standardizes numerical input parameters for crop prediction.
    Raises ValueError with descriptive guidance if inputs are missing or out of valid physiological bounds.
    """
    required_keys = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    parsed = {}
    
    for key in required_keys:
        if key not in data or data[key] is None or str(data[key]).strip() == "":
            raise ValueError(f"Missing required parameter: '{key}'")
        try:
            val = float(data[key])
        except (ValueError, TypeError):
            raise ValueError(f"Parameter '{key}' must be a valid number. Received: {data[key]}")
        parsed[key] = val

    # Agronomic range assertions
    if not (0 <= parsed["N"] <= 300):
        raise ValueError(f"Nitrogen (N) value of {parsed['N']} is out of physiological range (0 - 300 kg/ha).")
    if not (0 <= parsed["P"] <= 300):
        raise ValueError(f"Phosphorus (P) value of {parsed['P']} is out of physiological range (0 - 300 kg/ha).")
    if not (0 <= parsed["K"] <= 300):
        raise ValueError(f"Potassium (K) value of {parsed['K']} is out of physiological range (0 - 300 kg/ha).")
    if not (-10 <= parsed["temperature"] <= 60):
        raise ValueError(f"Temperature value of {parsed['temperature']}°C is out of biological bounds (-10°C to 60°C).")
    if not (0 <= parsed["humidity"] <= 100):
        raise ValueError(f"Relative humidity of {parsed['humidity']}% must be between 0% and 100%.")
    if not (0 <= parsed["ph"] <= 14):
        raise ValueError(f"Soil pH value of {parsed['ph']} must be between 0 and 14.")
    if not (0 <= parsed["rainfall"] <= 2000):
        raise ValueError(f"Rainfall of {parsed['rainfall']} mm must be between 0 and 2000 mm.")

    return parsed

def allowed_file(filename: str) -> bool:
    """Check if the uploaded image file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_file, target_size=(224, 224)) -> np.ndarray:
    """Preprocess image for CNN deep learning inference."""
    image = Image.open(image_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# SQLite Database Helper Functions
def init_db():
    """Initializes SQLite tables for logging predictions and advisory logs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS advisory_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            module TEXT NOT NULL,
            inputs TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_advisory(module: str, inputs_summary: str, prediction: str, confidence: float):
    """Logs a recommendation event to SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO advisory_logs (timestamp, module, inputs, prediction, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, (now, module, inputs_summary, prediction, float(confidence)))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Warning] Failed to log advisory to database: {e}")

def get_recent_advisories(limit=10) -> list:
    """Retrieves the most recent advisory logs."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, module, inputs, prediction, confidence
            FROM advisory_logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "timestamp": r[1],
                "module": r[2],
                "inputs": r[3],
                "prediction": r[4],
                "confidence": round(r[5] * 100, 1) if r[5] <= 1.0 else round(r[5], 1)
            })
        return results
    except Exception:
        return []
