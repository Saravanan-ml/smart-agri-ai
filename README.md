# 🌱 SMART AGRI AI – AI-Based Smart Agriculture Decision Support System

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![HTML5/CSS3](https://img.shields.io/badge/Frontend-HTML5%20%7C%20CSS3%20%7C%20JS-E34F26.svg)](https://developer.mozilla.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

> A full-stack, responsive, traditional web application designed as a college final-year engineering project. Converts machine learning and deep learning agronomic research into a practical decision-support platform for farmers and agricultural advisors.

---

## 🌟 Key Modules

1. **🌾 AI Crop Recommendation**:
   - Predicts the scientifically best-suited crop out of **22 cultivars** based on soil chemistry (Nitrogen, Phosphorus, Potassium) and climatic variables (Temperature, Humidity, Soil pH, Rainfall).
   - Powered by a genuine **Random Forest Classifier** trained on 2,200 empirical field records, achieving **99.55% test accuracy**.
   - Outputs primary recommendation, confidence percentage, agronomic care instructions, and a ranked top-5 suitable crops probability breakdown.

2. **🍃 Plant Disease Detection**:
   - Modern image upload interface supporting drag-and-drop file selection (JPG, JPEG, PNG) with instant dimension and file size analysis.
   - Deep Convolutional Neural Network (CNN) architecture ready for leaf pathology classification with graceful transparency when weights are awaiting dataset compilation.

3. **🧪 Fertilizer Recommendation**:
   - Formulates targeted fertilizer recommendations (Urea, DAP, MOP, Complex NPK) based on crop uptake needs, soil classification, and measured soil nutrient deficits.
   - Preprocessed scikit-learn pipeline with genuine evaluation metrics.

4. **📊 Agriculture Analytics Dashboard**:
   - Live model health status indicators (Ready vs. Not Trained).
   - Zero fabricated accuracy metrics: displays genuine test-split scores (Accuracy, Precision, Recall, F1-Score).
   - Chart.js visualizations for multi-model accuracy comparison and soil/climate feature importance (Gini impurity impact).
   - Interactive audit trail powered by SQLite logging every advisory query with timestamp, inputs, and confidence.

5. **📖 Academic Project Dossier (About)**:
   - Comprehensive documentation detailing the base paper inspiration, problem statement, mathematical ML/DL methodology, realized benefits, and future research scope (IoT, drone imagery, vernacular voice).

---

## 🏗️ Project Architecture

```
TN-AGRI/
├── frontend/                     # Traditional separated frontend
│   ├── index.html                # Home / Landing page
│   ├── crop.html                 # Crop recommendation form & results
│   ├── disease.html              # Plant disease leaf upload interface
│   ├── fertilizer.html           # Soil & crop fertilizer recommender
│   ├── dashboard.html            # Metrics, Chart.js graphs & SQLite logs
│   ├── about.html                # Academic base paper documentation
│   ├── css/
│   │   └── style.css             # Modern agriculture design system & responsive layout
│   └── js/
│       ├── main.js               # Mobile drawer, navigation & toast system
│       ├── crop.js               # Form validation, REST fetch & probability bars
│       ├── disease.js            # Drag-and-drop, image preview & CNN handling
│       ├── fertilizer.js         # Soil/crop validation & recommendation display
│       └── dashboard.js          # REST metrics fetch, charts & audit table
│
├── backend/                      # Python Flask REST API & ML pipelines
│   ├── app.py                    # Unified server serving API and static frontend
│   ├── config.py                 # Paths, parameters, allowed extensions
│   ├── requirements.txt          # Python dependencies
│   ├── models/                   # Serialized model weights & evaluation JSONs
│   │   ├── crop_model.joblib     # Trained Random Forest model
│   │   ├── crop_metrics.json     # Genuine test metrics & feature importances
│   │   ├── fertilizer_model.joblib # Trained Fertilizer model
│   │   └── fertilizer_metrics.json # Fertilizer evaluation metrics
│   ├── training/                 # Independent training scripts
│   │   ├── train_crop_model.py   # Crop model training pipeline
│   │   ├── train_fertilizer_model.py # Fertilizer model training pipeline
│   │   └── train_disease_model.py # CNN deep learning training pipeline
│   └── utils/
│       └── preprocessing.py      # Input validation, crop metadata & SQLite helpers
│
├── data/                         # Datasets
│   ├── Crop_recommendation.csv   # 2,200 agronomic field records
│   ├── fertilizer_dataset/       # Fertilizer recommendation dataset
│   └── disease_dataset/          # Plant leaf pathology image dataset structure
│
├── README.md                     # Comprehensive documentation
└── .gitignore                    # Version control exclusions
```

---

## 🚀 Getting Started (Windows & VS Code)

### Step 1: Open the Project in VS Code
Open VS Code, press `Ctrl + ~` to open the integrated terminal, and ensure your working directory is the project root:
```powershell
cd "d:\TN Agri"
```

### Step 2: Set Up Python Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r backend/requirements.txt
```

### Step 4: Train the Crop Model
Execute the Random Forest training pipeline on `data/Crop_recommendation.csv`:
```powershell
python backend/training/train_crop_model.py
```
*This will evaluate the model on an unseen 20% stratified test split and save `backend/models/crop_model.joblib` and `backend/models/crop_metrics.json`.*

### Step 5: (Optional) Train the Fertilizer Model
```powershell
python backend/training/train_fertilizer_model.py
```

### Step 6: Start the Flask Application
Run the unified Flask server (which serves both the REST API and the frontend):
```powershell
python backend/app.py
```

### Step 7: Open in Your Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📡 REST API Specification

| Method | Endpoint | Description | Sample Payload / Response |
|---|---|---|---|
| `GET` | `/api/health` | System and models status | Returns readiness status of all models |
| `POST` | `/api/predict-crop` | Predict optimal crop | `{"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}` |
| `POST` | `/api/predict-fertilizer` | Recommend fertilizer | `{"crop": "Cotton", "soil_type": "Clay", "N": 61, "P": 44, "K": 84}` |
| `POST` | `/api/predict-disease` | Classify leaf disease | Multipart Form-Data with `image` file |
| `GET` | `/api/dashboard` | Dashboard analytics | Returns genuine metrics, feature importances & logs |
| `GET` | `/api/history` | Query history | Recent SQLite advisory query records |

---

## 🎓 Viva & Presentation Talking Points

1. **Why Traditional Web Architecture over Streamlit?**
   - Streamlit re-runs the entire Python script on every user interaction, which introduces latency and lacks industrial design flexibility.
   - A traditional decoupled architecture with **HTML5, CSS3, Vanilla JS, and Flask REST API** reflects professional software engineering practices, supports clean RESTful API integration, and allows full control over responsive UI/UX and mobile interactions.

2. **Why Random Forest for Crop Recommendation?**
   - Agricultural datasets have non-linear interactions between soil nutrients and meteorological parameters.
   - Random Forest creates an ensemble of de-correlated decision trees with bootstrap aggregation, preventing overfitting and providing transparent feature importances (e.g., Rainfall and Humidity being the top predictive factors).

3. **Academic Integrity & Honest Model Evaluation**:
   - Unlike basic student projects that hardcode fake predictions or fabricate 99% accuracies, SMART AGRI AI computes all metrics strictly on unseen stratified test sets and stores them in JSON files for dashboard inspection.
   - When a model (such as the Disease CNN) has not yet been trained, the system transparently explains what dataset is needed instead of generating misleading fake results.

---

## 📄 License & Attribution
Developed for academic engineering presentation and demonstration purposes.
© 2026 Smart Agri AI | AI-Based Smart Agriculture Decision Support System.
