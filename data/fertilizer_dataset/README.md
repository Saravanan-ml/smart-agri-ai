# Fertilizer Recommendation Dataset Directory

Contains agricultural soil and fertilizer datasets for training fertilizer recommendation models.

Default dataset included:
- `fertilizer_recommendation.csv`

## Features:
- Soil_Type (Clay, Silt, Sandy, Loamy, etc.)
- Nitrogen_Level (N)
- Phosphorus_Level (P)
- Potassium_Level (K)
- Crop_Type (Cotton, Maize, Wheat, Rice, Tomato, etc.)
- Recommended_Fertilizer (Urea, DAP, 14-35-14, 28-28, MOP, 17-17-17, etc.)

Train the model:
```bash
python backend/training/train_fertilizer_model.py
```
This saves `backend/models/fertilizer_model.joblib` and evaluation metrics.
