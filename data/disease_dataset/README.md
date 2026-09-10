# Plant Disease Dataset Directory

Place your plant leaf disease image datasets here for training the Convolutional Neural Network (CNN).

## Recommended Directory Structure:
```
disease_dataset/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   └── image2.jpg
├── Apple___Black_rot/
├── Corn_(maize)___Common_rust_/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Tomato___Bacterial_spot/
├── Tomato___Early_blight/
└── Tomato___healthy/
```

Supported formats: `.jpg`, `.jpeg`, `.png`.
Standard benchmarks: PlantVillage dataset or Kaggle New Plant Diseases Dataset.

After copying images into class subdirectories, train the CNN model:
```bash
python backend/training/train_disease_model.py
```
This will generate `backend/models/disease_model.h5` and class labels `backend/models/disease_classes.json`.
