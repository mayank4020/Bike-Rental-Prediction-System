# Bike Rental Prediction System 🚲

This project predicts hourly bike rental demand based on environmental and seasonal conditions using machine learning.

## Technologies Used
- Python
- Scikit-learn
- Flask

## How to Run

### 1. Train Model
```bash
python train_model.py
```

### 2. Run API
```bash
python app.py
```

### 3. Send a Prediction Request
```json
POST /predict
{
  "features": [1, 0, 1, 9, 0, 5, 1, 2, 0.5, 0.48, 0.6, 0.2]
}
```

## Dataset Used
UCI Machine Learning Repository - Bike Sharing Dataset
