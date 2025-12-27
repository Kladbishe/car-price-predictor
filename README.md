# 🚗 Car Price Predictor

AI-powered car price prediction system for the Israeli market using Random Forest ML.

## Features

- **AI Price Prediction** - Random Forest model with 96.5% accuracy (R² = 0.965)
- **30 Brands, 200+ Models** - Comprehensive database of 5,000 cars
- **Hebrew UI** - RTL interface with modern gradient design
- **Real-time API** - RESTful Flask backend

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training data
python sample_data.py

# Train the model
python price_predictor.py

# Start the server
python app.py
```

Open http://localhost:5001

## Tech Stack

- **Backend**: Python, Flask, scikit-learn, pandas
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **ML Model**: Random Forest Regressor (200 trees, depth 25)

## Dataset

- **Size**: 5,000 cars
- **Brands**: Toyota, Mazda, Honda, BMW, Mercedes, Audi, Tesla, and 23 more
- **Features**: Brand, Model, Year, Mileage, Owners, Engine, City, Rarity

## Model Performance

```
R² Score: 0.965
MAE: ₪14,885
RMSE: ₪21,654
```

**Feature Importance:**
- Rarity Score: 36.6%
- Year: 22.7%
- Mileage: 19.5%
- Owners (Hand): 7.9%

## API Endpoints

```
GET  /api/brands              - Get all brands
GET  /api/models/:brand       - Get models by brand
GET  /api/years/:brand/:model - Get years for model
POST /api/predict             - Predict car price
GET  /api/stats               - Database statistics
```

## License

MIT
