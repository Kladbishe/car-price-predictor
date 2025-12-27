# 🚗 Car Price Predictor

AI-powered car price prediction system for the Israeli market using Random Forest ML.

## Features

- **AI Price Prediction** - Random Forest model with 98.3% accuracy (R² = 0.983)
- **30 Brands, 200+ Models** - Database of 15,000 synthetic cars
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

**Note: This project uses synthetic (generated) data, not real market data.**

- **Size**: 15,000 synthetic car records
- **Brands**: Toyota, Mazda, Honda, BMW, Mercedes, Audi, Tesla, and 23 more
- **Models**: 200+ car models across all brands
- **Features**: Brand, Model, Year, Mileage, Owners, Engine, City, Rarity Score
- **Price Calibration**: Generated using realistic depreciation curves and brand multipliers based on Israeli market observations

The synthetic data is designed to simulate realistic pricing patterns but may not reflect actual Yad2 market prices.

## Model Performance

```
R² Score: 0.983
MAE: ₪6,773
RMSE: ₪10,319
```

**Feature Importance:**
- Rarity Score: 39.5%
- Year: 21.6%
- Mileage: 17.6%
- Owners (Hand): 8.3%

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
