from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from price_predictor import CarPricePredictor
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize predictor
predictor = CarPricePredictor()

# Load model if exists, otherwise train it
MODEL_FILE = 'car_price_model.pkl'
DATA_FILE = 'cars_data.csv'

if os.path.exists(MODEL_FILE):
    logger.info("Loading existing model...")
    predictor.load_model(MODEL_FILE)
elif os.path.exists(DATA_FILE):
    logger.info("Training new model...")
    predictor.train_model(DATA_FILE)
    predictor.save_model(MODEL_FILE)
else:
    logger.warning("No model or data file found. Please run scraper first.")


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/brands', methods=['GET'])
def get_brands():
    """Get list of available car brands"""
    try:
        if not hasattr(predictor.brand_encoder, 'classes_'):
            return jsonify({'error': 'Model not trained yet'}), 400

        brands = sorted(predictor.brand_encoder.classes_.tolist())
        return jsonify({'brands': brands})
    except Exception as e:
        logger.error(f"Error getting brands: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<brand>', methods=['GET'])
def get_models(brand):
    """Get list of car models for a specific brand"""
    try:
        if not os.path.exists(DATA_FILE):
            return jsonify({'error': 'Data file not found'}), 400

        df = pd.read_csv(DATA_FILE)
        models = sorted(df[df['brand'] == brand]['model'].unique().tolist())

        return jsonify({'models': models})
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/years/<brand>/<model>', methods=['GET'])
def get_years(brand, model):
    """Get list of available years for a specific brand and model"""
    try:
        if not os.path.exists(DATA_FILE):
            return jsonify({'error': 'Data file not found'}), 400

        df = pd.read_csv(DATA_FILE)
        years = sorted(
            df[(df['brand'] == brand) & (df['model'] == model)]['year'].dropna().unique().tolist(),
            reverse=True
        )
        years = [int(year) for year in years]

        return jsonify({'years': years})
    except Exception as e:
        logger.error(f"Error getting years: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_price():
    """
    Predict car price based on input parameters

    Expected JSON input:
    {
        "brand": "Volvo",
        "model": "XC40",
        "year": 2020,
        "mileage_km": 50000,
        "hand": 1,
        "engine_capacity": 1600,
        "city": "Tel Aviv"
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['brand', 'model', 'year']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Extract parameters with defaults
        brand = data['brand']
        model = data['model']
        year = int(data['year'])
        mileage_km = int(data.get('mileage_km', 100000))
        hand = int(data.get('hand', 1))
        engine_capacity = int(data.get('engine_capacity', 1600))
        city = data.get('city', 'Unknown')

        # Validate year
        current_year = 2025
        if year < 1990 or year > current_year:
            return jsonify({'error': f'Year must be between 1990 and {current_year}'}), 400

        # Predict price
        predicted_price = predictor.predict_price(
            brand=brand,
            model=model,
            year=year,
            mileage_km=mileage_km,
            hand=hand,
            engine_capacity=engine_capacity,
            city=city
        )

        logger.info(f"Prediction: {brand} {model} {year} -> ₪{predicted_price:,.0f}")

        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'formatted_price': f"₪{predicted_price:,.0f}",
            'input': {
                'brand': brand,
                'model': model,
                'year': year,
                'mileage_km': mileage_km,
                'hand': hand,
                'engine_capacity': engine_capacity,
                'city': city
            }
        })

    except Exception as e:
        logger.error(f"Error predicting price: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the dataset"""
    try:
        if not os.path.exists(DATA_FILE):
            return jsonify({'error': 'Data file not found'}), 400

        df = pd.read_csv(DATA_FILE)

        stats = {
            'total_cars': len(df),
            'brands_count': df['brand'].nunique(),
            'models_count': df['model'].nunique(),
            'year_range': {
                'min': int(df['year'].min()),
                'max': int(df['year'].max())
            },
            'price_range': {
                'min': float(df['price'].min()),
                'max': float(df['price'].max()),
                'average': float(df['price'].mean())
            },
            'top_brands': df['brand'].value_counts().head(10).to_dict()
        }

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_loaded = predictor.model is not None
    data_exists = os.path.exists(DATA_FILE)

    return jsonify({
        'status': 'healthy' if model_loaded else 'model not loaded',
        'model_loaded': model_loaded,
        'data_exists': data_exists
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
