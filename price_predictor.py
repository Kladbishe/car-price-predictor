import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CarPricePredictor:
    def __init__(self):
        self.model = None
        self.brand_encoder = LabelEncoder()
        self.model_encoder = LabelEncoder()
        self.city_encoder = LabelEncoder()
        self.feature_columns = ['year', 'mileage_km', 'hand', 'engine_capacity']
        self.categorical_columns = ['brand', 'model', 'city']

        # Model rarity scores (same as in sample_data.py)
        self.model_rarity = {
            'Corolla': 1.0, 'Yaris': 1.0, 'Civic': 1.0, 'Golf': 1.0, 'Polo': 1.0,
            'Camry': 1.12, 'RAV4': 1.18, 'CR-V': 1.18, 'Tucson': 1.12, 'Qashqai': 1.08,
            'CX-5': 1.25, 'CX-30': 1.2, '3': 1.05, '6': 1.15, '323F': 0.85,
            'Land Cruiser': 1.5, 'Prius': 1.25, 'Highlander': 1.35, 'Pajero': 1.3,
            '3 Series': 1.4, '5 Series': 1.5, 'X3': 1.45, 'X5': 1.55, 'X1': 1.35,
            'C-Class': 1.4, 'E-Class': 1.5, 'GLC': 1.45, 'GLE': 1.55, 'A-Class': 1.3,
            'A4': 1.4, 'A6': 1.5, 'Q5': 1.45, 'Q7': 1.55, 'A3': 1.3, 'Q3': 1.35,
            'XC40': 1.35, 'XC60': 1.45, 'XC90': 1.55, 'V60': 1.3, 'S60': 1.35,
            'NX': 1.45, 'RX': 1.55, 'ES': 1.5, 'IS': 1.4,
            'Stinger': 1.35, 'Mustang': 1.45, 'MX-5': 1.35,
            'Tiguan': 1.15, 'Passat': 1.12, 'Touareg': 1.4,
            # New brands
            'Corsa': 0.95, 'Astra': 1.0, 'Insignia': 1.05,
            '500': 1.1, 'Panda': 0.95, 'Tipo': 0.95,
            'Renegade': 1.15, 'Compass': 1.2, 'Cherokee': 1.35, 'Grand Cherokee': 1.45, 'Wrangler': 1.5,
            'Cooper': 1.25, 'Countryman': 1.35,
            '911': 2.0, 'Cayenne': 1.7, 'Macan': 1.6, 'Panamera': 1.8, 'Taycan': 1.9,
            'Model 3': 1.6, 'Model Y': 1.65, 'Model S': 1.8, 'Model X': 1.85,
            'Range Rover': 1.8, 'Range Rover Sport': 1.75, 'Discovery': 1.6, 'Defender': 1.7, 'Evoque': 1.5,
            'XE': 1.4, 'XF': 1.5, 'F-Pace': 1.55, 'E-Pace': 1.45, 'I-Pace': 1.65
        }

    def load_data(self, csv_file: str) -> pd.DataFrame:
        """
        Load car data from CSV file

        Args:
            csv_file: Path to CSV file

        Returns:
            DataFrame with car data
        """
        logger.info(f"Loading data from {csv_file}")
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} records")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Preprocessing data...")

        # Remove rows with missing price (target variable)
        df = df.dropna(subset=['price'])

        # Remove outliers (prices that are too low or too high)
        df = df[(df['price'] >= 5000) & (df['price'] <= 500000)]

        # Fill missing values for numerical columns with median
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Fill missing values for categorical columns with 'Unknown'
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')

        # Remove duplicates
        df = df.drop_duplicates()

        logger.info(f"After preprocessing: {len(df)} records")
        return df

    def encode_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Encode categorical features

        Args:
            df: DataFrame to encode
            is_training: Whether this is training data (fit encoders) or prediction data (transform only)

        Returns:
            DataFrame with encoded features
        """
        df = df.copy()

        if is_training:
            df['brand_encoded'] = self.brand_encoder.fit_transform(df['brand'])
            df['model_encoded'] = self.model_encoder.fit_transform(df['model'])
            if 'city' in df.columns:
                df['city_encoded'] = self.city_encoder.fit_transform(df['city'])
        else:
            # For prediction, handle unknown categories
            df['brand_encoded'] = df['brand'].apply(
                lambda x: self.brand_encoder.transform([x])[0] if x in self.brand_encoder.classes_ else -1
            )
            df['model_encoded'] = df['model'].apply(
                lambda x: self.model_encoder.transform([x])[0] if x in self.model_encoder.classes_ else -1
            )
            if 'city' in df.columns:
                df['city_encoded'] = df['city'].apply(
                    lambda x: self.city_encoder.transform([x])[0] if x in self.city_encoder.classes_ else -1
                )

        return df

    def train_model(self, csv_file: str, test_size: float = 0.2):
        """
        Train the price prediction model

        Args:
            csv_file: Path to CSV file with car data
            test_size: Proportion of data to use for testing
        """
        # Load and preprocess data
        df = self.load_data(csv_file)
        df = self.preprocess_data(df)

        # Encode categorical features
        df = self.encode_features(df, is_training=True)

        # Prepare features and target
        feature_cols = ['brand_encoded', 'model_encoded', 'year', 'mileage_km', 'hand']

        # Add optional features if available
        if 'engine_capacity' in df.columns:
            feature_cols.append('engine_capacity')
        if 'city_encoded' in df.columns:
            feature_cols.append('city_encoded')
        if 'rarity_score' in df.columns:
            feature_cols.append('rarity_score')

        X = df[feature_cols]
        y = df['price']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        # Train Random Forest model with improved parameters
        logger.info("Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=200,  # More trees for better accuracy
            max_depth=25,      # Deeper trees to capture mileage nuances
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        logger.info(f"\nModel Performance:")
        logger.info(f"Mean Absolute Error: ₪{mae:,.0f}")
        logger.info(f"Root Mean Squared Error: ₪{rmse:,.0f}")
        logger.info(f"R² Score: {r2:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"\nFeature Importance:\n{feature_importance}")

        return self.model

    def predict_price(self, brand: str, model: str, year: int,
                     mileage_km: int = 100000, hand: int = 1,
                     engine_capacity: int = 1600, city: str = 'Unknown') -> float:
        """
        Predict car price based on features

        Args:
            brand: Car brand (e.g., 'Volvo')
            model: Car model (e.g., 'XC40')
            year: Manufacturing year
            mileage_km: Mileage in kilometers
            hand: Owner number (1 = first owner, 2 = second, etc.)
            engine_capacity: Engine capacity in cc
            city: City location

        Returns:
            Predicted price in NIS
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        # Get rarity score for this model
        rarity_score = self.model_rarity.get(model, 1.0)

        # Create DataFrame with input data
        input_data = pd.DataFrame({
            'brand': [brand],
            'model': [model],
            'year': [year],
            'mileage_km': [mileage_km],
            'hand': [hand],
            'engine_capacity': [engine_capacity],
            'city': [city],
            'rarity_score': [rarity_score]
        })

        # Encode features
        input_encoded = self.encode_features(input_data, is_training=False)

        # Prepare features in same order as training
        feature_cols = ['brand_encoded', 'model_encoded', 'year', 'mileage_km', 'hand',
                       'engine_capacity', 'city_encoded', 'rarity_score']

        X = input_encoded[feature_cols]

        # Predict
        predicted_price = self.model.predict(X)[0]

        return max(0, predicted_price)  # Ensure non-negative price

    def save_model(self, filename: str = 'car_price_model.pkl'):
        """
        Save trained model and encoders to file

        Args:
            filename: Name of the file to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")

        model_data = {
            'model': self.model,
            'brand_encoder': self.brand_encoder,
            'model_encoder': self.model_encoder,
            'city_encoder': self.city_encoder
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filename}")

    def load_model(self, filename: str = 'car_price_model.pkl'):
        """
        Load trained model and encoders from file

        Args:
            filename: Name of the file to load the model from
        """
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.brand_encoder = model_data['brand_encoder']
        self.model_encoder = model_data['model_encoder']
        self.city_encoder = model_data['city_encoder']

        logger.info(f"Model loaded from {filename}")


def main():
    """
    Main function to train and test the model
    """
    predictor = CarPricePredictor()

    # Train model
    predictor.train_model('cars_data.csv')

    # Save model
    predictor.save_model('car_price_model.pkl')

    # Test prediction
    print("\n" + "="*50)
    print("Testing predictions:")
    print("="*50)

    test_cases = [
        {'brand': 'Volvo', 'model': 'XC40', 'year': 2020, 'mileage_km': 50000, 'hand': 1},
        {'brand': 'Toyota', 'model': 'Corolla', 'year': 2018, 'mileage_km': 80000, 'hand': 2},
        {'brand': 'Mazda', 'model': 'CX-5', 'year': 2019, 'mileage_km': 60000, 'hand': 1},
    ]

    for case in test_cases:
        price = predictor.predict_price(**case)
        print(f"\n{case['brand']} {case['model']} ({case['year']}):")
        print(f"  Mileage: {case['mileage_km']:,} km, Hand: {case['hand']}")
        print(f"  Predicted Price: ₪{price:,.0f}")


if __name__ == "__main__":
    main()
