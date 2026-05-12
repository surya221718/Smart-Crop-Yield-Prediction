# fixed_app.py (XGBoost Version)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd
from xgboost import XGBRegressor
import joblib
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


class YieldPredictor:
    def __init__(self):
        self.model = None
        self.df = None
        self.load_data()
        self.train_model()

    def load_data(self):
        """Load and preprocess the crop yield data"""
        self.df = pd.read_csv('crop_yield_data.csv')

        self.df = self.df[
            ['State', 'Crop Type', 'Rainfall',
             'Temperature', 'Soil Type', 'Yield']
        ]

        self.df = pd.get_dummies(
            self.df,
            columns=['State', 'Crop Type', 'Soil Type']
        )

    def train_model(self):
        """Train or load the prediction model"""

        model_path = 'yield_model_xgb.pkl'

        # Features and target
        X = self.df.drop('Yield', axis=1)
        y = self.df['Yield']

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Load existing model or train new one
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)

        else:
            self.model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )

            # Train model
            self.model.fit(X_train, y_train)

            # Save model
            joblib.dump(self.model, model_path)

        # Predict test data
        y_pred = self.model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print("\n📊 MODEL PERFORMANCE")
        print(f"✅ R2 Score : {r2:.2f}")
        print(f"✅ MAE      : {mae:.2f}")
        print(f"✅ RMSE     : {rmse:.2f}")

# Initialize predictor
predictor = YieldPredictor()


@app.route('/predict', methods=['POST'])
def predict():
    print("🔮 Prediction route hit!")

    try:
        data = request.get_json() or request.form
        print(f"📥 Raw input data: {data}")

        state = data.get('state')
        crop = data.get('crop')
        temperature = float(data.get('temperature'))
        rainfall = float(data.get('rainfall'))
        soil_type = data.get('soil_type')
        print(f"🧾 Parsed inputs: state={state}, crop={crop}, temp={temperature}, rainfall={rainfall}, soil={soil_type}")

        # Prepare input dict
        input_data = {
            'Rainfall': [rainfall],
            'Temperature': [temperature],
        }

        # One-hot encode
        for col in predictor.df.columns:
            if col.startswith('State_'):
                input_data[col] = [1 if col == f'State_{state}' else 0]
            elif col.startswith('Crop Type_'):
                input_data[col] = [1 if col == f'Crop Type_{crop}' else 0]
            elif col.startswith('Soil Type_'):
                input_data[col] = [1 if col == f'Soil Type_{soil_type}' else 0]

        input_df = pd.DataFrame(input_data)
        print("✅ Input DataFrame BEFORE reindex:\n", input_df)

        expected_columns = predictor.df.drop('Yield', axis=1).columns
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)

        print("✅ Input DataFrame AFTER reindex:\n", input_df)

        prediction = predictor.model.predict(input_df)[0]
        print("✅ Prediction done:", prediction)
        return jsonify({'yield': float(round(prediction, 2))})


    except Exception as e:
        print(f"❌ ERROR during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500


if __name__ == '__main__':
    print("🚀 Starting Flask server on http://127.0.0.1:5001 ...")
    app.run(debug=True, port=5001)
