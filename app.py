# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import os
import json
from model import PregnancyRiskModel
from data.preprocess import PregnancyDataPreprocessor

app = Flask(__name__)
app.secret_key = 'pregnancy_risk_predictor_secret_key'

# Global variables
model = None
preprocessor = None

def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    global model, preprocessor
    
    model = PregnancyRiskModel()
    if os.path.exists('models/trained_model.pkl'):
        success = model.load_model('models/trained_model.pkl')
        if success:
            preprocessor = model.preprocessor
            print("Model and preprocessor loaded successfully")
            return True
    
    print("Model not found. Please train the model first.")
    return False

def get_risk_level_info(prediction, probabilities):
    """Get risk level information and recommendations"""
    # Assume prediction is encoded (0, 1, 2) for (Low, Medium, High)
    risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
    
    if hasattr(preprocessor, 'label_encoders'):
        # Check if target encoder exists
        target_encoders = ['RiskLevel', 'Risk', 'risk_level', 'risk', 'target', 'Target']
        target_encoder = None
        for encoder_name in target_encoders:
            if encoder_name in preprocessor.label_encoders:
                target_encoder = preprocessor.label_encoders[encoder_name]
                break
        
        if target_encoder:
            try:
                risk_level = target_encoder.inverse_transform([prediction])[0]
            except:
                risk_level = risk_levels[prediction] if prediction < len(risk_levels) else f"Risk Level {prediction}"
        else:
            risk_level = risk_levels[prediction] if prediction < len(risk_levels) else f"Risk Level {prediction}"
    else:
        risk_level = risk_levels[prediction] if prediction < len(risk_levels) else f"Risk Level {prediction}"
    
    # Risk level specific information
    risk_info = {
        'Low Risk': {
            'color': 'success',
            'message': 'Low risk pregnancy. Continue with regular prenatal care.',
            'recommendations': [
                'Maintain regular prenatal checkups',
                'Follow a healthy diet rich in folic acid, iron, and calcium',
                'Stay physically active as recommended by your doctor',
                'Avoid alcohol, smoking, and harmful substances',
                'Take prenatal vitamins as prescribed',
                'Monitor weight gain according to medical guidelines'
            ]
        },
        'Medium Risk': {
            'color': 'warning',
            'message': 'Medium risk pregnancy. Requires closer monitoring.',
            'recommendations': [
                'Schedule more frequent prenatal appointments',
                'Monitor blood pressure and blood sugar levels regularly',
                'Follow specific dietary restrictions if advised',
                'Consider genetic counseling if recommended',
                'Be aware of warning signs and symptoms',
                'Discuss birth plan options with your healthcare provider'
            ]
        },
        'High Risk': {
            'color': 'danger',
            'message': 'High risk pregnancy. Requires immediate medical attention and specialized care.',
            'recommendations': [
                'Seek immediate consultation with a maternal-fetal medicine specialist',
                'Schedule frequent medical monitoring and tests',
                'Follow strict medical guidelines and restrictions',
                'Consider hospitalization or bed rest if recommended',
                'Plan delivery at a high-risk obstetric facility',
                'Prepare for possible complications and emergency interventions'
            ]
        }
    }
    
    # Get confidence score
    max_prob = np.max(probabilities)
    confidence = f"{max_prob * 100:.1f}%"
    
    return {
        'risk_level': risk_level,
        'confidence': confidence,
        'color': risk_info.get(risk_level, risk_info['Medium Risk'])['color'],
        'message': risk_info.get(risk_level, risk_info['Medium Risk'])['message'],
        'recommendations': risk_info.get(risk_level, risk_info['Medium Risk'])['recommendations'],
        'probabilities': {f'Risk Level {i}': {'percentage': f"{prob*100:.1f}", 'width': f"{prob*100:.1f}"} for i, prob in enumerate(probabilities)}
    }

@app.route('/')
def index():
    """Main page with prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert numeric fields
        numeric_fields = [
            'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'
        ]
        
        features = {}
        for field, value in form_data.items():
            if field in numeric_fields:
                try:
                    features[field] = float(value)
                except ValueError:
                    flash(f'Invalid value for {field}. Please enter a valid number.', 'error')
                    return redirect(url_for('index'))
            else:
                features[field] = value
        
        # Make prediction
        if model and preprocessor:
            prediction, probabilities = model.predict_single(features)
            
            # Get risk level information
            risk_info = get_risk_level_info(prediction, probabilities)
            
            # Prepare data for results page
            result_data = {
                'features': features,
                'prediction': risk_info,
                'model_name': model.best_model_name
            }
            
            return render_template('results.html', **result_data)
        else:
            flash('Model not loaded. Please contact administrator.', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Error making prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        if model and preprocessor:
            prediction, probabilities = model.predict_single(data)
            risk_info = get_risk_level_info(prediction, probabilities)
            
            return jsonify({
                'success': True,
                'prediction': risk_info,
                'model_name': model.best_model_name
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Display model information"""
    if model and preprocessor:
        info = {
            'model_name': model.best_model_name,
            'features': preprocessor.feature_columns if hasattr(preprocessor, 'feature_columns') else [],
            'feature_importance': model.feature_importance.to_dict('records') if model.feature_importance is not None else []
        }
        return render_template('model_info.html', **info)
    else:
        flash('Model not loaded.', 'error')
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if model and preprocessor else 'unhealthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    }
    return jsonify(status)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

def create_sample_data():
    """Create sample pregnancy data if it doesn't exist"""
    data_path = 'data/pregnancy_data.csv'
    
    if not os.path.exists(data_path):
        print("Creating sample pregnancy data...")
        
        # Create sample data based on medical research
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic pregnancy risk data
        data = []
        
        for i in range(n_samples):
            # Age: typically 18-45, with higher risk at extremes
            age = np.random.normal(28, 6)
            age = max(15, min(45, age))
            
            # Blood pressure: Normal around 120/80, high risk if elevated
            systolic_bp = np.random.normal(125, 15)
            diastolic_bp = np.random.normal(82, 10)
            
            # Blood sugar: Normal fasting glucose around 70-100 mg/dL
            blood_sugar = np.random.normal(95, 20)
            
            # Body temperature: Normal around 98.6°F
            body_temp = np.random.normal(98.6, 0.8)
            
            # Heart rate: Normal around 60-100 bpm, pregnancy increases it
            heart_rate = np.random.normal(85, 12)
            
            # Categorical variables
            
            # Risk level calculation based on factors
            risk_factors = 0
            
            # Age risk
            if age < 20 or age > 35:
                risk_factors += 1
            
            # BP risk
            if systolic_bp > 140 or diastolic_bp > 90:
                risk_factors += 2
            
            # Blood sugar risk
            if blood_sugar > 125:
                risk_factors += 2
            
            # Temperature risk
            if body_temp > 100.4 or body_temp < 97:
                risk_factors += 1
            
            # Heart rate risk
            if heart_rate > 100 or heart_rate < 60:
                risk_factors += 1
            
            # Determine risk level
            if risk_factors >= 4:
                risk_level = 'high risk'
            elif risk_factors >= 2:
                risk_level = 'mid risk'
            else:
                risk_level = 'low risk'
            
            data.append({
                'Age': round(age),
                'SystolicBP': round(systolic_bp),
                'DiastolicBP': round(diastolic_bp),
                'BS': round(blood_sugar, 1),
                'BodyTemp': round(body_temp, 1),
                'HeartRate': round(heart_rate),
                'RiskLevel': risk_level
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Sample data created with {len(df)} records")
        print(f"Risk level distribution:")
        print(df['RiskLevel'].value_counts())

def initialize_app():
    """Initialize the application"""
    print("Initializing Pregnancy Risk Prediction App...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Create sample data if needed
    create_sample_data()
    
    # Try to load existing model
    if not load_model_and_preprocessor():
        print("No trained model found. Training new model...")
        try:
            from model import train_pregnancy_risk_model
            trained_model, metrics = train_pregnancy_risk_model()
            if trained_model:
                global model, preprocessor
                model = trained_model
                preprocessor = trained_model.preprocessor
                print("Model trained and loaded successfully")
            else:
                print("Error training model")
        except Exception as e:
            print(f"Error during model training: {e}")
    
    print("App initialization complete")

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
