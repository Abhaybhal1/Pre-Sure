import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data.preprocess import PregnancyDataPreprocessor

class PregnancyRiskModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = PregnancyDataPreprocessor()
        self.feature_importance = None
        
    def initialize_models(self):
        """Initialize different ML models"""
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        print("Models initialized successfully")
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and compare performance"""
        print("\n=== TRAINING MODELS ===")
        
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Handle multi-class ROC AUC
            try:
                if len(np.unique(y_test)) == 2:
                    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            except:
                roc_auc = 0.0
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            
            model_scores[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'cv_score': cv_mean,
                'cv_std': cv_scores.std()
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, CV Score: {cv_mean:.4f} (±{cv_scores.std():.4f})")
            print(f"Classification Report for {name}:")
            print(classification_report(y_test, y_pred))
        
        # Select best model
        best_score = 0
        for name, scores in model_scores.items():
            combined_score = (scores['accuracy'] + scores['cv_score']) / 2
            if combined_score > best_score:
                best_score = combined_score
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print(f"\n=== BEST MODEL: {self.best_model_name} ===")
        return model_scores
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning on the best model"""
        print(f"\n=== HYPERPARAMETER TUNING FOR {self.best_model_name} ===")
        
        if self.best_model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.best_model_name == 'GradientBoosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        elif self.best_model_name == 'LogisticRegression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif self.best_model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.models[self.best_model_name],
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update best model with tuned parameters
        self.best_model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def analyze_feature_importance(self, X_train):
        """Analyze feature importance"""
        print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models
            importance_scores = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # Linear models
            importance_scores = np.abs(self.best_model.coef_[0])
        else:
            print("Feature importance not available for this model type")
            return None
        
        # Create feature importance DataFrame
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(len(importance_scores))]
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        print(self.feature_importance.head(10))
        
        return self.feature_importance
    
    def plot_feature_importance(self, save_path='static/feature_importance.png'):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("Feature importance not calculated yet")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(15)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top 15 Feature Importance - {self.best_model_name}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to {save_path}")
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print(f"\n=== FINAL MODEL EVALUATION ===")
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            if len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except:
            roc_auc = 0.0
        
        print(f"Final Model: {self.best_model_name}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_single(self, features_dict):
        """Predict for a single instance with feature dictionary"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Apply same preprocessing as training data
        if hasattr(self.preprocessor, 'feature_columns'):
            # Ensure all required features are present
            for col in self.preprocessor.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0  # Default value for missing features
            
            # Reorder columns to match training data
            features_df = features_df[self.preprocessor.feature_columns]
        
        # Scale features
        if hasattr(self.preprocessor, 'scaler'):
            features_scaled = self.preprocessor.scaler.transform(features_df)
            features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
        
        # Make prediction
        prediction = self.best_model.predict(features_df)[0]
        probability = self.best_model.predict_proba(features_df)[0]
        
        return prediction, probability
    
    def save_model(self, filepath='models/trained_model.pkl'):
        """Save the trained model"""
        if self.best_model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_importance': self.feature_importance,
            'preprocessor': self.preprocessor
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/trained_model.pkl'):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.feature_importance = model_data.get('feature_importance', None)
            self.preprocessor = model_data.get('preprocessor', PregnancyDataPreprocessor())
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def train_pregnancy_risk_model():
    """Main function to train the pregnancy risk model"""
    print("=== PREGNANCY RISK PREDICTION MODEL TRAINING ===")
    
    # Initialize model
    model = PregnancyRiskModel()
    
    # Load preprocessed data
    try:
        X_train = pd.read_csv('data/X_train.csv')
        X_test = pd.read_csv('data/X_test.csv')
        y_train = pd.read_csv('data/y_train.csv')['target']
        y_test = pd.read_csv('data/y_test.csv')['target']
        print("Loaded preprocessed data successfully")
    except:
        # If preprocessed data not available, run preprocessing
        print("Preprocessed data not found. Running preprocessing...")
        preprocessor = PregnancyDataPreprocessor()
        data = preprocessor.load_data('data/pregnancy_data.csv')
        if data is not None:
            X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(data)
            model.preprocessor = preprocessor
        else:
            print("Error: Could not load pregnancy data")
            return None
    
    # Initialize and train models
    model.initialize_models()
    model_scores = model.train_models(X_train, y_train, X_test, y_test)
    
    # Hyperparameter tuning
    best_params = model.hyperparameter_tuning(X_train, y_train)
    
    # Retrain with best parameters and evaluate
    final_metrics = model.evaluate_model(X_test, y_test)
    
    # Analyze feature importance
    feature_importance = model.analyze_feature_importance(X_train)
    model.plot_feature_importance()
    
    # Save model
    model.save_model()
    
    print("\n=== MODEL TRAINING COMPLETE ===")
    return model, final_metrics

if __name__ == "__main__":
    train_pregnancy_risk_model()
