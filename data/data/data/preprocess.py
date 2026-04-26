import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class PregnancyDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self, filepath):
        """Load pregnancy data from CSV file"""
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self, data):
        """Explore the dataset"""
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {data.shape}")
        print(f"\nData types:\n{data.dtypes}")
        print(f"\nMissing values:\n{data.isnull().sum()}")
        print(f"\nBasic statistics:\n{data.describe()}")
        
        # Check for categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            print(f"\nCategorical columns: {categorical_cols}")
            for col in categorical_cols:
                print(f"\n{col} unique values: {data[col].nunique()}")
                print(f"Values: {data[col].unique()}")
        
        return data.info()
    
    def clean_data(self, data):
        """Clean the dataset"""
        print("\n=== DATA CLEANING ===")
        
        # Make a copy
        cleaned_data = data.copy()
        
        # Handle missing values
        if cleaned_data.isnull().sum().sum() > 0:
            print("Handling missing values...")
            
            # For numerical columns, fill with median
            numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if cleaned_data[col].isnull().sum() > 0:
                    median_val = cleaned_data[col].median()
                    cleaned_data[col].fillna(median_val, inplace=True)
                    print(f"Filled missing values in {col} with median: {median_val}")
            
            # For categorical columns, fill with mode
            categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if cleaned_data[col].isnull().sum() > 0:
                    mode_val = cleaned_data[col].mode()[0] if len(cleaned_data[col].mode()) > 0 else 'Unknown'
                    cleaned_data[col].fillna(mode_val, inplace=True)
                    print(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Remove duplicates
        initial_shape = cleaned_data.shape[0]
        cleaned_data = cleaned_data.drop_duplicates()
        final_shape = cleaned_data.shape[0]
        if initial_shape != final_shape:
            print(f"Removed {initial_shape - final_shape} duplicate rows")
        
        print(f"Cleaned data shape: {cleaned_data.shape}")
        return cleaned_data
    
    def encode_categorical_features(self, data):
        """Encode categorical features"""
        print("\n=== ENCODING CATEGORICAL FEATURES ===")
        
        encoded_data = data.copy()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from categorical encoding if it exists
        target_candidates = ['RiskLevel', 'Risk', 'risk_level', 'risk', 'target', 'Target']
        target_col = None
        for col in target_candidates:
            if col in categorical_cols:
                target_col = col
                break
        
        if target_col:
            categorical_cols.remove(target_col)
            print(f"Target column identified: {target_col}")
        
        # Encode categorical features
        for col in categorical_cols:
            if col in encoded_data.columns:
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Encode target variable separately
        if target_col and target_col in encoded_data.columns:
            le_target = LabelEncoder()
            encoded_data[target_col] = le_target.fit_transform(encoded_data[target_col].astype(str))
            self.label_encoders[target_col] = le_target
            print(f"Target encoded {target_col}: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
        
        return encoded_data, target_col
    
    def prepare_features_target(self, data, target_col=None):
        """Prepare features and target variables"""
        print("\n=== PREPARING FEATURES AND TARGET ===")
        
        # Auto-detect target column if not provided
        if target_col is None:
            target_candidates = ['RiskLevel', 'Risk', 'risk_level', 'risk', 'target', 'Target']
            for col in target_candidates:
                if col in data.columns:
                    target_col = col
                    break
        
        if target_col and target_col in data.columns:
            # Separate features and target
            X = data.drop(columns=[target_col])
            y = data[target_col]
            print(f"Features shape: {X.shape}")
            print(f"Target shape: {y.shape}")
            print(f"Target distribution:\n{y.value_counts()}")
        else:
            # If no target found, treat all as features
            X = data.copy()
            y = None
            print(f"No target column found. Using all data as features: {X.shape}")
        
        self.feature_columns = X.columns.tolist()
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        print("\n=== SCALING FEATURES ===")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            print(f"Scaled training features: {X_train_scaled.shape}")
            print(f"Scaled test features: {X_test_scaled.shape}")
            return X_train_scaled, X_test_scaled
        
        print(f"Scaled features: {X_train_scaled.shape}")
        return X_train_scaled
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save the preprocessor objects"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Load the preprocessor objects"""
        try:
            preprocessor_data = joblib.load(filepath)
            self.scaler = preprocessor_data['scaler']
            self.label_encoders = preprocessor_data['label_encoders']
            self.feature_columns = preprocessor_data['feature_columns']
            print(f"Preprocessor loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
            return False
    
    def preprocess_pipeline(self, data, target_col=None, test_size=0.2, random_state=42):
        """Complete preprocessing pipeline"""
        print("=== STARTING PREPROCESSING PIPELINE ===")
        
        # Explore data
        self.explore_data(data)
        
        # Clean data
        cleaned_data = self.clean_data(data)
        
        # Encode categorical features
        encoded_data, detected_target = self.encode_categorical_features(cleaned_data)
        
        # Use detected target if not provided
        if target_col is None:
            target_col = detected_target
        
        # Prepare features and target
        X, y = self.prepare_features_target(encoded_data, target_col)
        
        if y is not None:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            
            print("\n=== PREPROCESSING COMPLETE ===")
            print(f"Training set: {X_train_scaled.shape[0]} samples")
            print(f"Test set: {X_test_scaled.shape[0]} samples")
            print(f"Features: {len(self.feature_columns)}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
        else:
            # Scale all features
            X_scaled = self.scale_features(X)
            return X_scaled, None, None, None

def main():
    """Main function to run preprocessing"""
    # Initialize preprocessor
    preprocessor = PregnancyDataPreprocessor()
    
    # Load data
    data_path = 'data/pregnancy_data.csv'
    data = preprocessor.load_data(data_path)
    
    if data is not None:
        # Run preprocessing pipeline
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(data)
        
        # Save preprocessor
        preprocessor.save_preprocessor()
        
        # Save processed data
        os.makedirs('data', exist_ok=True)
        if X_train is not None:
            X_train.to_csv('data/X_train.csv', index=False)
            X_test.to_csv('data/X_test.csv', index=False)
            pd.Series(y_train).to_csv('data/y_train.csv', index=False, header=['target'])
            pd.Series(y_test).to_csv('data/y_test.csv', index=False, header=['target'])
            print("Processed data saved to CSV files")
        
        return X_train, X_test, y_train, y_test
    
    return None, None, None, None

if __name__ == "__main__":
    main()
