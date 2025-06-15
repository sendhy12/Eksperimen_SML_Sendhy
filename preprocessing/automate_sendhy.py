import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load raw dataset"""
        return pd.read_csv(file_path)
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        before = df.shape[0]
        df_clean = df.drop_duplicates()
        after = df_clean.shape[0]
        print(f"Removed {before - after} duplicate rows.")
        return df_clean

    def encode_categorical(self, df, target_column):
        """Encode categorical variables with LabelEncoder"""
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=['object']).columns:
            if col != target_column:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
        return df_encoded
    
    def scale_numerical(self, df, target_column):
        """Scale only numerical features excluding target"""
        df_scaled = df.copy()
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(target_column)  # exclude target
        df_scaled[numeric_cols] = self.scaler.fit_transform(df_scaled[numeric_cols])
        return df_scaled

    def preprocess_data(self, file_path, target_column, test_size=0.2):
        """Complete preprocessing pipeline"""
        print("Loading data...")
        df = self.load_data(file_path)

        print("Removing duplicates...")
        df = self.remove_duplicates(df)
        
        print("Encoding categorical variables...")
        df = self.encode_categorical(df, target_column)
        
        print("Scaling numerical features...")
        df = self.scale_numerical(df, target_column)
        
        print("Splitting data...")
        X = df.drop(target_column, axis=1)
        y = df[target_column].astype(int)  # ensure target is integer
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, output_dir):
        """Save preprocessed data and preprocessing objects"""
        os.makedirs(output_dir, exist_ok=True)
        
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        
        # Save scaler and encoders
        with open(f"{output_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f"{output_dir}/label_encoders.pkl", 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        print(f"Preprocessed data saved to {output_dir}")

# Main execution
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Path dataset dan nama kolom target
    raw_data_path = "../heart_raw.csv"
    target_column = "target"
    output_directory = "./heart_preprocessing"
    
    # Run preprocessing
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
        raw_data_path, target_column
    )
    
    # Save results
    preprocessor.save_preprocessed_data(
        X_train, X_test, y_train, y_test, output_directory
    )
    
    print("Preprocessing completed successfully!")
