"""
Data Preprocessing Pipeline for Real Estate Investment Advisor
Handles data cleaning, feature engineering, and target variable generation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import argparse
import os


class RealEstatePreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.fitted = False
        
    def load_data(self, file_path):
        """Load dataset from CSV"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        print("Handling missing values...")
        initial_missing = df.isnull().sum().sum()
        
        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        final_missing = df[col].isnull().sum().sum()
        print(f"Missing values before: {initial_missing}, after: {final_missing}")
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        print("Removing duplicates...")
        initial_rows = len(df)
        df = df.drop_duplicates()
        final_rows = len(df)
        print(f"Removed {initial_rows - final_rows} duplicate rows")
        return df
    
    def engineer_features(self, df):
        """Create derived features"""
        print("Engineering features...")
        
        # Current year for age calculations (2025)
        current_year = 2025
        
        # Age of property (already provided, but ensure it's correct)
        if 'Age_of_Property' in df.columns:
            df['Age_of_Property'] = current_year - df['Year_Built']
        
        # Price per Sq Ft (already provided, but recalculate for consistency)
        if 'Price_per_SqFt' not in df.columns:
            df['Price_per_SqFt'] = df['Price_in_Lakhs'] / df['Size_in_SqFt']
        
        # Total nearby amenities score
        df['Amenities_Score'] = (
            df.get('Nearby_Schools',0).fillna(0) + 
            df.get('Nearby_Hospitals', 0).fillna(0)
        )
        
        # Transport accessibility binary
        if 'Public_Transport_Accessibility' in df.columns:
            df['Has_High_Transport'] = (df['Public_Transport_Accessibility'] == 'High').astype(int)
        
        # Security binary
        if 'Security' in df.columns:
            df['Has_Security'] = (df['Security'] == 'Yes').astype(int)
        
        # Parking binary
        if 'Parking_Space' in df.columns:
            df['Has_Parking'] = (df['Parking_Space'] == 'Yes').astype(int)
        
        # Is ready to move
        if 'Availability_Status' in df.columns:
            df['Is_Ready_To_Move'] = (df['Availability_Status'] == 'Ready_to_Move').astype(int)
        
        # Furnished score (0=Unfurnished, 1=Semi, 2=Furnished)
        if 'Furnished_Status' in df.columns:
            furnished_map = {'Unfurnished': 0, 'Semi-furnished': 1, 'Furnished': 2}
            df['Furnished_Score'] = df['Furnished_Status'].map(furnished_map).fillna(0)
        
        print(f"Created {6} new features")
        return df
    
    def create_target_variables(self, df):
        """Create regression and classification targets"""
        print("Creating target variables...")
        
        # REGRESSION TARGET: Future_Price_5Y
        # Using 8% annual growth rate
        growth_rate = 0.08
        years = 5
        df['Future_Price_5Y'] = df['Price_in_Lakhs'] * ((1 + growth_rate) ** years)
        
        # CLASSIFICATION TARGET: Good_Investment
        # Multi-factor approach:
        # 1. Price below median for the city
        # 2. Price per sqft below median for the city
        # 3. BHK >= 3
        # 4. Ready to move
        # 5. Has amenities
        
        df['Price_Below_Median'] = 0
        df['PricePerSqFt_Below_Median'] = 0
        
        for city in df['City'].unique():
            city_mask = df['City'] == city
            median_price = df.loc[city_mask, 'Price_in_Lakhs'].median()
            median_price_sqft = df.loc[city_mask, 'Price_per_SqFt'].median()
            
            df.loc[city_mask, 'Price_Below_Median'] = (df.loc[city_mask, 'Price_in_Lakhs'] <= median_price).astype(int)
            df.loc[city_mask, 'PricePerSqFt_Below_Median'] = (df.loc[city_mask, 'Price_per_SqFt'] <= median_price_sqft).astype(int)
        
        # Investment score
        df['Investment_Score'] = (
            df['Price_Below_Median'] +
            df['PricePerSqFt_Below_Median'] +
            (df['BHK'] >= 3).astype(int) +
            df.get('Is_Ready_To_Move', 0) +
            (df.get('Amenities_Score', 0) > 10).astype(int)
        )
        
        # Good investment if score >= 3 out of 5
        df['Good_Investment'] = (df['Investment_Score'] >= 3).astype(int)
        
        print(f"Good Investment distribution: {df['Good_Investment'].value_counts().to_dict()}")
        
        # Drop intermediate columns
        df.drop(['Price_Below_Median', 'PricePerSqFt_Below_Median', 'Investment_Score'], axis=1, inplace=True)
        
        return df
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        print("Encoding categorical variables...")
        
        categorical_cols = ['State', 'City', 'Locality', 'Property_Type', 
                           'Furnished_Status', 'Facing', 'Owner_Type', 
                           'Availability_Status', 'Public_Transport_Accessibility']
        
        # Filter to existing columns
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # handle unseen labels
                    le = self.label_encoders[col]
                    df[col] = df[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
        
        print(f"Encoded {len(categorical_cols)} categorical columns")
        return df
    
    def handle_outliers(self, df, columns=None):
        """Handle outliers using IQR method"""
        if columns is None:
            columns = ['Price_in_Lakhs', 'Size_in_SqFt', 'Price_per_SqFt']
        
        print("Handling outliers...")
        initial_rows = len(df)
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        final_rows = len(df)
        print(f"Removed {initial_rows - final_rows} outlier rows")
        return df
    
    def scale_features(self, df, fit=True, exclude_cols=None):
        """Scale numeric features"""
        if exclude_cols is None:
            exclude_cols = ['ID', 'Good_Investment', 'Future_Price_5Y']
        
        print("Scaling numeric features...")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target and ID columns
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if fit:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        else:
            df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        print(f"Scaled {len(scale_cols)} numeric columns")
        return df
    
    def fit_transform(self, df):
        """Fit and transform the data"""
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.engineer_features(df)
        df = self.create_target_variables(df)
        df = self.encode_categorical(df, fit=True)
        df = self.handle_outliers(df)
        #df = self.scale_features(df, fit=True)  # Scale only when training
        
        self.fitted = True
        return df
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        df = self.handle_missing_values(df)
        df = self.engineer_features(df)
        df = self.encode_categorical(df, fit=False)
        #df = self.scale_features(df, fit=False)  # Scale using fitted scaler
        
        return df
    
    def save(self, filepath):
        """Save preprocessor to disk"""
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load preprocessor from disk"""
        preprocessor = joblib.load(filepath)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def main():
    parser = argparse.ArgumentParser(description='Preprocess real estate data')
    parser.add_argument('--input', type=str, default='data/india_housing_prices.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default='data/cleaned_dataset.csv',
                       help='Output CSV file path')
    parser.add_argument('--save-preprocessor', type=str, default='artifacts/preprocessor.pkl',
                       help='Path to save preprocessor')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0-1)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_preprocessor), exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = RealEstatePreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_data(args.input)
    df_cleaned = preprocessor.fit_transform(df)
    
    # Split into train and test
    train_df, test_df = train_test_split(
        df_cleaned, 
        test_size=args.test_size, 
        random_state=args.random_seed,
        stratify=df_cleaned['Good_Investment']
    )
    
    print(f"\nTrain set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    
    # Save cleaned dataset
    df_cleaned.to_csv(args.output, index=False)
    print(f"\nCleaned data saved to {args.output}")
    
    # Save train/test splits
    train_output = args.output.replace('.csv', '_train.csv')
    test_output = args.output.replace('.csv', '_test.csv')
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    print(f"Train set saved to {train_output}")
    print(f"Test set saved to {test_output}")
    
    # Save preprocessor
    preprocessor.save(args.save_preprocessor)
    
    print("\n" + "="*50)
    print("Preprocessing complete!")
    print("="*50)
    print(f"Features: {len(df_cleaned.columns)}")
    print(f"Rows: {len(df_cleaned)}")
    print(f"Good Investment %: {df_cleaned['Good_Investment'].mean()*100:.2f}%")
    print(f"Avg Future Price: ₹{df_cleaned['Future_Price_5Y'].mean():.2f} Lakhs")


if __name__ == '__main__':
    main()
