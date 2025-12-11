"""
Regression Model Training Script for Real Estate Investment Advisor
Trains a regression model to predict "Future_Price_5Y"
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib
import json
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path):
    """Load hyperparameter config from JSON"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f).get('regression', {})
    return {}


def get_model(model_type, params):
    """Initialize model based on type and parameters"""
    if model_type == 'linear':
        return LinearRegression(**params)
    elif model_type == 'ridge':
        return Ridge(**params)
    elif model_type == 'random_forest':
        return RandomForestRegressor(**params)
    elif model_type == 'gradient_boosting':
        return GradientBoostingRegressor(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_regression_model(args):
    """Main training function"""
    
    # Set MLflow tracking URI
    if 'MLFLOW_TRACKING_URI' in os.environ:
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    else:
        mlflow.set_tracking_uri("http://localhost:5000")
    
    # Set experiment
    mlflow.set_experiment("real_estate_regression")
    
    # Load data
    print(f"Loading training data from {args.data_path}...")
    train_df = pd.read_csv(args.data_path)
    
    print(f"Loaded {len(train_df)} training samples")
    print(f"Features: {len(train_df.columns)}")
    
    # Separate features and target
    target = 'Future_Price_5Y'
    exclude_cols = ['ID', 'Good_Investment', 'Future_Price_5Y']
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    
    print(f"\nTarget statistics:")
    print(f"Mean:   ₹{y_train.mean():.2f} Lakhs")
    print(f"Median: ₹{y_train.median():.2f} Lakhs")
    print(f"Std:    ₹{y_train.std():.2f} Lakhs")
    
    # Load hyperparameters
    config = load_config(args.params_json) if args.params_json else {}
    model_params = config.get(args.model_type, {})
    
    # Set default parameters if not provided
    if args.model_type == 'random_forest' and not model_params:
        model_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': args.random_seed,
            'n_jobs': -1
        }
    elif args.model_type in ['linear', 'ridge'] and not model_params:
        model_params = {}
        if args.model_type == 'ridge':
            model_params['random_state'] = args.random_seed
    elif args.model_type == 'gradient_boosting' and not model_params:
        model_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': args.random_seed
        }
    
    print(f"\nModel: {args.model_type}")
    print(f"Parameters: {model_params}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{args.model_type}_regression"):
        
        # Log parameters
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("random_seed", args.random_seed)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_samples", len(X_train))
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        
        # Initialize and train model
        print("\nTraining model...")
        model = get_model(args.model_type, model_params)
        model.fit(X_train, y_train)
        
        # Make predictions on training set
        y_train_pred = model.predict(X_train)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
        
        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("train_mape", train_mape)
        
        print("\n" + "="*50)
        print("Training Metrics:")
        print("="*50)
        print(f"RMSE:  ₹{train_rmse:.2f} Lakhs")
        print(f"MAE:   ₹{train_mae:.2f} Lakhs")
        print(f"R²:    {train_r2:.4f}")
        print(f"MAPE:  {train_mape:.2f}%")
        
        # Log feature importances if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Feature Importances:")
            print(feature_importance.head(10))
            
            # Save feature importance
            importance_path = 'feature_importance_regression.csv'
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            os.remove(importance_path)
        
        # Save model
        model_path = args.model_output
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            "regression_model",
            registered_model_name="real_estate_regressor"
        )
        
        print("\nModel logged to MLflow")
        
        # Save feature columns
        feature_cols_path = model_path.replace('.pkl', '_features.json')
        with open(feature_cols_path, 'w') as f:
            json.dump(feature_cols, f)
        mlflow.log_artifact(feature_cols_path)
        
        print(f"Feature columns saved to {feature_cols_path}")
        
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Train regression model')
    parser.add_argument('--data-path', type=str, default='data/cleaned_dataset_train.csv',
                       help='Path to training data CSV')
    parser.add_argument('--model-output', type=str, default='models/regression_model.pkl',
                       help='Path to save trained model')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['linear', 'ridge', 'random_forest', 'gradient_boosting'],
                       help='Type of regression model')
    parser.add_argument('--params-json', type=str, default=None,
                       help='Path to hyperparameters JSON file')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with smaller dataset')
    
    args = parser.parse_args()
    
    # Quick mode: use only 10% of data
    if args.quick:
        print("QUICK MODE: Using 10% of data")
        df = pd.read_csv(args.data_path)
        df_sample = df.sample(frac=0.1, random_seed=args.random_seed)
        temp_path = args.data_path.replace('.csv', '_quick.csv')
        df_sample.to_csv(temp_path, index=False)
        args.data_path = temp_path
    
    train_regression_model(args)
    
    # Clean up quick mode file
    if args.quick and os.path.exists(temp_path):
        os.remove(temp_path)


if __name__ == '__main__':
    main()
