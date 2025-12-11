"""
Classification Model Training Script for Real Estate Investment Advisor
Trains a binary classifier to predict "Good_Investment"
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib
import json
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path):
    """Load hyperparameter config from JSON"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f).get('classification', {})
    return {}


def get_model(model_type, params):
    """Initialize model based on type and parameters"""
    if model_type == 'logistic':
        return LogisticRegression(**params)
    elif model_type == 'random_forest':
        return RandomForestClassifier(**params)
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_classification_model(args):
    """Main training function"""
    
    # Set MLflow tracking URI
    if 'MLFLOW_TRACKING_URI' in os.environ:
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    else:
        mlflow.set_tracking_uri("http://localhost:5000")
    
    # Set experiment
    mlflow.set_experiment("real_estate_classification")
    
    # Load data
    print(f"Loading training data from {args.data_path}...")
    train_df = pd.read_csv(args.data_path)
    
    print(f"Loaded {len(train_df)} training samples")
    print(f"Features: {len(train_df.columns)}")
    
    # Separate features and target
    target = 'Good_Investment'
    exclude_cols = ['ID', 'Good_Investment', 'Future_Price_5Y']
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    
    print(f"\nTarget distribution:")
    print(y_train.value_counts(normalize=True))
    
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
    elif args.model_type == 'logistic' and not model_params:
        model_params = {
            'max_iter': 1000,
            'random_state': args.random_seed,
            'n_jobs': -1
        }
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
    with mlflow.start_run(run_name=f"{args.model_type}_classification"):
        
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
        y_train_proba = model.predict_proba(X_train)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_proba)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        
        print("\n" + "="*50)
        print("Training Metrics:")
        print("="*50)
        print(f"Accuracy:  {train_accuracy:.4f}")
        print(f"Precision: {train_precision:.4f}")
        print(f"Recall:    {train_recall:.4f}")
        print(f"F1-Score:  {train_f1:.4f}")
        print(f"ROC-AUC:   {train_roc_auc:.4f}")
        
        # Log feature importances if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Feature Importances:")
            print(feature_importance.head(10))
            
            # Save feature importance
            importance_path = 'feature_importance_classification.csv'
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
            "classification_model",
            registered_model_name="real_estate_classifier"
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
    parser = argparse.ArgumentParser(description='Train classification model')
    parser.add_argument('--data-path', type=str, default='data/cleaned_dataset_train.csv',
                       help='Path to training data CSV')
    parser.add_argument('--model-output', type=str, default='models/classification_model.pkl',
                       help='Path to save trained model')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['logistic', 'random_forest', 'gradient_boosting'],
                       help='Type of classification model')
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
        df_sample = df.sample(frac=0.1, random_state=args.random_seed)
        temp_path = args.data_path.replace('.csv', '_quick.csv')
        df_sample.to_csv(temp_path, index=False)
        args.data_path = temp_path
    
    train_classification_model(args)
    
    # Clean up quick mode file
    if args.quick and os.path.exists(temp_path):
        os.remove(temp_path)


if __name__ == '__main__':
    main()
