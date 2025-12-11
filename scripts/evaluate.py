"""
Model Evaluation Script for Real Estate Investment Advisor
Generates comprehensive metrics and visualizations for classification and regression models
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    # Classification
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    # Regression
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')


def load_data_and_models(args):
    """Load test data and trained models"""
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    
    print(f"Loading models...")
    clf_model = joblib.load(args.classification_model)
    reg_model = joblib.load(args.regression_model)
    
    # Load feature columns
    clf_features_path = args.classification_model.replace('.pkl', '_features.json')
    reg_features_path = args.regression_model.replace('.pkl', '_features.json')
    
    with open(clf_features_path, 'r') as f:
        clf_features = json.load(f)
    with open(reg_features_path, 'r') as f:
        reg_features = json.load(f)
    
    return test_df, clf_model, reg_model, clf_features, reg_features


def evaluate_classification(test_df, model, features, output_dir):
    """Evaluate classification model"""
    print("\n" + "="*50)
    print("CLASSIFICATION MODEL EVALUATION")
    print("="*50)
    
    # Prepare data
    X_test = test_df[features]
    y_test = test_df['Good_Investment']
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    # Print metrics
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'classification_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to {metrics_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Good', 'Good Investment'],
                yticklabels=['Not Good', 'Good Investment'])
    plt.title('Confusion Matrix - Classification Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {cm_path}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Classification Model')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved to {roc_path}")
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Classification Model')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Precision-recall curve saved to {pr_path}")
    
    # Classification Report
    report = classification_report(y_test, y_pred, target_names=['Not Good', 'Good Investment'])
    print("\nClassification Report:")
    print(report)
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Classification report saved to {report_path}")
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top 20
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances - Classification')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        fi_path = os.path.join(output_dir, 'classification_feature_importance.png')
        plt.savefig(fi_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Feature importance plot saved to {fi_path}")
        
        # Save to CSV
        fi_csv_path = os.path.join(output_dir, 'classification_feature_importance.csv')
        feature_importance.to_csv(fi_csv_path, index=False)
        print(f"✅ Feature importance CSV saved to {fi_csv_path}")
    
    return metrics


def evaluate_regression(test_df, model, features, output_dir):
    """Evaluate regression model"""
    print("\n" + "="*50)
    print("REGRESSION MODEL EVALUATION")
    print("="*50)
    
    # Prepare data
    X_test = test_df[features]
    y_test = test_df['Future_Price_5Y']
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'mape': mape
    }
    
    # Print metrics
    print(f"\nTest Set Metrics:")
    print(f"  RMSE:  ₹{rmse:.2f} Lakhs")
    print(f"  MAE:   ₹{mae:.2f} Lakhs")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'regression_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to {metrics_path}")
    
    # Actual vs Predicted Scatter Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price (Lakhs)')
    plt.ylabel('Predicted Price (Lakhs)')
    plt.title(f'Actual vs Predicted Prices (R² = {r2:.4f})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Actual vs predicted plot saved to {scatter_path}")
    
    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price (Lakhs)')
    plt.ylabel('Residuals (Lakhs)')
    plt.title('Residual Plot - Regression Model')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    residual_path = os.path.join(output_dir, 'residual_plot.png')
    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Residual plot saved to {residual_path}")
    
    # Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.xlabel('Residual (Lakhs)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    error_dist_path = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Error distribution saved to {error_dist_path}")
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top 20
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances - Regression')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        fi_path = os.path.join(output_dir, 'regression_feature_importance.png')
        plt.savefig(fi_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Feature importance plot saved to {fi_path}")
        
        # Save to CSV
        fi_csv_path = os.path.join(output_dir, 'regression_feature_importance.csv')
        feature_importance.to_csv(fi_csv_path, index=False)
        print(f"✅ Feature importance CSV saved to {fi_csv_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--test-data', type=str, default='data/cleaned_dataset_test.csv',
                       help='Path to test data CSV')
    parser.add_argument('--classification-model', type=str, default='models/classification_model.pkl',
                       help='Path to classification model')
    parser.add_argument('--regression-model', type=str, default='models/regression_model.pkl',
                       help='Path to regression model')
    parser.add_argument('--output-dir', type=str, default='reports/',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data and models
    test_df, clf_model, reg_model, clf_features, reg_features = load_data_and_models(args)
    
    print(f"\nTest set size: {len(test_df)} samples")
    
    # Evaluate classification model
    clf_metrics = evaluate_classification(test_df, clf_model, clf_features, args.output_dir)
    
    # Evaluate regression model
    reg_metrics = evaluate_regression(test_df, reg_model, reg_features, args.output_dir)
    
    # Summary
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"\nClassification Accuracy: {clf_metrics['accuracy']*100:.2f}%")
    print(f"Regression R² Score: {reg_metrics['r2_score']:.4f}")
    print(f"\nAll results saved to: {args.output_dir}")
    print("\nGenerated Files:")
    print("  - classification_metrics.json")
    print("  - regression_metrics.json")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - precision_recall_curve.png")
    print("  - actual_vs_predicted.png")
    print("  - residual_plot.png")
    print("  - error_distribution.png")
    print("  - feature_importance plots & CSVs")


if __name__ == '__main__':
    main()
