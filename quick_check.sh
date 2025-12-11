#!/bin/bash
# Quick validation script for Real Estate Investment Advisor

set -e  # Exit on error

echo "🚀 Real Estate Investment ML - Quick Check"
echo "=========================================="

# Check Python version
echo ""
echo "✓ Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "✓ Installing dependencies..."
pip install -q -r requirements.txt

# Check data files exist
echo ""
echo "✓ Checking data files..."
if [ ! -f "data/cleaned_dataset.csv" ]; then
    echo "❌ Error: Cleaned dataset not found!"
    echo "   Run: python scripts/preprocess.py"
    exit 1
fi
echo "   ✓ cleaned_dataset.csv found"

# Check model files exist
echo ""
echo "✓ Checking model files..."
if [ ! -f "models/classification_model.pkl" ]; then
    echo "⚠️  Warning: Classification model not found!"
    echo "   Run: python scripts/train_classification.py"
fi

if [ ! -f "models/regression_model.pkl" ]; then
    echo "⚠️  Warning: Regression model not found!"
    echo "   Run: python scripts/train_regression.py"
fi

# Run pipeline check
echo ""
echo "✓ Running pipeline validation..."
python3 scripts/check_pipeline.py

# Test model predictions (if models exist)
if [ -f "models/classification_model.pkl" ] && [ -f "models/regression_model.pkl" ]; then
    echo ""
    echo "✓ Testing model predictions..."
    python3 -c "
import joblib
import pandas as pd
import numpy as np

# Load models
clf = joblib.load('models/classification_model.pkl')
reg = joblib.load('models/regression_model.pkl')

# Load test data
df = pd.read_csv('data/cleaned_dataset_test.csv')

# Get features
import json
with open('models/classification_model_features.json') as f:
    clf_features = json.load(f)
with open('models/regression_model_features.json') as f:
    reg_features = json.load(f)

# Test predictions
X_clf = df[clf_features].head(1)
X_reg = df[reg_features].head(1)

clf_pred = clf.predict(X_clf)[0]
reg_pred = reg.predict(X_reg)[0]

print(f'   ✓ Classification prediction: {clf_pred}')
print(f'   ✓ Regression prediction: ₹{reg_pred:.2f} Lakhs')
"
fi

# Check if Streamlit can start
echo ""
echo "✓ Testing Streamlit app..."
timeout 10s streamlit run app/app.py --server.headless true &> /dev/null &
STREAM_PID=$!
sleep 5

if ps -p $STREAM_PID > /dev/null; then
    echo "   ✓ Streamlit started successfully"
    kill $STREAM_PID 2>/dev/null || true
else
    echo "   ⚠️  Warning: Streamlit may have issues"
fi

echo ""
echo "=========================================="
echo "✅ Quick Check Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run Streamlit: streamlit run app/app.py"
echo "  2. View reports: ls reports/"
echo "  3. Check MLflow: mlflow ui"
echo ""
