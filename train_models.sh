#!/bin/bash
# Train models on Streamlit Cloud startup if models don't exist

echo "Checking for trained models..."

if [ ! -f "models/classification_model.pkl" ] || [ ! -f "models/regression_model.pkl" ]; then
    echo "Models not found. Training models..."
    
    # Train classification model
    python scripts/train_classification.py \
        --data-path data/cleaned_dataset_train.csv \
        --model-output models/classification_model.pkl \
        --model-type random_forest \
        --random-seed 42
    
    # Train regression model
    python scripts/train_regression.py \
        --data-path data/cleaned_dataset_train.csv \
        --model-output models/regression_model.pkl \
        --model-type random_forest \
        --random-seed 42
    
    echo "✅ Models trained successfully!"
else
    echo "✅ Models already exist"
fi
