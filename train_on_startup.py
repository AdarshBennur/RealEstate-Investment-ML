"""
Streamlit startup script - trains models if they don't exist
This is called before the main app starts
"""

import os
import subprocess
import sys

def ensure_models_exist():
    """Train models if they don't exist"""
    clf_model_path = 'models/classification_model.pkl'
    reg_model_path = 'models/regression_model.pkl'
    
    if not os.path.exists(clf_model_path) or not os.path.exists(reg_model_path):
        print("=" * 60)
        print("TRAINING MODELS ON FIRST RUN...")
        print("=" * 60)
        print("This will take ~2 minutes. The app will load automatically when complete.")
        print()
        
        try:
            #Train classification
            print("Training classification model...")
            subprocess.run([
                sys.executable, 'scripts/train_classification.py',
                '--data-path', 'data/cleaned_dataset_train.csv',
                '--model-output', clf_model_path,
                '--model-type', 'random_forest',
                '--random-seed', '42'
            ], check=True)
            
            # Train regression
            print("\nTraining regression model...")
            subprocess.run([
                sys.executable, 'scripts/train_regression.py',
                '--data-path', 'data/cleaned_dataset_train.csv',
                '--model-output', reg_model_path,
                '--model-type', 'random_forest',
                '--random-seed', '42'
            ], check=True)
            
            print("\n" + "=" * 60)
            print("✅ MODELS TRAINED SUCCESSFULLY!")
            print("="  * 60)
            
        except Exception as e:
            print(f"\n❌ Error training models: {e}")
            sys.exit(1)
    else:
        print("✅ Models already exist, skipping training")

if __name__ == '__main__':
    ensure_models_exist()
