"""
Pipeline Validation Tests for Real Estate Investment Advisor
Unit-style tests to ensure pipeline correctness
"""

import os
import sys
import joblib
import json
import pandas as pd
import numpy as np


def test_preprocessor():
    """Test preprocessor loads and has correct attributes"""
    print("Testing preprocessor...")
    
    preprocessor_path = 'artifacts/preprocessor.pkl'
    if not os.path.exists(preprocessor_path):
        print(f"❌ Preprocessor not found at {preprocessor_path}")
        return False
    
    preprocessor = joblib.load(preprocessor_path)
    
    # Check attributes
    assert hasattr(preprocessor, 'label_encoders'), "Missing label_encoders"
    assert hasattr(preprocessor, 'scaler'), "Missing scaler"
    assert hasattr(preprocessor, 'fitted'), "Missing fitted attribute"
    assert preprocessor.fitted, "Preprocessor not fitted"
    
    print("  ✅ Preprocessor valid")
    return True


def test_classification_model():
    """Test classification model loads and can predict"""
    print("Testing classification model...")
    
    model_path = 'models/classification_model.pkl'
    features_path = 'models/classification_model_features.json'
    
    if not os.path.exists(model_path):
        print(f"⚠️  Classification model not found at {model_path}")
        return False
    
    # Load model
    model = joblib.load(model_path)
    
    # Load features
    with open(features_path, 'r') as f:
        features = json.load(f)
    
    # Check model attributes
    assert hasattr(model, 'predict'), "Model missing predict method"
    assert hasattr(model, 'predict_proba'), "Model missing predict_proba method"
    
    # Test prediction shape
    test_data = pd.read_csv('data/cleaned_dataset_test.csv')
    X_test = test_data[features].head(10)
    
    predictions = model.predict(X_test)
    probas = model.predict_proba(X_test)
    
    assert predictions.shape == (10,), f"Unexpected prediction shape: {predictions.shape}"
    assert probas.shape == (10, 2), f"Unexpected proba shape: {probas.shape}"
    assert np.all((predictions == 0) | (predictions == 1)), "Invalid prediction values"
    
    print(f"  ✅ Classification model valid (predicts binary labels)")
    return True


def test_regression_model():
    """Test regression model loads and can predict"""
    print("Testing regression model...")
    
    model_path = 'models/regression_model.pkl'
    features_path = 'models/regression_model_features.json'
    
    if not os.path.exists(model_path):
        print(f"⚠️  Regression model not found at {model_path}")
        return False
    
    # Load model
    model = joblib.load(model_path)
    
    # Load features
    with open(features_path, 'r') as f:
        features = json.load(f)
    
    # Check model attributes
    assert hasattr(model, 'predict'), "Model missing predict method"
    
    # Test prediction shape
    test_data = pd.read_csv('data/cleaned_dataset_test.csv')
    X_test = test_data[features].head(10)
    
    predictions = model.predict(X_test)
    
    assert predictions.shape == (10,), f"Unexpected prediction shape: {predictions.shape}"
    assert np.all(predictions >= 0), "Negative price predictions found"
    
    print(f"  ✅ Regression model valid (predicts positive prices)")
    return True


def test_feature_columns_match():
    """Test that feature columns are consistent"""
    print("Testing feature column consistency...")
    
    # Load feature lists
    with open('models/classification_model_features.json', 'r') as f:
        clf_features = json.load(f)
    with open('models/regression_model_features.json', 'r') as f:
        reg_features = json.load(f)
    
    # Load test data
    test_data = pd.read_csv('data/cleaned_dataset_test.csv')
    
    # Check classification features exist
    missing_clf = [f for f in clf_features if f not in test_data.columns]
    if missing_clf:
        print(f"  ❌ Missing classification features: {missing_clf}")
        return False
    
    # Check regression features exist
    missing_reg = [f for f in reg_features if f not in test_data.columns]
    if missing_reg:
        print(f"  ❌ Missing regression features: {missing_reg}")
        return False
    
    print(f"  ✅ All features present in test data")
    print(f"     Classification: {len(clf_features)} features")
    print(f"     Regression: {len(reg_features)} features")
    return True


def test_data_files():
    """Test that all required data files exist"""
    print("Testing data files...")
    
    required_files = [
        'data/cleaned_dataset.csv',
        'data/cleaned_dataset_train.csv',
        'data/cleaned_dataset_test.csv'
    ]
    
    all_exist = True
    for filepath in required_files:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"  ✅ {filepath} ({len(df)} rows)")
        else:
            print(f"  ❌ Missing: {filepath}")
            all_exist = False
    
    return all_exist


def main():
    print("="*50)
    print("PIPELINE VALIDATION TESTS")
    print("="*50)
    print()
    
    tests = [
        ("Data Files", test_data_files),
        ("Preprocessor", test_preprocessor),
        ("Classification Model", test_classification_model),
        ("Regression Model", test_regression_model),
        ("Feature Consistency", test_feature_columns_match)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Error in {test_name}: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("="*50)
    print("TEST SUMMARY")
    print("="*50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
