# Real Estate Investment Advisor

## Predicting Property Profitability & Future Value

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-orange)
![Tracking](https://img.shields.io/badge/Tracking-MLflow-green)
![UI](https://img.shields.io/badge/UI-Streamlit-red)

**Machine Learning application to assist real estate investors in making data-driven investment decisions.**

---

## 📊 Project Overview

This project builds a comprehensive ML system that:

1. **Classifies** properties as "Good Investment" or not
2. **Predicts** estimated property price after 5 years
3. Provides an interactive **Streamlit dashboard** for predictions and insights

### Key Features

- ✅ End-to-end ML pipeline (preprocessing, training, evaluation)
- ✅ MLflow experiment tracking and model registry
- ✅ Multiple model support (Random Forest, Gradient Boosting, Linear)
- ✅ Streamlit web dashboard with visualizations
- ✅ Dockerized deployment
- ✅ Reproducible with fixed random seeds

---

## 🏗️ Architecture

```
┌─────────────────┐
│ Raw Data (CSV)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Preprocessing Pipeline     │
│  - Missing value imputation │
│  - Feature engineering      │
│  - Target creation          │
│  - Encoding & scaling       │
└────────┬────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         Model Training (MLflow)           │
│  ┌────────────┐      ┌──────────────┐   │
│  │ Classifier │      │  Regressor   │   │
│  └────────────┘      └──────────────┘   │
└────────┬──────────────────────────┬──────┘
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────┐
│      MLflow Model Registry                │
│      (Production Tagged Models)           │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│       Streamlit Dashboard                 │
│  - Property details input                 │
│  - Investment prediction                  │
│  - Price forecast                         │
│  - Visualizations & insights              │
└───────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
5_RealEstate-Investment/
├── data/
│   ├── india_housing_prices.csv         # Original dataset (250K rows)
│   ├── cleaned_dataset.csv              # Cleaned data (242K rows)
│   ├── cleaned_dataset_train.csv        # Training set (194K)
│   └── cleaned_dataset_test.csv         # Test set (48K)
├── scripts/
│   ├── preprocess.py                    # Data preprocessing pipeline
│   ├── train_classification.py          # Classification model training
│   ├── train_regression.py              # Regression model training
│   └── evaluate.py                      # Model evaluation & reporting
├── app/
│   └── app.py                           # Streamlit dashboard
├── artifacts/
│   └── preprocessor.pkl                 # Fitted preprocessor
├── models/
│   ├── classification_model.pkl
│   └── regression_model.pkl
├── reports/
│   └── *.png                            # Evaluation plots
├── mlruns/                              # MLflow artifacts
├── requirements.txt
├── Dockerfile
├── README.md                            # This file
└── ProjectImplementationPlan.md         # Implementation strategy
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
python scripts/preprocess.py \
  --input data/india_housing_prices.csv \
  --output data/cleaned_dataset.csv \
  --save-preprocessor artifacts/preprocessor.pkl
```

**Output**: Cleaned dataset (242,630 rows) with engineered features and targets

### 3. Start MLflow Server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

Access MLflow UI: <http://localhost:5000>

### 4. Train Models

**Classification (Good Investment)**:

```bash
python scripts/train_classification.py \
  --data-path data/cleaned_dataset_train.csv \
  --model-output models/classification_model.pkl \
  --model-type random_forest \
  --random-seed 42
```

**Regression (Future Price)**:

```bash
python scripts/train_regression.py \
  --data-path data/cleaned_dataset_train.csv \
  --model-output models/regression_model.pkl \
  --model-type random_forest \
  --random-seed 42
```

### 5. Evaluate Models

```bash
python scripts/evaluate.py \
  --test-data data/cleaned_dataset_test.csv \
  --classification-model models/classification_model.pkl \
  --regression-model models/regression_model.pkl \
  --output-dir reports/
```

### 6. Run Streamlit Dashboard

```bash
streamlit run app/app.py
```

Access Dashboard: <http://localhost:8501>

---

## 🧪 Dataset

**Source**: `india_housing_prices.csv` (250,000 records)

**Features** (23 original + 8 engineered):

- **Location**: State, City, Locality
- **Property Details**: Type, BHK, Size (sqft), Price, Year Built
- **Amenities**: Nearby Schools/Hospitals, Transport, Parking, Security
- **Status**: Furnished, Availability, Owner Type

**Target Variables**:

1. **Good_Investment** (Classification): Binary label based on price competitiveness, BHK, amenities, and status
2. **Future_Price_5Y** (Regression): Estimated price in 5 years using 8% annual growth

---

## 📈 Model Performance

### Classification Metrics

| Metric      | Value  |
|-------------|--------|
| Accuracy    | TBD    |
| Precision   | TBD    |
| Recall      | TBD    |
| F1-Score    | TBD    |
| ROC-AUC     | TBD    |

### Regression Metrics

| Metric      | Value       |
|-------------|-------------|
| RMSE        | TBD Lakhs   |
| MAE         | TBD Lakhs   |
| R²          | TBD         |
| MAPE        | TBD%        |

*(Metrics will be populated after training)*

---

## 🎯 Key Decisions

### Colab vs Antigravity Distribution

**Google Colab** (Fast prototyping):

- Initial EDA and visualizations
- Hyperparameter exploration
- Model comparison experiments
- Outputs exported back to repo

**Antigravity IDE** (Production code):

- Git version control
- Deterministic training scripts
- MLflow server and tracking
- Streamlit dashboard
- Docker containerization
- All deployment artifacts

### Feature Engineering

- `Amenities_Score`: Sum of nearby schools + hospitals
- `Has_High_Transport`: Binary for high transport accessibility
- `Has_Security`, `Has_Parking`: Binary flags
- `Is_Ready_To_Move`: Binary for property availability
- `Furnished_Score`: Ordinal encoding (0-2)

### Target Creation

- **Classification**: Multi-factor score (price competitiveness + BHK + amenities + status)
- **Regression**: 8% compounded annual growth over 5 years

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t real-estate-advisor .

# Run container
docker run -p 8501:8501 real-estate-advisor
```

---

## 🔄 Reproducibility

All scripts use fixed random seeds (default: 42) to ensure reproducible results.

**Environment**:

- Python 3.10+
- See `requirements.txt` for package versions

**Commands** (in order):

1. Data preprocessing
2. MLflow server start
3. Model training (classification + regression)
4. Model evaluation
5. Streamlit dashboard

---

## ✅ Deliverables Checklist

- [✓] Cleaned dataset (`cleaned_dataset.csv`)
- [✓] Python preprocessing script (`preprocess.py`)
- [✓] Training scripts (classification & regression)
- [✓] MLflow experiment tracking
- [ ] Streamlit application
- [ ] Evaluation reports with plots
- [ ] Model artifacts (pickled models)
- [ ] Docker container
- [ ] Complete documentation

---

## 📝 Future Improvements

- Integrate XGBoost and LightGBM models
- Add hyperparameter tuning with Optuna
- Deploy to cloud (AWS/GCP)
- Add real-time data ingestion
- Implement A/B testing framework
- Create REST API for predictions

---

## 👨‍💻 Author

**Autonomous Implementation**  
Branch: `autonomous-impl`  
GitHub: <https://github.com/AdarshBennur/RealEstate-Investment-ML>

---

## 📄 License

This project is part of a capstone assignment for educational purposes.

---

**Last Updated**: December 2025
