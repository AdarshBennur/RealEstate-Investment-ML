# Project Implementation Plan

## Real Estate Investment Advisor - Autonomous Implementation

### Executive Summary

This document outlines the autonomous implementation strategy for the Real Estate Investment ML project. Tasks are divided between **Antigravity IDE** (for persistent infrastructure, version control, and deployment artifacts) and **Google Colab** (for GPU-accelerated prototyping and experimentation).

---

## 1. Component Task Distribution

### 1.1 Antigravity IDE Tasks (Persistent, Reproducible, Deployable)

- Git repository management and branching (`autonomous-impl`)
- Data cleaning pipeline (scripted preprocessing)
- Model training scripts (`train_classification.py`, `train_regression.py`)
- Model evaluation and reporting (`evaluate.py`)
- MLflow server and experiment tracking infrastructure
- Streamlit dashboard application
- Docker containerization
- CI/CD validation scripts
- Requirements management and documentation

**Rationale**: These tasks require deterministic execution, version control, local artifact storage, and integration with deployment infrastructure.

### 1.2 Google Colab Tasks (Fast Prototyping, GPU if needed)

- Initial EDA and visualization prototyping (`eda_prototypes.ipynb`)
- Feature engineering experiments
- Hyperparameter exploration (if heavy tuning needed)
- Model comparison experiments
- Export best parameters/preprocessors back to repo

**Rationale**: Colab provides free compute and rapid iteration for exploratory analysis. All outputs will be exported to repo for reproducibility.

---

## 2. Implementation Roadmap

### Phase 1: Setup & Data Preparation (Antigravity + Colab)

**Location**: Primarily Antigravity, EDA in Colab

- [x] Clone/verify repository structure
- [ ] Create `autonomous-impl` branch
- [ ] Quick data validation script in Antigravity
- [ ] Create `/notebooks/eda_prototypes.ipynb` for Colab
- [ ] Run EDA in Colab (20 questions from project guide)
- [ ] Export cleaned data to `/data/cleaned_dataset.csv`
- [ ] Create data preprocessing pipeline (`/scripts/preprocess.py`)
- [ ] Serialize preprocessing artifacts (scalers, encoders) with joblib

**Deliverables**:

- `/data/cleaned_dataset.csv`
- `/scripts/preprocess.py`
- `/artifacts/preprocessor.pkl`
- `/notebooks/eda_prototypes.ipynb` (with outputs)

---

### Phase 2: Target Variable Engineering (Antigravity)

**Location**: Antigravity

Create target variables as per project specifications:

**Regression Target: `Future_Price_5Y`**

- Formula: `Price_in_Lakhs * 1.08^5` (8% annual growth)
- Alternative: Location/property-type specific growth rates
- Script: Add this as feature in preprocessing

**Classification Target: `Good_Investment`**

- Binary label based on:
  - Price ≤ median price per city
  - Price_per_SqFt ≤ median per city
  - Multi-factor score (BHK≥3, RERA if exists, ready-to-move)
- Script: Logic in preprocessing pipeline

**Deliverables**:

- Updated `/scripts/preprocess.py` with target generation
- Updated `/data/cleaned_dataset.csv` with targets

---

### Phase 3: Model Development (Colab for prototyping → Antigravity for final scripts)

**Location**: Initial experiments in Colab, finalized scripts in Antigravity

#### 3.1 Colab Prototyping

- Test multiple models (Logistic Regression, Random Forest, XGBoost for both tasks)
- Tune hyperparameters
- Export best hyperparams to JSON
- Save best model weights

#### 3.2 Antigravity Production Scripts

Create deterministic training scripts:

**`/scripts/train_classification.py`**

- CLI args: `--data-path`, `--random-seed`, `--params-json`
- Load preprocessor
- Train classification model (Good_Investment)
- Log to MLflow: params, metrics (Accuracy, Precision, Recall, ROC-AUC), artifacts
- Save model to MLflow or local

**`/scripts/train_regression.py`**

- CLI args: `--data-path`, `--random-seed`, `--params-json`
- Load preprocessor
- Train regression model (Future_Price_5Y)
- Log to MLflow: params, metrics (RMSE, MAE, R²), artifacts
- Save model

**Deliverables**:

- `/scripts/train_classification.py`
- `/scripts/train_regression.py`
- `/configs/model_params.json` (best hyperparams from Colab)
- MLflow logged experiments

---

### Phase 4: MLflow Integration (Antigravity)

**Location**: Antigravity

- [ ] Start local MLflow server: `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000`
- [ ] Ensure training scripts log to `MLFLOW_TRACKING_URI` (env var)
- [ ] Run classification and regression training
- [ ] Register best models in MLflow Model Registry
- [ ] Tag Production version

**Deliverables**:

- Running MLflow server
- Registered models in Model Registry
- Production-tagged model versions

---

### Phase 5: Model Evaluation & Reporting (Antigravity)

**Location**: Antigravity

**`/scripts/evaluate.py`**

- Load test dataset
- Load Production model from MLflow
- Generate predictions
- Compute metrics
- Save plots to `/reports/`:
  - Confusion matrix (classification)
  - Feature importance
  - ROC curve
  - Actual vs Predicted (regression)
  - Residual plot

**Deliverables**:

- `/scripts/evaluate.py`
- `/reports/*.png` (evaluation plots)
- `/reports/model_metrics.json`

---

### Phase 6: Streamlit Dashboard (Antigravity)

**Location**: Antigravity

**`/app/app.py`**
Features:

- Load Production model from MLflow (fallback to local pickle)
- User input form:
  - Property details (BHK, Size, Location, Property_Type, etc.)
  - Filters (area, price range, BHK)
- Predictions:
  - Classification: "Good Investment?" with confidence/probability
  - Regression: "Estimated Price in 5 Years"
- Visualizations:
  - Feature importance
  - Location-wise heatmaps
  - Price trends by city
  - Model confidence scores

**Deliverables**:

- `/app/app.py`
- Run locally: `streamlit run app/app.py`

---

### Phase 7: Deployment Artifacts (Antigravity)

**Location**: Antigravity

#### 7.1 Docker

**`/Dockerfile`**

- Base: `python:3.10-slim`
- Install requirements
- Expose MLflow port (5000) and Streamlit port (8501)
- CMD to run Streamlit

#### 7.2 Requirements

**`/requirements.txt`**

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
mlflow>=2.10.0
streamlit>=1.30.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

#### 7.3 Quick Check Script

**`/quick_check.sh`**

- Install requirements
- Run quick training with `--quick` flag (small subset)
- Verify MLflow logs created
- Start Streamlit and check it runs

#### 7.4 Pipeline Validation

**`/scripts/check_pipeline.py`**

- Unit tests for preprocessing
- Validate model input/output shapes
- Test predict functions

**Deliverables**:

- `/Dockerfile`
- `/requirements.txt`
- `/quick_check.sh`
- `/scripts/check_pipeline.py`

---

### Phase 8: Documentation & Final Repo Structure (Antigravity)

**Location**: Antigravity

**`/README.md`**
Sections:

- Project Overview
- Architecture Diagram
- Data Pipeline
- Model Training
  - Classification
  - Regression
- MLflow Tracking
- Streamlit Dashboard
- Reproduction Steps:
  1. Install requirements
  2. Start MLflow server
  3. Run preprocessing
  4. Train models
  5. Evaluate models
  6. Run Streamlit app
- Colab Usage Notes (where and why Colab was used)
- Deliverables Checklist
- Deployment Instructions

**Deliverables**:

- Comprehensive `/README.md`
- Clean repo structure
- All code committed to `autonomous-impl` branch

---

## 3. Final Repository Structure

```
5_RealEstate-Investment/
├── ProjectDescription/
│   ├── Project implementation.md
│   └── Screenshot 2025-12-11 at 4.00.54 PM.png
├── data/
│   ├── india_housing_prices.csv (original)
│   └── cleaned_dataset.csv (processed)
├── notebooks/
│   └── eda_prototypes.ipynb (Colab-friendly)
├── scripts/
│   ├── preprocess.py
│   ├── train_classification.py
│   ├── train_regression.py
│   ├── evaluate.py
│   └── check_pipeline.py
├── app/
│   └── app.py (Streamlit)
├── configs/
│   └── model_params.json
├── artifacts/
│   └── preprocessor.pkl
├── reports/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curve.png
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   └── model_metrics.json
├── mlruns/ (MLflow artifacts)
├── mlflow.db (SQLite backend)
├── requirements.txt
├── Dockerfile
├── README.md
├── ProjectImplementationPlan.md (this file)
├── quick_check.sh
└── .gitignore
```

---

## 4. Execution Timeline & Progress Tracking

### Milestones

1. ✅ Planning Complete
2. ⏳ Data Preparation & EDA
3. ⏳ Model Training Pipeline
4. ⏳ MLflow Integration
5. ⏳ Streamlit Dashboard
6. ⏳ Deployment Artifacts
7. ⏳ Documentation & Push

---

## 5. Key Decisions & Rationale

### Why Colab for EDA?

- Fast iteration on visualizations
- No local resource usage
- Easy sharing of prototype notebooks
- All outputs exported back to repo for reproducibility

### Why Antigravity for Training Scripts?

- Deterministic execution with fixed seeds
- Version control integration
- Direct access to MLflow server
- Easier debugging and testing
- Deployment-ready code structure

### Model Selection

- Classification: Start with Random Forest (good baseline), test XGBoost
- Regression: Start with Linear Regression, progress to Random Forest/XGBoost
- Rationale: Ensemble methods handle mixed data types well

### MLflow Strategy

- Local SQLite backend for development
- Easy migration to PostgreSQL for production
- Model registry ensures proper versioning
- Production tags for deployment clarity

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Large dataset (250K rows) | Use Colab for heavy EDA, sample for quick checks |
| Missing values in critical features | Robust imputation strategy in preprocessing |
| Model overfitting | Cross-validation, regularization, proper train/test split |
| MLflow port conflicts | Use env vars, document port config |
| Streamlit memory issues | Optimize data loading, cache models |

---

## 7. Success Criteria

- [✓] Cleaned dataset available
- [ ] Classification model: Accuracy > 75%, ROC-AUC > 0.80
- [ ] Regression model: R² > 0.75, RMSE < 50 lakhs
- [ ] MLflow server running with logged experiments
- [ ] Streamlit app functional with predictions
- [ ] Dockerfile builds successfully
- [ ] README with complete reproduction steps
- [ ] All code committed to `autonomous-impl` branch

---

**Autonomous Execution Status**: Ready to proceed with Phase 1
**Next Action**: Create branch and start data validation
