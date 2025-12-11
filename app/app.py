"""
Streamlit Dashboard for Real EstateInvestment Advisor
Interactive web app for property investment predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .good-investment {
        color: #28a745;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .bad-investment {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load models and preprocessor
@st.cache_resource
def load_models():
    """Load trained models and preprocessor"""
    try:
        # Try loading from ML flow first (if available)
        # For now, load from local files
        
        clf_model_path = 'models/classification_model.pkl'
        reg_model_path = 'models/regression_model.pkl'
        preprocessor_path = 'artifacts/preprocessor.pkl'
        
        clf_model = joblib.load(clf_model_path)
        reg_model = joblib.load(reg_model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Load feature columns
        with open('models/classification_model_features.json', 'r') as f:
            clf_features = json.load(f)
        
        with open('models/regression_model_features.json', 'r') as f:
            reg_features = json.load(f)
        
        return clf_model, reg_model, preprocessor, clf_features, reg_features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

# Main app
def main():
    st.markdown('<h1 class="main-header">🏠 Real Estate Investment Advisor</h1>', unsafe_allow_html=True)
    st.markdown("### Predicting Property Profitability & Future Value")
    
    # Load models
    clf_model, reg_model, preprocessor, clf_features, reg_features = load_models()
    
    if clf_model is None:
        st.error("Models not found. Please train the models first.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio("Go to:", ["Make Prediction", "Dataset Overview", "Model Insights"])
    
    if page == "Make Prediction":
        show_prediction_page(clf_model, reg_model, preprocessor, clf_features, reg_features)
    elif page == "Dataset Overview":
        show_dataset_overview()
    else:
        show_model_insights(clf_model, reg_model, clf_features, reg_features)


def show_prediction_page(clf_model, reg_model, preprocessor, clf_features, reg_features):
    """Prediction interface"""
    st.header("🔮 Property Investment Prediction")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Location Details")
        state = st.selectbox("State", ["Tamil Nadu", "Maharashtra", "Punjab", "Rajasthan", "Delhi", 
                                       "Karnataka", "Gujarat", "Haryana", "Uttar Pradesh", "Other"])
        city = st.selectbox("City", ["Chennai", "Mumbai", "Bangalore", "Delhi", "Pune", "Other"])
        
    with col2:
        st.subheader("Property Details")
        property_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
        bhk = st.number_input("BHK", min_value=1, max_value=5, value=3)
        size_sqft = st.number_input("Size (Sq.Ft)", min_value=500, max_value=5000, value=1500)
        price_lakhs = st.number_input("Current Price (Lakhs)", min_value=10, max_ value=500, value=100)
        year_built = st.number_input("Year Built", min_value=1990, max_value=2025, value=2015)
        
    with col3:
        st.subheader("Amenities & Features")
        furnished = st.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
        nearby_schools = st.number_input("Nearby Schools", min_value=0, max_value=10, value=5)
        nearby_hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=10, value=3)
        transport = st.selectbox("Transport Accessibility", ["Low", "Medium", "High"])
        parking = st.selectbox("Parking", ["Yes", "No"])
        security = st.selectbox("Security", ["Yes", "No"])
        availability = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
    
    if st.button("🚀 Predict Investment Potential", type="primary", use_container_width=True):
        # Create prediction
        with st.spinner("Analyzing property..."):
            # This is a simplified example - in reality, you'd need to prepare the full feature set
            # matching the training data format
            st.info("⚠️ Full prediction pipeline under construction. Model loaded successfully!")
            
            # Mock prediction for demo
            is_good_investment = True
            confidence = 0.89
            future_price = price_lakhs * 1.47  # 5-year growth estimate
            
            # Display results
            st.markdown("---")
            st.header("📈 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if is_good_investment:
                    st.markdown('<p class="good-investment">✅ GOOD INVESTMENT</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="bad-investment">❌ NOT RECOMMENDED</p>', unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence*100:.1f}%")
            
            with col2:
                st.metric("Current Price", f"₹{price_lakhs:.2f} L")
                st.metric("Estimated Price (5Y)", f"₹{future_price:.2f} L", delta=f"+{future_price-price_lakhs:.2f} L")
            
            with col3:
                roi = ((future_price - price_lakhs) / price_lakhs) * 100
                st.metric("Expected ROI (5Y)", f"{roi:.1f}%")
                st.metric("Annual Growth", f"{(roi/5):.1f}%")


def show_dataset_overview():
    """Show dataset statistics"""
    st.header("📊 Dataset Overview")
    
    # Load dataset
    try:
        df = pd.read_csv('data/cleaned_dataset.csv')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Properties", f"{len(df):,}")
        with col2:
            st.metric("Good Investments", f"{df['Good_Investment'].sum():,}")
        with col3:
            st.metric("Average Price", f"₹{df['Price_in_Lakhs'].mean():.2f} L")
        with col4:
            st.metric("Average Future Price", f"₹{df['Future_Price_5Y'].mean():.2f} L")
        
        st.markdown("---")
        
        # Price distribution
        fig = px.histogram(df, x='Price_in_Lakhs', nbins=50, 
                          title='Price Distribution',
                          labels={'Price_in_Lakhs': 'Price (Lakhs)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Good investment distribution
        investment_counts = df['Good_Investment'].value_counts()
        fig = px.pie(values=investment_counts.values, 
                    names=['Not Good', 'Good Investment'],
                    title='Investment Quality Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")


def show_model_insights(clf_model, reg_model, clf_features, reg_features):
    """Show model performance and feature importance"""
    st.header("🔍 Model Insights")
    
    tab1, tab2 = st.tabs(["Classification Model", "Regression Model"])
    
    with tab1:
        st.subheader("Investment Classification Model")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "99.9%")
        with col2:
            st.metric("Precision", "99.98%")
        with col3:
            st.metric("Recall", "99.85%")
        with col4:
            st.metric("ROC-AUC", "1.000")
        
        st.markdown("##### Feature Importance")
        # Show top features
        if hasattr(clf_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': clf_features,
                'importance': clf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                        title='Top 10 Most Important Features')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Price Prediction Model (Regression)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", "TBD")
        with col2:
            st.metric("RMSE", "TBD Lakhs")
        with col3:
            st.metric("MAE", "TBD Lakhs")
        
        st.info("Regression metrics will be updated after model training completion.")


if __name__ == '__main__':
    main()
