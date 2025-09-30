import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Custom Page Config
st.set_page_config(
    page_title="Insurance Price Predictor",
    layout="wide",
    page_icon="💰",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    with open("tunned_gbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- Page Title ---
st.markdown("""
    <h2 style='color:#3E3E3E;'>💰 Insurance Charges Prediction App</h2>
    <p style='color:#6c757d;'>Estimate your insurance charges based on personal attributes using a trained Gradient Boosting model.</p>
    <hr style='border:1px solid #dee2e6'/>
""", unsafe_allow_html=True)

# --- Create 2 columns layout: left for input, right for output ---
left_col, right_col = st.columns([1.2, 2])

# --- LEFT: Input Form ---
with left_col:
    st.markdown("### 🧾 Enter Your Details", unsafe_allow_html=True)

    # Create 3 columns × 2 rows layout
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
    with col3:
        bmi = st.slider("BMI", 10.0, 50.0, value=25.0)

    col4, col5, col6 = st.columns(3)
    with col4:
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
    with col5:
        smoker = st.selectbox("Smoker", ["yes", "no"])
    with col6:
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    predict_btn = st.button("🔍 Predict Insurance Charges", use_container_width=True)

# --- RIGHT: Output & Visualization ---
with right_col:
    if predict_btn:
        # Create DataFrame
        input_df = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })

        prediction = model.predict(input_df)[0]

        st.markdown(f"""
            <div style='background-color: #e0f7fa; padding: 20px; border-radius: 8px;'>
                <h3 style='color: #00796b;'>💡 Estimated Insurance Charges:</h3>
                <h1 style='color: #4a148c;'>${prediction:,.2f}</h1>
            </div>
        """, unsafe_allow_html=True)

        # --- Feature Importance ---
        st.markdown("### 📊 Feature Importance", unsafe_allow_html=True)
        try:
            feature_names = model.named_steps['preprocess'].get_feature_names_out()
            importances = model.named_steps['regressor'].feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values()

            fig1, ax1 = plt.subplots(figsize=(8, 6))
            feat_imp.plot(kind='barh', ax=ax1, color='#4a148c')
            ax1.set_title("Feature Importance")
            ax1.set_xlabel("Importance")
            ax1.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig1)
        except Exception as e:
            st.info("Feature importances could not be loaded.")

        # --- Optional: Actual vs Predicted plot (if test data exists) ---
        try:
            x_test = pd.read_csv("X_test.csv")
            y_test = pd.read_csv("y_test.csv").squeeze()
            y_pred = model.predict(x_test)

            st.markdown("### 📈 Actual vs Predicted (Test Set)", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.scatter(y_test, y_pred, alpha=0.6, color='#00695c')
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax2.set_xlabel("Actual Charges")
            ax2.set_ylabel("Predicted Charges")
            ax2.set_title("Actual vs Predicted Charges")
            ax2.grid(True, linestyle='--', alpha=0.4)
            st.pyplot(fig2)
        except Exception as e:
            st.info("Could not load test data for actual vs predicted chart.")

# --- Footer ---
st.markdown("""<hr style='border:1px solid #dee2e6'/>""", unsafe_allow_html=True)
st.markdown("<small style='color:#6c757d;'>Built with 💙 using Streamlit</small>", unsafe_allow_html=True)

