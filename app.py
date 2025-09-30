import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
@st.cache_data
def load_data():
    return pd.read_pickle("cleaned_data.pkl")
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
if predict_btn:
    # Prepare input data for prediction
    input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
            })
    
    prediction = model.predict(input_df)[0]

    with right_col:
        st.markdown(f"""
            <div style='background-color: #e0f7fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;'>
                 <h3 style='color: #00796b;'>💡 Estimated Insurance Charges:</h3>
                 <h1 style='color: #4a148c;'>${prediction:,.2f}</h1>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <p style='color:#555555; font-size:16px;'>
                Below are the distributions of important features in the dataset. The red dashed line shows your input value compared to the overall population.
        </p>
    """, unsafe_allow_html=True)
    
            # Load dataset to plot distributions (make sure you have df loaded in your app)
          
    
    df = load_data()
    
     # If df not defined, show warning
    if 'df' not in globals():
        st.warning("Dataset (df) is not loaded. Can't show feature distributions.")
    else:
        user_inputs = {'age': age,'bmi': bmi,'children': children, 'charges': prediction  }
    features = list(user_inputs.keys())
    n_features = len(features)
    
    fig, axs = plt.subplots(1, 4, figsize=(6*n_features, 6))
    for i, feature in enumerate(features):
        sns.histplot(df[feature], bins=30, kde=True, color='skyblue', ax=axs[i])
        axs[i].axvline(user_inputs[feature], color='red', linestyle='--', linewidth=2)
        axs[i].text(user_inputs[feature], axs[i].get_ylim()[1]*0.9,f'Your {feature}:\n{user_inputs[feature]:.2f}',color='red', fontsize=10, ha='center')
        axs[i].set_title(f"{feature.capitalize()} Distribution")
        axs[i].set_xlabel(feature.capitalize())
        axs[i].set_ylabel("Frequency")
    
    plt.tight_layout()
    st.pyplot(fig)

# --- Footer ---
st.markdown("""<hr style='border:1px solid #dee2e6'/>""", unsafe_allow_html=True)
st.markdown("<small style='color:#6c757d;'>Built with 💙 using Streamlit</small>", unsafe_allow_html=True)
