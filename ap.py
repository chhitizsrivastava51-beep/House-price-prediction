import pandas as pd
import streamlit as st
import joblib

# Load the model
model = joblib.load('xgb_model.jb')

st.title("House Price Prediction")
st.write("Enter the details below to predict the House Price")

inputs = [
    "OverallQual", "GrLivArea", "GarageArea", "1stFlrSF", "FullBath", 
    "YearBuilt", "YearRemodAdd", "MasVnrArea", "Fireplaces", "BsmtFinSF1", 
    "LotFrontage", "WoodDeckSF", "OpenPorchSF", "LotArea", "CentralAir"
]

input_data = {}

for feature in inputs:
    if feature == "CentralAir":
        input_data[feature] = st.selectbox(f"{feature}", options=["Yes", "No"], index=0)
    else:
        # Use a higher step for wider ranges if necessary
        input_data[feature] = st.number_input(
            f"{feature}", 
            value=0.0, 
            step=1.0 if feature in ["OverallQual", "FullBath", "Fireplaces"] else 10.0 
        )

if st.button("Predict Price"):
    # Convert 'Yes'/'No' to 1/0 for the model
    input_data["CentralAir"] = 1 if input_data["CentralAir"] == "Yes" else 0
    
    # Create DataFrame and enforce original column ordering
    input_df = pd.DataFrame([input_data], columns=inputs)
    
    # Make Prediction
    predictions = model.predict(input_df)
    
    # Display Result
    st.success(f"Predicted House Price: ${predictions[0]:,.2f}")
