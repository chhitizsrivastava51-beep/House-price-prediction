import pandas as pd
import streamlit as st
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Property Valuation",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load("xgb_model.jb")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🏠 AI Property Valuation Assistant")

st.markdown(
    "### Machine Learning powered property price prediction"
)

st.info(
    "Enter property characteristics below to generate an AI-based "
    "estimated valuation and understand the major factors influencing it."
)

# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------

inputs = [
    "OverallQual",
    "GrLivArea",
    "GarageArea",
    "1stFlrSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "Fireplaces",
    "BsmtFinSF1",
    "LotFrontage",
    "WoodDeckSF",
    "OpenPorchSF",
    "LotArea",
    "CentralAir"
]

# Human-readable names
feature_labels = {
    "OverallQual": "⭐ Overall Quality",
    "GrLivArea": "🏠 Living Area (sq ft)",
    "GarageArea": "🚗 Garage Area (sq ft)",
    "1stFlrSF": "🏢 First Floor Area (sq ft)",
    "FullBath": "🛁 Full Bathrooms",
    "YearBuilt": "📅 Year Built",
    "YearRemodAdd": "🔨 Year Remodeled",
    "MasVnrArea": "🧱 Masonry Area (sq ft)",
    "Fireplaces": "🔥 Fireplaces",
    "BsmtFinSF1": "🏚️ Basement Area (sq ft)",
    "LotFrontage": "📐 Lot Frontage",
    "WoodDeckSF": "🌳 Wood Deck Area",
    "OpenPorchSF": "☀️ Open Porch Area",
    "LotArea": "🌱 Lot Area (sq ft)",
    "CentralAir": "❄️ Central Air Conditioning"
}

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.subheader("🏡 Property Details")

input_data = {}

# Row 1
col1, col2, col3 = st.columns(3)

with col1:
    input_data["OverallQual"] = st.number_input(
        "⭐ Overall Quality",
        min_value=0.0,
        value=5.0,
        step=1.0
    )

with col2:
    input_data["GrLivArea"] = st.number_input(
        "🏠 Living Area (sq ft)",
        min_value=0.0,
        value=1500.0,
        step=10.0
    )

with col3:
    input_data["GarageArea"] = st.number_input(
        "🚗 Garage Area (sq ft)",
        min_value=0.0,
        value=400.0,
        step=10.0
    )

# Row 2
col1, col2, col3 = st.columns(3)

with col1:
    input_data["1stFlrSF"] = st.number_input(
        "🏢 First Floor Area (sq ft)",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )

with col2:
    input_data["FullBath"] = st.number_input(
        "🛁 Full Bathrooms",
        min_value=0.0,
        value=2.0,
        step=1.0
    )

with col3:
    input_data["YearBuilt"] = st.number_input(
        "📅 Year Built",
        min_value=1800.0,
        max_value=2026.0,
        value=2000.0,
        step=1.0
    )

# Row 3
col1, col2, col3 = st.columns(3)

with col1:
    input_data["YearRemodAdd"] = st.number_input(
        "🔨 Year Remodeled",
        min_value=1800.0,
        max_value=2026.0,
        value=2000.0,
        step=1.0
    )

with col2:
    input_data["MasVnrArea"] = st.number_input(
        "🧱 Masonry Area (sq ft)",
        min_value=0.0,
        value=0.0,
        step=10.0
    )

with col3:
    input_data["Fireplaces"] = st.number_input(
        "🔥 Fireplaces",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

# Row 4
col1, col2, col3 = st.columns(3)

with col1:
    input_data["BsmtFinSF1"] = st.number_input(
        "🏚️ Basement Area (sq ft)",
        min_value=0.0,
        value=500.0,
        step=10.0
    )

with col2:
    input_data["LotFrontage"] = st.number_input(
        "📐 Lot Frontage",
        min_value=0.0,
        value=60.0,
        step=5.0
    )

with col3:
    input_data["WoodDeckSF"] = st.number_input(
        "🌳 Wood Deck Area",
        min_value=0.0,
        value=0.0,
        step=10.0
    )

# Row 5
col1, col2, col3 = st.columns(3)

with col1:
    input_data["OpenPorchSF"] = st.number_input(
        "☀️ Open Porch Area",
        min_value=0.0,
        value=50.0,
        step=10.0
    )

with col2:
    input_data["LotArea"] = st.number_input(
        "🌱 Lot Area (sq ft)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

with col3:
    central_air = st.selectbox(
        "❄️ Central Air Conditioning",
        ["Yes", "No"]
    )

input_data["CentralAir"] = 1 if central_air == "Yes" else 0

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

st.divider()

predict_button = st.button(
    "🔮 Predict Property Price",
    type="primary",
    use_container_width=True
)

if predict_button:

    # Maintain exact feature order used by model
    input_df = pd.DataFrame(
        [input_data],
        columns=inputs
    )

    # Prediction
    prediction = model.predict(input_df)[0]

    # ------------------------------------------------
    # RESULT
    # ------------------------------------------------

    st.success("Prediction generated successfully!")

    st.subheader("💰 Estimated Property Value")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Price",
            f"${prediction:,.0f}"
        )

    with col2:
        st.metric(
            "Model Output",
            f"{prediction:,.2f}"
        )

    # ------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------

    st.divider()

    st.subheader("🧠 Why did the model predict this price?")

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": inputs,
            "Importance": importance
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        # Top 5
        top_features = importance_df.head(5).copy()

        top_features["Feature"] = top_features["Feature"].map(
            feature_labels
        )

        st.markdown(
            "The model learned the relative importance of different "
            "property characteristics from the training data."
        )

        st.bar_chart(
            top_features.set_index("Feature")["Importance"]
        )

        # ------------------------------------------------
        # AI INSIGHT
        # ------------------------------------------------

        st.subheader("🤖 AI Property Insight")

        top_feature = importance_df.iloc[0]["Feature"]
        second_feature = importance_df.iloc[1]["Feature"]

        readable_top = feature_labels.get(
            top_feature,
            top_feature
        )

        readable_second = feature_labels.get(
            second_feature,
            second_feature
        )

        st.info(
            f"According to the trained model, **{readable_top}** "
            f"is the strongest contributing feature among the "
            f"provided inputs, followed by **{readable_second}**. "
            f"The final valuation is generated by combining the "
            f"patterns learned across all input features."
        )

    else:

        st.warning(
            "Feature importance is not available for this model."
        )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "⚠️ This is an AI/ML-based estimated valuation. "
    "Actual property prices depend on market conditions, location "
    "and other factors."
)

st.caption(
    "Prototype developed for Build in AI for India — Delhi Edition 2026."
)
