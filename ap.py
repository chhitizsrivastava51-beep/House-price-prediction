import os
import pandas as pd
import streamlit as st
import joblib
from dotenv import load_dotenv

# Import Gnani TTS module
from gnani_tts import (
    SUPPORTED_LANGUAGES,
    DEFAULT_ENDPOINT,
    format_inr,
    generate_valuation_script,
    synthesize_speech
)

# Load environment variables from .env if present
load_dotenv()

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Property Valuation & Voice Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM STYLING (Modern Glassmorphism & Cards)
# ---------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.12));
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    .badge-gnani {
        background-color: #0d6efd;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-fallback {
        background-color: #ffc107;
        color: #212529;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    .voice-box {
        background: rgba(13, 110, 253, 0.08);
        border: 1px solid rgba(13, 110, 253, 0.25);
        border-radius: 12px;
        padding: 18px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# MODEL CACHING
# ---------------------------------------------------
@st.cache_resource(show_spinner="Loading valuation model...")
def get_model():
    """Load machine learning model once and cache it in memory."""
    try:
        return joblib.load("xgb_model.jb")
    except Exception as e:
        st.error(f"⚠️ Failed to load model file 'xgb_model.jb': {e}")
        return None

model = get_model()

# ---------------------------------------------------
# FEATURES DEFINITIONS
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
    "BsmtFinSF1": "🏚️ Basement Finished Area (sq ft)",
    "LotFrontage": "📐 Lot Frontage (ft)",
    "WoodDeckSF": "🌳 Wood Deck Area (sq ft)",
    "OpenPorchSF": "☀️ Open Porch Area (sq ft)",
    "LotArea": "🌱 Lot Area (sq ft)",
    "CentralAir": "❄️ Central Air Conditioning"
}

# Quick Preset Configurations
PRESETS = {
    "🏡 Budget Family Home": {
        "OverallQual": 5.0, "GrLivArea": 1300.0, "GarageArea": 350.0,
        "1stFlrSF": 900.0, "FullBath": 1.0, "YearBuilt": 1975.0,
        "YearRemodAdd": 1990.0, "MasVnrArea": 0.0, "Fireplaces": 0.0,
        "BsmtFinSF1": 350.0, "LotFrontage": 55.0, "WoodDeckSF": 0.0,
        "OpenPorchSF": 30.0, "LotArea": 6500.0, "CentralAir": "Yes"
    },
    "🏠 Modern Suburban Home": {
        "OverallQual": 7.0, "GrLivArea": 1800.0, "GarageArea": 500.0,
        "1stFlrSF": 1100.0, "FullBath": 2.0, "YearBuilt": 2005.0,
        "YearRemodAdd": 2010.0, "MasVnrArea": 150.0, "Fireplaces": 1.0,
        "BsmtFinSF1": 600.0, "LotFrontage": 70.0, "WoodDeckSF": 120.0,
        "OpenPorchSF": 60.0, "LotArea": 9500.0, "CentralAir": "Yes"
    },
    "🏰 Luxury Villa Estate": {
        "OverallQual": 9.0, "GrLivArea": 3200.0, "GarageArea": 850.0,
        "1stFlrSF": 1900.0, "FullBath": 3.0, "YearBuilt": 2018.0,
        "YearRemodAdd": 2021.0, "MasVnrArea": 350.0, "Fireplaces": 2.0,
        "BsmtFinSF1": 1200.0, "LotFrontage": 95.0, "WoodDeckSF": 250.0,
        "OpenPorchSF": 150.0, "LotArea": 16000.0, "CentralAir": "Yes"
    }
}

# ---------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------
if "preset_selected" not in st.session_state:
    st.session_state.preset_selected = "🏠 Modern Suburban Home"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}

# ---------------------------------------------------
# SIDEBAR: GNANI VOICE AI & SETTINGS
# ---------------------------------------------------
with st.sidebar:
    st.header("🎙️ Gnani.ai Voice AI")
    st.caption("Indic Speech & Natural Voice Synthesis")

    # API Configuration
    env_key = os.getenv("GNANI_API_KEY", "")
    env_token = os.getenv("GNANI_TOKEN", "")
    env_endpoint = os.getenv("GNANI_TTS_ENDPOINT", DEFAULT_ENDPOINT)

    st.markdown("#### ⚙️ Credentials")
    gnani_api_key = st.text_input(
        "Gnani API Key",
        value=env_key,
        type="password",
        help="Enter your API Key from Gnani.ai developer console"
    )
    gnani_token = st.text_input(
        "Gnani Access Token (Optional)",
        value=env_token,
        type="password",
        help="Enter Gnani Access Token if required by your subscription"
    )
    gnani_endpoint = st.text_input(
        "Gnani TTS Endpoint",
        value=env_endpoint,
        help="Default: https://tts.gnani.ai/api/v1/tts"
    )

    is_gnani_configured = bool(gnani_api_key.strip() or gnani_token.strip())
    if is_gnani_configured:
        st.markdown('<span class="badge-gnani">🟢 Gnani.ai Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-fallback">🟡 Fallback Voice Engine Active</span>', unsafe_allow_html=True)
        st.caption("💡 No Gnani API key provided yet. The app will automatically use the built-in voice synthesizer so audio plays seamlessly!")

    st.divider()

    # Voice test button
    st.markdown("#### 🧪 Test Voice Synthesis")
    test_lang = st.selectbox(
        "Test Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda code: SUPPORTED_LANGUAGES[code]["name"],
        index=0
    )
    
    if st.button("🔊 Test Voice Greeting", use_container_width=True):
        with st.spinner("Generating test speech..."):
            greeting = "नमस्ते! ज्ञानी एआई वॉइस असिस्टेंट में आपका स्वागत है।" if test_lang.startswith("hi") else "Hello! Welcome to Gnani AI voice assistant."
            audio_data, mime_type, engine, msg = synthesize_speech(
                text=greeting,
                lang=test_lang,
                voice="female",
                api_key=gnani_api_key,
                token=gnani_token,
                endpoint=gnani_endpoint
            )
            if audio_data:
                st.audio(audio_data, format=mime_type)
                st.success(f"Synthesized with {engine.upper()} engine!")
            else:
                st.error(f"Voice generation failed: {msg}")

    st.divider()

    # Currency preference
    st.markdown("#### 💱 Currency Localization")
    currency_mode = st.radio(
        "Primary Currency",
        options=["INR (₹) - Indian Rupees", "USD ($) - US Dollars"],
        index=0,
        help="Prototype optimized for Build in AI for India — Delhi Edition 2026"
    )
    usd_to_inr_rate = st.number_input(
        "USD to INR Exchange Rate",
        min_value=50.0,
        max_value=120.0,
        value=83.50,
        step=0.5
    )

# ---------------------------------------------------
# MAIN APP HEADER
# ---------------------------------------------------
st.title("🏠 AI Property Valuation & Voice Assistant")
st.markdown(
    "### Machine Learning powered property valuation with **Gnani.ai Indic Voice Synthesis**"
)

st.info(
    "Enter property characteristics or choose a 1-click preset below. "
    "Get AI-based valuations, interpret driving factors, and **listen to valuation reports in your preferred Indian language**."
)

# ---------------------------------------------------
# PRESET SELECTION
# ---------------------------------------------------
st.subheader("⚡ 1-Click Demo Presets")
p_cols = st.columns(len(PRESETS))

for idx, (p_name, p_values) in enumerate(PRESETS.items()):
    if p_cols[idx].button(p_name, use_container_width=True):
        st.session_state.preset_selected = p_name

current_preset = PRESETS.get(st.session_state.preset_selected, PRESETS["🏠 Modern Suburban Home"])

# ---------------------------------------------------
# PROPERTY DETAILS INPUTS (Organized in 3 Tabs)
# ---------------------------------------------------
st.subheader("🏡 Property Characteristics")

tab1, tab2, tab3 = st.tabs(["📐 Area & Layout", "🏗️ Quality & Construction", "🌳 Exterior & Facilities"])

input_data = {}

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["GrLivArea"] = st.number_input(
            "🏠 Above Grade Living Area (sq ft)",
            min_value=100.0,
            max_value=10000.0,
            value=float(current_preset["GrLivArea"]),
            step=25.0
        )
        input_data["1stFlrSF"] = st.number_input(
            "🏢 First Floor Area (sq ft)",
            min_value=100.0,
            max_value=8000.0,
            value=float(current_preset["1stFlrSF"]),
            step=25.0
        )
    with col2:
        input_data["BsmtFinSF1"] = st.number_input(
            "🏚️ Finished Basement Area (sq ft)",
            min_value=0.0,
            max_value=5000.0,
            value=float(current_preset["BsmtFinSF1"]),
            step=25.0
        )
        input_data["GarageArea"] = st.number_input(
            "🚗 Garage Area (sq ft)",
            min_value=0.0,
            max_value=2000.0,
            value=float(current_preset["GarageArea"]),
            step=25.0
        )
    with col3:
        input_data["FullBath"] = st.number_input(
            "🛁 Full Bathrooms",
            min_value=0.0,
            max_value=8.0,
            value=float(current_preset["FullBath"]),
            step=1.0
        )
        input_data["Fireplaces"] = st.number_input(
            "🔥 Fireplaces",
            min_value=0.0,
            max_value=5.0,
            value=float(current_preset["Fireplaces"]),
            step=1.0
        )

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["OverallQual"] = st.slider(
            "⭐ Overall Material & Finish Quality (1 to 10)",
            min_value=1.0,
            max_value=10.0,
            value=float(current_preset["OverallQual"]),
            step=1.0,
            help="1 = Very Poor, 5 = Average, 10 = Very Excellent"
        )
        input_data["MasVnrArea"] = st.number_input(
            "🧱 Masonry Veneer Area (sq ft)",
            min_value=0.0,
            max_value=2000.0,
            value=float(current_preset["MasVnrArea"]),
            step=20.0
        )
    with col2:
        input_data["YearBuilt"] = st.number_input(
            "📅 Year Built",
            min_value=1850.0,
            max_value=2026.0,
            value=float(current_preset["YearBuilt"]),
            step=1.0
        )
    with col3:
        input_data["YearRemodAdd"] = st.number_input(
            "🔨 Year Remodeled / Addition",
            min_value=1850.0,
            max_value=2026.0,
            value=float(current_preset["YearRemodAdd"]),
            step=1.0
        )

with tab3:
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["LotArea"] = st.number_input(
            "🌱 Lot Area (sq ft)",
            min_value=500.0,
            max_value=150000.0,
            value=float(current_preset["LotArea"]),
            step=100.0
        )
        input_data["LotFrontage"] = st.number_input(
            "📐 Linear Feet of Street Frontage",
            min_value=0.0,
            max_value=400.0,
            value=float(current_preset["LotFrontage"]),
            step=5.0
        )
    with col2:
        input_data["WoodDeckSF"] = st.number_input(
            "🌳 Wood Deck Area (sq ft)",
            min_value=0.0,
            max_value=2000.0,
            value=float(current_preset["WoodDeckSF"]),
            step=20.0
        )
        input_data["OpenPorchSF"] = st.number_input(
            "☀️ Open Porch Area (sq ft)",
            min_value=0.0,
            max_value=1000.0,
            value=float(current_preset["OpenPorchSF"]),
            step=10.0
        )
    with col3:
        ca_val = current_preset.get("CentralAir", "Yes")
        ca_choice = st.selectbox(
            "❄️ Central Air Conditioning",
            options=["Yes", "No"],
            index=0 if ca_val == "Yes" else 1
        )
        input_data["CentralAir"] = 1.0 if ca_choice == "Yes" else 0.0

# ---------------------------------------------------
# VALIDATION CHECKS
# ---------------------------------------------------
if input_data["YearRemodAdd"] < input_data["YearBuilt"]:
    st.warning("⚠️ Note: Year Remodeled is earlier than Year Built. The remodel year should typically be equal to or greater than Year Built.")

if input_data["1stFlrSF"] > input_data["GrLivArea"]:
    st.warning("⚠️ Note: First Floor Area exceeds Total Living Area. Typically First Floor Area <= Total Living Area.")

# ---------------------------------------------------
# PREDICTION ACTION
# ---------------------------------------------------
st.divider()

if st.button("🔮 Generate AI Property Valuation", type="primary", use_container_width=True):
    if model is None:
        st.error("Cannot run prediction: Model is not loaded.")
    else:
        # Prepare exact feature vector
        input_df = pd.DataFrame([input_data], columns=inputs)
        
        # Predict
        raw_pred = float(model.predict(input_df)[0])
        # Prevent impossible negative valuations
        pred_usd = max(10000.0, raw_pred)
        pred_inr = pred_usd * usd_to_inr_rate

        # Compute feature importance
        top_features = []
        importance_df = None
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": inputs,
                "Importance": imp
            }).sort_values(by="Importance", ascending=False)
            
            top_raw = importance_df.head(5)["Feature"].tolist()
            top_features = [feature_labels.get(f, f).split(" ", 1)[-1] for f in top_raw]

        # Store in session state for persistence
        st.session_state.prediction_result = {
            "usd": pred_usd,
            "inr": pred_inr,
            "top_features": top_features,
            "importance_df": importance_df,
            "input_data": input_data
        }
        # Reset audio cache for new prediction
        st.session_state.audio_cache = {}

# ---------------------------------------------------
# DISPLAY PREDICTION RESULTS (Persistent via Session State)
# ---------------------------------------------------
res = st.session_state.prediction_result

if res is not None:
    st.success("✨ Valuation computed successfully!")
    
    st.subheader("💰 Estimated Property Valuation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "INR" in currency_mode:
            st.metric("Estimated Market Value (INR)", format_inr(res["inr"]))
        else:
            st.metric("Estimated Market Value (USD)", f"${res['usd']:,.0f}")
            
    with col2:
        if "INR" in currency_mode:
            st.metric("Equivalent in USD", f"${res['usd']:,.0f}")
        else:
            st.metric("Equivalent in INR", format_inr(res["inr"]))
            
    with col3:
        st.metric("Raw Model Output", f"{res['usd']:,.2f}")

    # ---------------------------------------------------
    # FEATURE IMPORTANCE & AI EXPLANATION
    # ---------------------------------------------------
    st.divider()
    st.subheader("🧠 Model Interpretation: Why this valuation?")

    if res["importance_df"] is not None:
        top_5_df = res["importance_df"].head(5).copy()
        top_5_df["Display"] = top_5_df["Feature"].map(feature_labels)
        
        c_chart, c_insight = st.columns([1.2, 1.0])
        
        with c_chart:
            st.markdown("**Top 5 Influential Property Attributes:**")
            st.bar_chart(top_5_df.set_index("Display")["Importance"])
            
        with c_insight:
            f1 = res["top_features"][0] if len(res["top_features"]) > 0 else "Overall Quality"
            f2 = res["top_features"][1] if len(res["top_features"]) > 1 else "Living Area"
            st.markdown("#### 🤖 AI Property Insight")
            st.info(
                f"Based on historical sales patterns, **{f1}** had the greatest positive impact on "
                f"this assessment, followed closely by **{f2}**. "
                f"Properties scoring higher in these dimensions consistently achieve premium valuations."
            )
    else:
        st.warning("Feature importance not supported for this model.")

    # ---------------------------------------------------
    # GNANI.AI TEXT-TO-SPEECH (VOICE ASSISTANT) SECTION
    # ---------------------------------------------------
    st.divider()
    st.subheader("🎙️ Gnani.ai Voice Summary (आवाज़ में रिपोर्ट सुनें)")

    st.markdown("""
    Listen to an AI-generated natural voice narration of this property valuation report.
    Powered by **Gnani.ai Indic Voice Technology** with multi-lingual dialect support.
    """)

    vcol1, vcol2 = st.columns([1, 1])
    
    with vcol1:
        voice_lang = st.selectbox(
            "Choose Narration Language (भाषा चुनें)",
            options=list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda code: SUPPORTED_LANGUAGES[code]["name"],
            index=0,
            key="voice_lang_select"
        )
    with vcol2:
        voice_gender = st.selectbox(
            "Voice Gender / Tone",
            options=["Female", "Male"],
            index=0,
            key="voice_gender_select"
        )

    # Generate speech script
    narration_script = generate_valuation_script(
        price_usd=res["usd"],
        price_inr=res["inr"],
        top_features=res["top_features"],
        lang=voice_lang
    )

    with st.expander("📝 View Narration Script (जो बोला जाएगा)"):
        st.write(narration_script)

    play_button = st.button("🔊 Synthesize & Listen to Voice Report", type="secondary", use_container_width=True)

    cache_key = f"{voice_lang}_{voice_gender}_{round(res['usd'], -2)}"

    if play_button:
        with st.spinner("Synthesizing speech with Gnani.ai Voice AI..."):
            audio_bytes, mime_type, engine_used, status_msg = synthesize_speech(
                text=narration_script,
                lang=voice_lang,
                voice=voice_gender.lower(),
                api_key=gnani_api_key,
                token=gnani_token,
                endpoint=gnani_endpoint
            )
            if audio_bytes:
                st.session_state.audio_cache[cache_key] = {
                    "bytes": audio_bytes,
                    "mime": mime_type,
                    "engine": engine_used,
                    "msg": status_msg
                }
            else:
                st.error(f"Failed to generate audio: {status_msg}")

    # Display audio player if available in cache
    if cache_key in st.session_state.audio_cache:
        audio_info = st.session_state.audio_cache[cache_key]
        st.markdown('<div class="voice-box">', unsafe_allow_html=True)
        st.markdown(f"**Voice Report ({SUPPORTED_LANGUAGES[voice_lang]['name']}):**")
        st.audio(audio_info["bytes"], format=audio_info["mime"])
        
        col_down, col_status = st.columns([1, 2])
        with col_down:
            st.download_button(
                "⬇️ Download Audio Report",
                data=audio_info["bytes"],
                file_name=f"property_valuation_{voice_lang}.mp3",
                mime=audio_info["mime"]
            )
        with col_status:
            if audio_info["engine"] == "gnani":
                st.markdown('<span class="badge-gnani">⚡ Synthesized by Gnani.ai Voice AI</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-fallback">🛡️ High-Fidelity Voice Synthesizer</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.divider()
fcol1, fcol2 = st.columns([2, 1])

with fcol1:
    st.caption("⚠️ **Disclaimer**: AI/ML generated estimates are based on historical Ames housing patterns and are intended for valuation guidance. Local market dynamics and property inspections should also be considered.")

with fcol2:
    st.caption("🇮🇳 Developed for **Build in AI for India — Delhi Edition 2026**")
