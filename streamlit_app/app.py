import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from streamlit_app.utils.model_loader import ModelLoader
from streamlit_app.utils.visualizations import Visualizer

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .risk-high {
        background-color: #FFE5E5;
        border: 2px solid #FF0000;
    }
    .risk-medium {
        background-color: #FFF3E0;
        border: 2px solid #FF9800;
    }
    .risk-low {
        background-color: #E8F5E9;
        border: 2px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

if 'model_loader' not in st.session_state:
    st.session_state.model_loader = ModelLoader()

if 'predictions' not in st.session_state:
    st.session_state.predictions = []

if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

st.markdown('<h1 class="main-header">🛡️ Real-Time Fraud Detection System</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked--v1.png", width=80)
    st.title("Navigation")
    
    page = st.radio(
        "Go to",
        ["🏠 Dashboard", "🔍 Single Prediction", "📊 Batch Prediction", "ℹ️ Model Info"]
    )
    
    st.divider()
    
    st.subheader("📊 System Status")
    model_info = st.session_state.model_loader.get_model_info()
    
    if st.session_state.model_loader.model:
        st.success("✅ Model Loaded")
        st.info(f"Model: {model_info['model_type']}")
        st.info(f"Features: {model_info['features_used']}")
    else:
        st.error("❌ Model Not Loaded")
    
    st.divider()
    
    st.subheader("📈 Statistics")
    st.metric("Total Predictions", st.session_state.total_predictions)
    
    if st.session_state.predictions:
        high_risk = sum(1 for p in st.session_state.predictions if p['risk_level'] == 'HIGH')
        st.metric("High Risk Alerts", high_risk)
        st.metric("Alert Rate", f"{high_risk/len(st.session_state.predictions)*100:.1f}%")
    
    st.divider()
    
    if st.button("🗑️ Clear History"):
        st.session_state.predictions = []
        st.session_state.total_predictions = 0
        st.rerun()

if page == "🏠 Dashboard":
    from streamlit_app.pages import 01_Dashboard as dashboard
    dashboard.show()
elif page == "🔍 Single Prediction":
    from streamlit_app.pages import 02_Predict as predict
    predict.show()
elif page == "📊 Batch Prediction":
    from streamlit_app.pages import 03_Batch_Predict as batch_predict
    batch_predict.show()
elif page == "ℹ️ Model Info":
    from streamlit_app.pages import 04_Model_Info as model_info
    model_info.show()