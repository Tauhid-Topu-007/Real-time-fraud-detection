import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import FraudPredictor

# ============================================
# PAGE CONFIGURATION - Must be first
# ============================================
st.set_page_config(
    page_title="🛡️ Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if 'predictor' not in st.session_state:
    try:
        st.session_state.predictor = FraudPredictor()
    except Exception as e:
        st.session_state.predictor = None

if 'predictions' not in st.session_state:
    st.session_state.predictions = []

if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #FF3333;
        color: white;
    }
    div[data-testid="stSidebarNav"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# MAIN HEADER
# ============================================
st.markdown('<h1 class="main-header">🛡️ Real-Time Fraud Detection System</h1>', unsafe_allow_html=True)

# ============================================
# SIDEBAR NAVIGATION
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked--v1.png", width=80)
    st.title("Navigation")
    
    # Custom navigation buttons
    page = st.radio(
        "Select Page",
        ["🏠 Dashboard", "🔍 Predict", "📊 Batch Predict", "ℹ️ Model Info"],
        index=0
    )
    
    st.divider()
    
    # System Status
    st.subheader("📊 System Status")
    if st.session_state.predictor and st.session_state.predictor.model:
        st.success("✅ Model Loaded")
        try:
            info = st.session_state.predictor.get_model_info()
            st.info(f"📊 Features: {info.get('features', 0)}")
            st.info(f"🎯 Threshold: {info.get('threshold', 0.5)}")
        except:
            pass
    else:
        st.error("❌ Model Not Loaded")
        st.info("💡 Run: python scripts/train_for_streamlit.py")
    
    st.divider()
    
    # Statistics
    st.subheader("📈 Statistics")
    st.metric("Total Predictions", st.session_state.total_predictions)
    
    if st.session_state.predictions:
        high_risk = sum(1 for p in st.session_state.predictions if p.get('risk_level') == 'HIGH')
        st.metric("High Risk Alerts", high_risk)
        if len(st.session_state.predictions) > 0:
            alert_rate = high_risk / len(st.session_state.predictions) * 100
            st.metric("Alert Rate", f"{alert_rate:.1f}%")
    
    st.divider()
    
    if st.button("🗑️ Clear History"):
        st.session_state.predictions = []
        st.session_state.total_predictions = 0
        st.rerun()

# ============================================
# PAGE CONTENT
# ============================================

# DASHBOARD PAGE
if page == "🏠 Dashboard":
    st.title("📊 Dashboard")
    
    if not st.session_state.predictions:
        st.info("ℹ️ No predictions yet. Go to 'Predict' page to get started.")
    else:
        df = pd.DataFrame(st.session_state.predictions)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            high_risk = len(df[df['risk_level'] == 'HIGH'])
            st.metric("High Risk", high_risk, delta=f"{high_risk/len(df)*100:.1f}%")
        with col3:
            medium_risk = len(df[df['risk_level'] == 'MEDIUM'])
            st.metric("Medium Risk", medium_risk)
        with col4:
            low_risk = len(df[df['risk_level'] == 'LOW'])
            st.metric("Low Risk", low_risk)
        
        st.divider()
        
        # Recent predictions
        st.subheader("📋 Recent Predictions")
        display_df = df.tail(10).copy()
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
        display_df = display_df[['timestamp', 'transaction_id', 'fraud_probability', 'risk_level', 'decision']]
        display_df.columns = ['Time', 'Transaction ID', 'Probability', 'Risk Level', 'Decision']
        
        def color_risk(val):
            if val == 'HIGH':
                return 'background-color: #FFE5E5'
            elif val == 'MEDIUM':
                return 'background-color: #FFF3E0'
            elif val == 'LOW':
                return 'background-color: #E8F5E9'
            return ''
        
        st.dataframe(
            display_df.style.applymap(color_risk, subset=['Risk Level']),
            use_container_width=True,
            height=300
        )

# PREDICT PAGE
elif page == "🔍 Predict":
    st.title("🔍 Single Transaction Prediction")
    
    if st.session_state.predictor is None or st.session_state.predictor.model is None:
        st.error("❌ Model not loaded. Please train the model first.")
        st.info("Run: python scripts/train_for_streamlit.py")
    else:
        predictor = st.session_state.predictor
        
        col1, col2 = st.columns(2)
        
        with col1:
            step = st.number_input("Step (Hour)", min_value=0, value=5)
            amount = st.number_input("Amount ($)", min_value=0.01, value=2450.0, step=100.0)
            type = st.selectbox(
                "Transaction Type",
                ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
                index=1
            )
            nameOrig = st.text_input("Origin Account", "C1234567890")
            oldbalanceOrg = st.number_input("Origin Balance Before ($)", min_value=0.0, value=5000.0, step=100.0)
        
        with col2:
            newbalanceOrig = st.number_input("Origin Balance After ($)", min_value=0.0, value=2550.0, step=100.0)
            nameDest = st.text_input("Destination Account", "M9876543210")
            oldbalanceDest = st.number_input("Destination Balance Before ($)", min_value=0.0, value=1000.0, step=100.0)
            newbalanceDest = st.number_input("Destination Balance After ($)", min_value=0.0, value=3450.0, step=100.0)
            transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
            st.info(f"🆔 {transaction_id}")
        
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            with st.spinner("Analyzing transaction..."):
                transaction = {
                    'step': step, 'type': type, 'amount': amount,
                    'nameOrig': nameOrig, 'oldbalanceOrg': oldbalanceOrg,
                    'newbalanceOrig': newbalanceOrig, 'nameDest': nameDest,
                    'oldbalanceDest': oldbalanceDest, 'newbalanceDest': newbalanceDest,
                    'transaction_id': transaction_id
                }
                
                result = predictor.predict(transaction)
                
                if result:
                    result['timestamp'] = datetime.now().isoformat()
                    result['transaction_id'] = transaction_id
                    
                    st.session_state.predictions.append(result)
                    st.session_state.total_predictions += 1
                    
                    st.divider()
                    st.subheader("📊 Prediction Results")
                    
                    risk_level = result['risk_level']
                    colors = {
                        'HIGH': ('#FFE5E5', '#FF0000', '⚠️ HIGH RISK', 'BLOCK'),
                        'MEDIUM': ('#FFF3E0', '#FF9800', '⚡ MEDIUM RISK', 'REVIEW'),
                        'LOW': ('#E8F5E9', '#4CAF50', '✅ LOW RISK', 'APPROVE')
                    }
                    
                    bg, border, title, decision = colors[risk_level]
                    
                    st.markdown(f"""
                    <div style="padding:1.5rem;border-radius:10px;background-color:{bg};border:2px solid {border};text-align:center;">
                        <h2 style="color:{border};">{title}</h2>
                        <h3>Decision: {decision}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fraud Probability", f"{result['fraud_probability']*100:.1f}%")
                    with col2:
                        st.metric("Risk Level", risk_level)
                    with col3:
                        st.metric("Decision", result['decision'])

# BATCH PREDICT PAGE
elif page == "📊 Batch Predict":
    st.title("📊 Batch Prediction")
    
    st.markdown("""
    Upload a CSV file with multiple transactions.
    
    **Required columns:** `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, 
    `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`
    """)
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} transactions")
            
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                if st.session_state.predictor is None or st.session_state.predictor.model is None:
                    st.error("❌ Model not loaded")
                else:
                    with st.spinner(f"Processing {len(df)} transactions..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            transaction = row.to_dict()
                            transaction['transaction_id'] = f"BATCH_{idx}"
                            result = st.session_state.predictor.predict(transaction)
                            if result:
                                result['transaction_id'] = transaction['transaction_id']
                                result['timestamp'] = datetime.now().isoformat()
                                results.append(result)
                                st.session_state.predictions.append(result)
                                st.session_state.total_predictions += 1
                            progress_bar.progress((idx + 1) / len(df))
                        
                        if results:
                            st.divider()
                            st.subheader("📊 Results")
                            results_df = pd.DataFrame(results)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total", len(results_df))
                            with col2:
                                high_risk = len(results_df[results_df['risk_level'] == 'HIGH'])
                                st.metric("High Risk", high_risk)
                            with col3:
                                medium_risk = len(results_df[results_df['risk_level'] == 'MEDIUM'])
                                st.metric("Medium Risk", medium_risk)
                            with col4:
                                low_risk = len(results_df[results_df['risk_level'] == 'LOW'])
                                st.metric("Low Risk", low_risk)
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
        except Exception as e:
            st.error(f"Error: {e}")

# MODEL INFO PAGE
elif page == "ℹ️ Model Info":
    st.title("ℹ️ Model Information")
    
    if st.session_state.predictor is None:
        st.error("❌ Model not loaded")
    else:
        predictor = st.session_state.predictor
        info = predictor.get_model_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Details")
            info_data = {
                'Model Type': [info.get('model_type', 'xgboost')],
                'Features': [info.get('features', 0)],
                'Threshold': [info.get('threshold', 0.5)]
            }
            st.dataframe(pd.DataFrame(info_data), use_container_width=True)
            
            metrics = info.get('metrics', {})
            if metrics:
                st.subheader("📈 Performance")
                st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
        
        with col2:
            st.subheader("📋 Features")
            if hasattr(predictor, 'feature_columns') and predictor.feature_columns:
                st.write(f"Total: {len(predictor.feature_columns)} features")
                st.write("First 10:")
                for i, feat in enumerate(predictor.feature_columns[:10], 1):
                    st.write(f"{i}. {feat}")
            else:
                st.info("No feature information available")