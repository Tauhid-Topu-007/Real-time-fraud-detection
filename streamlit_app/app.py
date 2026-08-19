# ============================================
# streamlit_app/app.py
# COMPLETE FIXED VERSION (applymap → map)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import FraudPredictor

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="🛡️ Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .risk-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
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

# ============================================
# HELPER FUNCTION FOR COLORING (FIXED)
# ============================================
def color_risk(val):
    """Color rows based on risk level"""
    if val == 'HIGH':
        return 'background-color: #FFE5E5'
    elif val == 'MEDIUM':
        return 'background-color: #FFF3E0'
    elif val == 'LOW':
        return 'background-color: #E8F5E9'
    return ''

def apply_style_to_df(df):
    """Apply styling to dataframe using map (not applymap)"""
    try:
        # Try using map (pandas 2.0+)
        return df.style.map(color_risk, subset=['Risk Level'])
    except AttributeError:
        # Fallback for older pandas
        return df.style.applymap(color_risk, subset=['Risk Level'])

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if 'predictor' not in st.session_state:
    try:
        st.session_state.predictor = FraudPredictor()
        if st.session_state.predictor.model:
            st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.session_state.predictor = None
        st.error(f"❌ Error loading model: {e}")

if 'predictions' not in st.session_state:
    st.session_state.predictions = []

if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

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
    
    page = st.radio(
        "Select Page",
        ["🏠 Dashboard", "🔍 Predict", "📊 Batch Predict", "ℹ️ Model Info"],
        index=0
    )
    
    st.divider()
    
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
        st.info("💡 Make sure model files are in 'models/' directory")
    
    st.divider()
    
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
# PAGE: DASHBOARD (FIXED)
# ============================================
if page == "🏠 Dashboard":
    st.title("📊 Dashboard")
    
    if not st.session_state.predictions:
        st.info("ℹ️ No predictions yet. Go to 'Predict' page to get started.")
    else:
        df = pd.DataFrame(st.session_state.predictions)
        
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
        
        st.subheader("📋 Recent Predictions")
        display_df = df.tail(10).copy()
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
        display_df = display_df[['timestamp', 'transaction_id', 'fraud_probability', 'risk_level', 'decision']]
        display_df.columns = ['Time', 'Transaction ID', 'Probability', 'Risk Level', 'Decision']
        
        # ============================================
        # FIXED: Use map instead of applymap
        # ============================================
        styled_df = display_df.style.map(color_risk, subset=['Risk Level'])
        st.dataframe(styled_df, use_container_width=True, height=300)

# ============================================
# PAGE: PREDICT
# ============================================
elif page == "🔍 Predict":
    st.title("🔍 Single Transaction Prediction")
    
    if st.session_state.predictor is None or st.session_state.predictor.model is None:
        st.error("❌ Model not loaded. Please check the models directory.")
        st.info("Make sure these files exist in 'models/':")
        st.code("""
models/
├── xgboost_model.pkl
├── feature_engineer.pkl
├── feature_columns.pkl
└── model_info.json
        """)
    else:
        predictor = st.session_state.predictor
        
        with st.expander("📊 Model Info", expanded=False):
            info = predictor.get_model_info()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Features", info.get('features', 0))
            with col2:
                st.metric("Threshold", f"{info.get('threshold', 0.5):.2f}")
            with col3:
                st.metric("Amount Mean", f"{info.get('amount_mean', 0):.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            step = st.number_input("Step (Hour)", min_value=0, value=5)
            amount = st.number_input("Amount ($)", min_value=0.01, value=50000.0, step=1000.0)
            txn_type = st.selectbox(
                "Transaction Type",
                ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
                index=1
            )
            nameOrig = st.text_input("Origin Account", "C1234567890")
            oldbalanceOrg = st.number_input("Origin Balance Before ($)", min_value=0.0, value=5000.0, step=100.0)
        
        with col2:
            newbalanceOrig = st.number_input("Origin Balance After ($)", min_value=0.0, value=0.0, step=100.0)
            nameDest = st.text_input("Destination Account", "M9876543210")
            oldbalanceDest = st.number_input("Destination Balance Before ($)", min_value=0.0, value=1000.0, step=100.0)
            newbalanceDest = st.number_input("Destination Balance After ($)", min_value=0.0, value=51000.0, step=100.0)
            transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
            st.info(f"🆔 {transaction_id}")
        
        st.subheader("📝 Quick Examples")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚨 Fraudulent Transfer", use_container_width=True):
                st.session_state.example = {
                    'step': 1, 'type': 'TRANSFER', 'amount': 999999.0,
                    'nameOrig': 'C1111111111', 'oldbalanceOrg': 1000000.0, 'newbalanceOrig': 1.0,
                    'nameDest': 'M9999999999', 'oldbalanceDest': 0.0, 'newbalanceDest': 999999.0
                }
                st.rerun()
        
        with col2:
            if st.button("⚠️ Suspicious Cash Out", use_container_width=True):
                st.session_state.example = {
                    'step': 2, 'type': 'CASH_OUT', 'amount': 500000.0,
                    'nameOrig': 'C2222222222', 'oldbalanceOrg': 600000.0, 'newbalanceOrig': 100000.0,
                    'nameDest': 'M8888888888', 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0
                }
                st.rerun()
        
        with col3:
            if st.button("✅ Normal Payment", use_container_width=True):
                st.session_state.example = {
                    'step': 10, 'type': 'PAYMENT', 'amount': 50.0,
                    'nameOrig': 'C4444444444', 'oldbalanceOrg': 1000.0, 'newbalanceOrig': 950.0,
                    'nameDest': 'M6666666666', 'oldbalanceDest': 500.0, 'newbalanceDest': 550.0
                }
                st.rerun()
        
        if 'example' in st.session_state:
            example = st.session_state.example
            step = example['step']
            amount = example['amount']
            txn_type = example['type']
            nameOrig = example['nameOrig']
            oldbalanceOrg = example['oldbalanceOrg']
            newbalanceOrig = example['newbalanceOrig']
            nameDest = example['nameDest']
            oldbalanceDest = example['oldbalanceDest']
            newbalanceDest = example['newbalanceDest']
        
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            with st.spinner("Analyzing transaction..."):
                transaction = {
                    'step': int(step), 'type': txn_type, 'amount': float(amount),
                    'nameOrig': str(nameOrig), 'oldbalanceOrg': float(oldbalanceOrg),
                    'newbalanceOrig': float(newbalanceOrig), 'nameDest': str(nameDest),
                    'oldbalanceDest': float(oldbalanceDest), 'newbalanceDest': float(newbalanceDest)
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
                    prob = result['fraud_probability']
                    
                    if prob > 0.7:
                        st.error(f"🚨 HIGH RISK - Probability: {prob*100:.2f}%")
                    elif prob > 0.3:
                        st.warning(f"⚠️ MEDIUM RISK - Probability: {prob*100:.2f}%")
                    else:
                        st.success(f"✅ LOW RISK - Probability: {prob*100:.2f}%")
                    
                    if risk_level == "HIGH":
                        st.markdown(f"""
                        <div class="risk-box risk-high">
                            <h2 style="color:#FF0000;">🚨 HIGH RISK</h2>
                            <h3>Decision: BLOCK</h3>
                            <p>Probability: {prob*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif risk_level == "MEDIUM":
                        st.markdown(f"""
                        <div class="risk-box risk-medium">
                            <h2 style="color:#FF9800;">⚡ MEDIUM RISK</h2>
                            <h3>Decision: REVIEW</h3>
                            <p>Probability: {prob*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-box risk-low">
                            <h2 style="color:#4CAF50;">✅ LOW RISK</h2>
                            <h3>Decision: APPROVE</h3>
                            <p>Probability: {prob*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fraud Probability", f"{prob*100:.2f}%")
                    with col2:
                        st.metric("Risk Level", risk_level)
                    with col3:
                        st.metric("Decision", result['decision'])
                    
                    if 'method' in result and result['method'] == 'rule_based':
                        st.info("⚡ Rule-based system detected fraud patterns")
                        if 'reasons' in result and result['reasons']:
                            st.write("**📋 Fraud Indicators Found:**")
                            for reason in result['reasons']:
                                st.write(f"   - 🔴 {reason}")
                        if 'score' in result:
                            st.progress(result['score'] / 100)
                            st.write(f"**Risk Score:** {result['score']}/100")
                    
                    with st.expander("🔍 Debug Info", expanded=False):
                        st.write("**Transaction Data:**")
                        st.json(transaction)
                        st.write(f"**Threshold:** {predictor.threshold}")
                        st.write(f"**Probability:** {prob*100:.4f}%")
                        st.write(f"**Risk Level:** {risk_level}")
                        st.write(f"**Decision:** {result['decision']}")
                else:
                    st.error("❌ Prediction failed. Please check the logs.")

# ============================================
# PAGE: BATCH PREDICT (FIXED)
# ============================================
elif page == "📊 Batch Predict":
    st.title("📊 Batch Prediction")
    
    st.markdown("""
    Upload a CSV file with multiple transactions.
    
    **Required columns:** `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, 
    `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`
    """)
    
    sample_data = """step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest
1,TRANSFER,999999.0,C1111111111,1000000.0,1.0,M9999999999,0.0,999999.0
5,TRANSFER,50000.0,C1234567890,5000.0,0.0,M9876543210,1000.0,51000.0
2,CASH_OUT,500000.0,C2222222222,600000.0,100000.0,M8888888888,0.0,0.0
10,PAYMENT,50.0,C4444444444,1000.0,950.0,M6666666666,500.0,550.0"""
    
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_data,
        file_name="sample_batch.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} transactions")
            
            # Column mapping
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ['step', 'steps', 'time_step']:
                    column_mapping[col] = 'step'
                elif col_lower in ['type', 'txn_type', 'transaction_type']:
                    column_mapping[col] = 'type'
                elif col_lower in ['amount', 'amt', 'transaction_amount']:
                    column_mapping[col] = 'amount'
                elif col_lower in ['nameorig', 'origin', 'origin_account']:
                    column_mapping[col] = 'nameOrig'
                elif col_lower in ['oldbalanceorg', 'origin_balance']:
                    column_mapping[col] = 'oldbalanceOrg'
                elif col_lower in ['newbalanceorig', 'origin_balance_after']:
                    column_mapping[col] = 'newbalanceOrig'
                elif col_lower in ['namedest', 'destination', 'dest_account']:
                    column_mapping[col] = 'nameDest'
                elif col_lower in ['oldbalancedest', 'dest_balance']:
                    column_mapping[col] = 'oldbalanceDest'
                elif col_lower in ['newbalancedest', 'dest_balance_after']:
                    column_mapping[col] = 'newbalanceDest'
            
            df = df.rename(columns=column_mapping)
            
            required_cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                           'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
                st.stop()
            
            with st.expander("📊 Column Mapping"):
                for orig, new in column_mapping.items():
                    st.write(f"   {orig} → {new}")
            
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                if st.session_state.predictor is None or st.session_state.predictor.model is None:
                    st.error("❌ Model not loaded")
                else:
                    with st.spinner(f"Processing {len(df)} transactions..."):
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, row in df.iterrows():
                            status_text.text(f"Processing transaction {idx+1}/{len(df)}")
                            
                            transaction = row.to_dict()
                            transaction['transaction_id'] = f"BATCH_{idx}"
                            
                            result = st.session_state.predictor.predict(transaction)
                            
                            if result:
                                result['transaction_id'] = transaction['transaction_id']
                                result['timestamp'] = datetime.now().isoformat()
                                results.append(result)
                                st.session_state.predictions.append(result)
                                st.session_state.total_predictions += 1
                            else:
                                rule_result = st.session_state.predictor.rule_based_detect(transaction)
                                rule_result['transaction_id'] = transaction['transaction_id']
                                rule_result['timestamp'] = datetime.now().isoformat()
                                results.append(rule_result)
                                st.session_state.predictions.append(rule_result)
                                st.session_state.total_predictions += 1
                            
                            progress_bar.progress((idx + 1) / len(df))
                        
                        status_text.text("✅ Batch prediction complete!")
                        
                        if results:
                            st.divider()
                            st.subheader("📊 Results")
                            results_df = pd.DataFrame(results)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total", len(results_df))
                            with col2:
                                high_risk = len(results_df[results_df['risk_level'] == 'HIGH'])
                                st.metric("High Risk", high_risk, delta=f"{high_risk/len(results_df)*100:.1f}%")
                            with col3:
                                medium_risk = len(results_df[results_df['risk_level'] == 'MEDIUM'])
                                st.metric("Medium Risk", medium_risk)
                            with col4:
                                low_risk = len(results_df[results_df['risk_level'] == 'LOW'])
                                st.metric("Low Risk", low_risk)
                            
                            if 'method' in results_df.columns:
                                method_counts = results_df['method'].value_counts()
                                st.write("**Method Used:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"ML: {method_counts.get('ml', 0)}")
                                with col2:
                                    st.write(f"Rule-Based: {method_counts.get('rule_based', 0)}")
                            
                            display_cols = ['transaction_id', 'fraud_probability', 'risk_level', 'decision']
                            if 'method' in results_df.columns:
                                display_cols.append('method')
                            if 'score' in results_df.columns:
                                display_cols.append('score')
                            
                            display_df = results_df[display_cols].copy()
                            display_df['fraud_probability'] = display_df['fraud_probability'] * 100
                            display_df.columns = [c.replace('fraud_probability', 'Probability %').replace('_', ' ').title() for c in display_df.columns]
                            
                            # ============================================
                            # FIXED: Use map instead of applymap
                            # ============================================
                            styled_df = display_df.style.map(color_risk, subset=['Risk Level'])
                            st.dataframe(styled_df, use_container_width=True, height=400)
                            
                            st.subheader("📊 Summary Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Avg Probability", f"{results_df['fraud_probability'].mean()*100:.2f}%")
                            with col2:
                                st.metric("Max Probability", f"{results_df['fraud_probability'].max()*100:.2f}%")
                            with col3:
                                st.metric("Min Probability", f"{results_df['fraud_probability'].min()*100:.2f}%")
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv,
                                file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ============================================
# PAGE: MODEL INFO
# ============================================
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
                'Model Type': [info.get('model_type', 'xgboost + rule-based')],
                'Features': [info.get('features', 0)],
                'Threshold': [info.get('threshold', 0.5)],
                'Amount Mean': [f"{info.get('amount_mean', 0):.2f}"],
                'Amount Std': [f"{info.get('amount_std', 0):.2f}"]
            }
            st.dataframe(pd.DataFrame(info_data), use_container_width=True)
            
            metrics = info.get('metrics', {})
            if metrics:
                st.subheader("📈 Performance Metrics")
                st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
        
        with col2:
            st.subheader("📋 Feature Information")
            st.write(f"**Total Features:** {info.get('features', 0)}")
            
            if hasattr(predictor, 'feature_columns') and predictor.feature_columns:
                st.write("**First 10 Features:**")
                for i, feat in enumerate(predictor.feature_columns[:10], 1):
                    st.write(f"{i}. {feat}")
                if len(predictor.feature_columns) > 10:
                    st.write(f"... and {len(predictor.feature_columns) - 10} more")
            
            st.subheader("📁 Model Files")
            st.code("""
models/
├── xgboost_model.pkl
├── feature_engineer.pkl
├── feature_columns.pkl
└── model_info.json
            """)
            
            st.subheader("📊 How Risk is Determined")
            st.info("""
            **Hybrid System (ML + Rule-Based):**
            
            1. **ML Model:** XGBoost predicts probability
            2. **Rule-Based:** 8 rules check for fraud patterns
            3. **Override:** If rules detect HIGH risk, ML is overridden
            """)