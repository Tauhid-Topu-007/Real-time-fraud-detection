import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_app.utils.visualizations import Visualizer

def show():
    st.title("🔍 Single Transaction Prediction")
    
    st.markdown("Enter transaction details below to check fraud probability.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        step = st.number_input("Step (Hour)", min_value=0, max_value=1000, value=5)
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=2450.0, step=100.0)
        type = st.selectbox(
            "Transaction Type",
            options=["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
            index=1
        )
        nameOrig = st.text_input("Origin Account", value="C1234567890")
        oldbalanceOrg = st.number_input("Origin Balance Before ($)", min_value=0.0, value=5000.0, step=100.0)
        newbalanceOrig = st.number_input("Origin Balance After ($)", min_value=0.0, value=2550.0, step=100.0)
    
    with col2:
        nameDest = st.text_input("Destination Account", value="M9876543210")
        oldbalanceDest = st.number_input("Destination Balance Before ($)", min_value=0.0, value=1000.0, step=100.0)
        newbalanceDest = st.number_input("Destination Balance After ($)", min_value=0.0, value=3450.0, step=100.0)
        isFlaggedFraud = st.selectbox("System Flag", options=[0, 1], index=0)
        transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
        st.info(f"Transaction ID: {transaction_id}")
    
    if st.button("🔮 Predict Fraud", type="primary", use_container_width=True):
        with st.spinner("Analyzing transaction..."):
            transaction = {
                'step': step,
                'type': type,
                'amount': amount,
                'nameOrig': nameOrig,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrig': newbalanceOrig,
                'nameDest': nameDest,
                'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest,
                'isFlaggedFraud': isFlaggedFraud,
                'transaction_id': transaction_id
            }
            
            result = st.session_state.model_loader.predict(transaction)
            
            if result:
                result['timestamp'] = datetime.now().isoformat()
                result['transaction_id'] = transaction_id
                st.session_state.predictions.append(result)
                st.session_state.total_predictions += 1
                
                st.divider()
                st.subheader("📊 Prediction Results")
                
                risk_level = result['risk_level']
                
                if risk_level == "HIGH":
                    st.markdown(f"""
                    <div class="prediction-box risk-high">
                        <h2 style="color: #FF0000;">⚠️ HIGH RISK</h2>
                        <h3>Decision: BLOCK</h3>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == "MEDIUM":
                    st.markdown(f"""
                    <div class="prediction-box risk-medium">
                        <h2 style="color: #FF9800;">⚡ MEDIUM RISK</h2>
                        <h3>Decision: REVIEW</h3>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box risk-low">
                        <h2 style="color: #4CAF50;">✅ LOW RISK</h2>
                        <h3>Decision: APPROVE</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fraud Probability", f"{result['fraud_probability']*100:.1f}%")
                with col2:
                    st.metric("Risk Level", result['risk_level'])
                with col3:
                    st.metric("Decision", result['decision'])
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig_gauge = Visualizer.create_gauge_chart(result['fraud_probability'])
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    if result.get('top_features'):
                        fig_importance = Visualizer.create_feature_importance_chart(result['top_features'])
                        if fig_importance:
                            st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.error("❌ Error making prediction. Please check if the model is loaded.")
    
    with st.expander("📝 Quick Example Transactions"):
        examples = [
            {
                "name": "🚨 Fraudulent Transfer",
                "data": {
                    "step": 5, "type": "TRANSFER", "amount": 2450.0,
                    "nameOrig": "C1234567890", "oldbalanceOrg": 5000.0, "newbalanceOrig": 2550.0,
                    "nameDest": "M9876543210", "oldbalanceDest": 1000.0, "newbalanceDest": 3450.0
                }
            },
            {
                "name": "✅ Legitimate Payment",
                "data": {
                    "step": 10, "type": "PAYMENT", "amount": 100.0,
                    "nameOrig": "C0987654321", "oldbalanceOrg": 1000.0, "newbalanceOrig": 900.0,
                    "nameDest": "M1234567890", "oldbalanceDest": 500.0, "newbalanceDest": 600.0
                }
            }
        ]
        
        for example in examples:
            if st.button(example["name"], key=f"example_{example['name']}"):
                st.json(example["data"])