import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def show():
    st.title("ℹ️ Model Information")
    
    model_info = st.session_state.model_loader.get_model_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Details")
        
        if st.session_state.model_loader.model:
            st.success("✅ Model is loaded and ready")
            
            info_data = {
                'Model Type': [model_info.get('model_type', 'N/A')],
                'Features Used': [model_info.get('features_used', 0)],
                'N_Estimators': [model_info.get('n_estimators', 'N/A')],
                'Max Depth': [model_info.get('max_depth', 'N/A')],
                'Model Version': ['v1.0.0']
            }
            
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df, use_container_width=True)
        else:
            st.error("❌ No model loaded")
    
    with col2:
        st.subheader("⚙️ Configuration")
        
        config = model_info.get('config', {})
        if config:
            thresholds = config.get('thresholds', {})
            st.metric("Approve Threshold", thresholds.get('approve', 0.30))
            st.metric("Block Threshold", thresholds.get('optimal', 0.75))
            
            model_config = config.get('model', {})
            st.write("**Model Parameters:**")
            st.json(model_config.get('parameters', {}))
    
    st.divider()
    
    st.subheader("📋 Features")
    
    features = st.session_state.model_loader.feature_columns
    
    if features:
        st.write(f"Total Features: {len(features)}")
        
        feature_groups = {
            'Transaction Type': ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 
                                'type_PAYMENT', 'type_TRANSFER', 'type_risk_score'],
            'Balance': ['balance_change_orig', 'balance_change_dest', 
                       'balance_error_orig', 'balance_error_dest'],
            'Amount': ['amount_log', 'amount_z_score', 'amount_percentile', 
                      'is_high_amount', 'is_rounded_amount'],
            'Customer': ['orig_txn_count', 'orig_avg_amount', 'orig_fraud_ratio',
                        'dest_txn_count', 'dest_avg_amount'],
            'Time': ['hour', 'day', 'is_weekend', 'is_night', 'hour_frequency'],
            'Velocity': ['transactions_last_1h', 'transactions_last_5h', 
                        'transactions_last_24h', 'time_since_last_txn']
        }
        
        for group_name, group_features in feature_groups.items():
            available = [f for f in group_features if f in features]
            if available:
                with st.expander(f"📁 {group_name} ({len(available)} features)"):
                    st.write(", ".join(available))
    else:
        st.warning("No feature information available")
    
    st.divider()
    st.subheader("🖥️ System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions", st.session_state.total_predictions)
    
    with col2:
        if st.session_state.predictions:
            high_risk = sum(1 for p in st.session_state.predictions if p['risk_level'] == 'HIGH')
            st.metric("High Risk Alerts", high_risk)
        else:
            st.metric("High Risk Alerts", 0)
    
    with col3:
        if st.session_state.predictions:
            avg_prob = sum(p['fraud_probability'] for p in st.session_state.predictions) / len(st.session_state.predictions)
            st.metric("Avg Probability", f"{avg_prob:.2%}")
        else:
            st.metric("Avg Probability", "N/A")