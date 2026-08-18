import streamlit as st
import pandas as pd

def show():
    """Model info page"""
    st.title("ℹ️ Model Information")
    
    # Check if predictor exists
    if not hasattr(st.session_state, 'predictor') or st.session_state.predictor is None:
        st.error("❌ Model not loaded. Please check the model files.")
        return
    
    predictor = st.session_state.predictor
    
    try:
        info = predictor.get_model_info()
    except:
        info = {
            'model_type': 'xgboost',
            'features': 19,
            'threshold': 0.55,
            'metrics': {}
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Details")
        
        info_data = {
            'Model Type': [info.get('model_type', 'xgboost')],
            'Features Used': [info.get('features', 0)],
            'Optimal Threshold': [info.get('threshold', 0.5)]
        }
        
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True)
        
        # Show metrics
        metrics = info.get('metrics', {})
        if metrics:
            st.subheader("📈 Performance Metrics")
            metrics_df = pd.DataFrame([metrics])
            st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        st.subheader("⚙️ Feature Information")
        
        # Show feature info
        if hasattr(predictor, 'feature_columns'):
            features = predictor.feature_columns
            st.write(f"**Total Features:** {len(features)}")
            
            # Show first 10 features
            st.write("**First 10 Features:**")
            for i, feat in enumerate(features[:10], 1):
                st.write(f"{i}. {feat}")
            
            if len(features) > 10:
                st.write(f"... and {len(features) - 10} more")
    
    st.divider()
    
    # System info
    st.subheader("🖥️ System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions", st.session_state.total_predictions)
    
    with col2:
        if st.session_state.predictions:
            high_risk = sum(1 for p in st.session_state.predictions if p.get('risk_level') == 'HIGH')
            st.metric("High Risk Alerts", high_risk)
        else:
            st.metric("High Risk Alerts", 0)
    
    with col3:
        if st.session_state.predictions:
            avg_prob = sum(p.get('fraud_probability', 0) for p in st.session_state.predictions) / len(st.session_state.predictions)
            st.metric("Avg Probability", f"{avg_prob:.2%}")
        else:
            st.metric("Avg Probability", "N/A")