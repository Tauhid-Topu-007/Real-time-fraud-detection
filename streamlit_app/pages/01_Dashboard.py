import streamlit as st
import pandas as pd
from streamlit_app.utils.visualizations import Visualizer

def show():
    st.title("📊 Dashboard")
    
    if not st.session_state.predictions:
        st.info("ℹ️ No predictions yet. Go to 'Single Prediction' or 'Batch Prediction' page.")
        return
    
    df = pd.DataFrame(st.session_state.predictions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", len(df))
    with col2:
        fraud_count = len(df[df['risk_level'] == 'HIGH'])
        st.metric("High Risk", fraud_count, delta=f"{fraud_count/len(df)*100:.1f}%")
    with col3:
        review_count = len(df[df['risk_level'] == 'MEDIUM'])
        st.metric("Medium Risk", review_count, delta=f"{review_count/len(df)*100:.1f}%")
    with col4:
        approve_count = len(df[df['risk_level'] == 'LOW'])
        st.metric("Low Risk", approve_count, delta=f"{approve_count/len(df)*100:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_risk = Visualizer.create_risk_distribution(st.session_state.predictions)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        fig_decision = Visualizer.create_decision_distribution(st.session_state.predictions)
        st.plotly_chart(fig_decision, use_container_width=True)
    
    fig_prob = Visualizer.create_probability_distribution(st.session_state.predictions)
    st.plotly_chart(fig_prob, use_container_width=True)
    
    st.subheader("📋 Recent Predictions")
    
    display_df = df.tail(20).copy()
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
        height=400
    )