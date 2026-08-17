import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_app.utils.visualizations import Visualizer

def show():
    st.title("📊 Batch Prediction")
    
    st.markdown("""
    Upload a CSV file with multiple transactions for batch prediction.
    
    Required columns:
    - `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`
    - `nameDest`, `oldbalanceDest`, `newbalanceDest`
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} transactions")
            
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
            
            required_cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                           'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
                return
            
            if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(df)} transactions..."):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, row in df.iterrows():
                        status_text.text(f"Processing transaction {idx+1}/{len(df)}")
                        
                        transaction = row.to_dict()
                        transaction['transaction_id'] = f"BATCH_{idx}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        
                        result = st.session_state.model_loader.predict(transaction)
                        
                        if result:
                            result['transaction_id'] = transaction['transaction_id']
                            result['timestamp'] = datetime.now().isoformat()
                            results.append(result)
                            st.session_state.predictions.append(result)
                            st.session_state.total_predictions += 1
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    status_text.text("✅ Batch prediction complete!")
                    
                    st.divider()
                    st.subheader("📊 Batch Prediction Results")
                    
                    results_df = pd.DataFrame(results)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Transactions", len(results_df))
                    with col2:
                        high_risk = len(results_df[results_df['risk_level'] == 'HIGH'])
                        st.metric("High Risk", high_risk, delta=f"{high_risk/len(results_df)*100:.1f}%")
                    with col3:
                        medium_risk = len(results_df[results_df['risk_level'] == 'MEDIUM'])
                        st.metric("Medium Risk", medium_risk, delta=f"{medium_risk/len(results_df)*100:.1f}%")
                    with col4:
                        low_risk = len(results_df[results_df['risk_level'] == 'LOW'])
                        st.metric("Low Risk", low_risk, delta=f"{low_risk/len(results_df)*100:.1f}%")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_risk = Visualizer.create_risk_distribution(results)
                        st.plotly_chart(fig_risk, use_container_width=True)
                    with col2:
                        fig_decision = Visualizer.create_decision_distribution(results)
                        st.plotly_chart(fig_decision, use_container_width=True)
                    
                    fig_prob = Visualizer.create_probability_distribution(results)
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    display_df = results_df.copy()
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
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    
    else:
        st.info("📤 Upload a CSV file to get started")
        
        with st.expander("📋 Sample Data Format"):
            sample_data = {
                'step': [5, 10],
                'type': ['TRANSFER', 'PAYMENT'],
                'amount': [2450.0, 100.0],
                'nameOrig': ['C1234567890', 'C0987654321'],
                'oldbalanceOrg': [5000.0, 1000.0],
                'newbalanceOrig': [2550.0, 900.0],
                'nameDest': ['M9876543210', 'M1234567890'],
                'oldbalanceDest': [1000.0, 500.0],
                'newbalanceDest': [3450.0, 600.0]
            }
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)
            
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv,
                file_name="sample_transactions.csv",
                mime="text/csv"
            )