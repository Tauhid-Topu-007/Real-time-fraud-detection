import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class Visualizer:
    """Visualization utilities for Streamlit app"""
    
    @staticmethod
    def create_gauge_chart(probability, title="Fraud Probability"):
        """Create a gauge chart for fraud probability"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            title={'text': title},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if probability > 0.7 else "orange" if probability > 0.3 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': probability * 100
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        return fig
    
    @staticmethod
    def create_feature_importance_chart(features_importance):
        """Create feature importance chart"""
        if not features_importance:
            return None
        
        df = pd.DataFrame(features_importance)
        df = df.sort_values('importance', ascending=True)
        
        fig = px.bar(
            df,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 5 Feature Importances',
            color='importance',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_decision_distribution(predictions):
        """Create decision distribution chart"""
        df = pd.DataFrame(predictions)
        decision_counts = df['decision'].value_counts()
        
        colors = {'APPROVE': 'green', 'REVIEW': 'orange', 'BLOCK': 'red'}
        
        fig = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            title='Decision Distribution',
            color=decision_counts.index,
            color_discrete_map=colors
        )
        
        fig.update_layout(height=350)
        return fig
    
    @staticmethod
    def create_risk_distribution(predictions):
        """Create risk distribution chart"""
        df = pd.DataFrame(predictions)
        risk_counts = df['risk_level'].value_counts()
        
        colors = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
        
        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title='Risk Level Distribution',
            color=risk_counts.index,
            color_discrete_map=colors
        )
        
        fig.update_layout(
            height=300,
            xaxis_title="Risk Level",
            yaxis_title="Count"
        )
        
        return fig
    
    @staticmethod
    def create_probability_distribution(predictions):
        """Create probability distribution chart"""
        df = pd.DataFrame(predictions)
        
        fig = px.histogram(
            df,
            x='fraud_probability',
            nbins=30,
            title='Fraud Probability Distribution',
            color_discrete_sequence=['blue']
        )
        
        fig.update_layout(
            height=300,
            xaxis_title="Fraud Probability",
            yaxis_title="Count"
        )
        
        fig.add_vline(x=0.3, line_dash="dash", line_color="green", 
                     annotation_text="Approve")
        fig.add_vline(x=0.7, line_dash="dash", line_color="red", 
                     annotation_text="Block")
        
        return fig
    
    @staticmethod
    def create_transaction_summary(transaction):
        """Create transaction summary visualization"""
        summary_data = {
            'Field': ['Step', 'Type', 'Amount', 'Origin', 'Destination'],
            'Value': [
                transaction.get('step', 'N/A'),
                transaction.get('type', 'N/A'),
                f"${transaction.get('amount', 0):,.2f}",
                transaction.get('nameOrig', 'N/A'),
                transaction.get('nameDest', 'N/A')
            ]
        }
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Field', 'Value'],
                fill_color='paleturquoise',
                align='left',
                font=dict(size=14)
            ),
            cells=dict(
                values=[summary_data['Field'], summary_data['Value']],
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
        
        return fig