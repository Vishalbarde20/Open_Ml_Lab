import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import streamlit as st

class MetricsVisualizer:
    """Create visualizations for model metrics"""
    
    @staticmethod
    def plot_metrics_comparison(metrics):
        """Create bar chart of metrics"""
        metrics_df = pd.DataFrame({
            'Metric': ['Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Score': [
                metrics['train_accuracy'],
                metrics['test_accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score']
            ]
        })
        
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Score',
            title='Model Performance Metrics',
            color='Score',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    @staticmethod
    def plot_confusion_matrix(cm, class_names=None):
        """Create confusion matrix heatmap"""
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=class_names,
            y=class_names,
            title="Confusion Matrix",
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def plot_feature_importance(model, feature_names):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            fig = go.Figure([go.Bar(
                x=importances[indices],
                y=[feature_names[i] for i in indices],
                orientation='h'
            )])
            fig.update_layout(
                title='Top 10 Feature Importances',
                xaxis_title='Importance',
                yaxis_title='Features',
                height=400
            )
            return fig
        return None
