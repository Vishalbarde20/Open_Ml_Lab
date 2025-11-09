import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
import streamlit as st

class DatasetLoader:
    """Load built-in and custom datasets"""
    
    @staticmethod
    def get_builtin_datasets():
        return {
            "Iris": load_iris,
            "Wine": load_wine,
            "Breast Cancer": load_breast_cancer,
            "Diabetes": load_diabetes
        }
    
    @staticmethod
    def load_dataset(dataset_name):
        """Load a dataset by name"""
        datasets = DatasetLoader.get_builtin_datasets()
        if dataset_name in datasets:
            data = datasets[dataset_name]()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df, data.feature_names, data.target
        return None, None, None
    
    @staticmethod
    def load_custom_dataset(uploaded_file):
        """Load user-uploaded CSV"""
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
