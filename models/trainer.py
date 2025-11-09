import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import streamlit as st

class ModelTrainer:
    """Train and evaluate ML models"""
    
    def __init__(self, model, X, y, test_size=0.2, random_state=42):
        self.model = model
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def train(self):
        """Train the model and return metrics"""
        start_time = time.time()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Training model...")
        progress_bar.progress(30)
        
        # Train model
        self.model.fit(self.X_train_scaled, self.y_train)
        
        progress_bar.progress(70)
        status_text.text("Evaluating model...")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        training_time = time.time() - start_time
        
        progress_bar.progress(100)
        status_text.text("Training complete!")
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(self.y_train, y_train_pred),
            'test_accuracy': accuracy_score(self.y_test, y_test_pred),
            'precision': precision_score(self.y_test, y_test_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_test_pred, average='weighted'),
            'f1_score': f1_score(self.y_test, y_test_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(self.y_test, y_test_pred),
            'training_time': training_time,
            'y_test': self.y_test,
            'y_pred': y_test_pred
        }
        
        return metrics
