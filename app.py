import streamlit as st
import pandas as pd
import numpy as np
from models.model_registery import ModelRegistry
from models.trainer import ModelTrainer
from data.datasets import DatasetLoader
from visualizations.metrics import MetricsVisualizer


# Page configuration
st.set_page_config(
    page_title="Open ML Lab",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Interactive ML Training Platform")
st.markdown("Select a model, dataset, and tune hyperparameters in real-time!")

# Sidebar for dataset selection
st.sidebar.header("📊 Dataset Selection")
dataset_option = st.sidebar.radio(
    "Choose dataset source:",
    ["Built-in Datasets", "Upload Custom Dataset"]
)

df = None
feature_names = None
target_names = None

if dataset_option == "Built-in Datasets":
    dataset_name = st.sidebar.selectbox(
        "Select Dataset:",
        list(DatasetLoader.get_builtin_datasets().keys())
    )
    if st.sidebar.button("Load Dataset"):
        df, feature_names, target_names = DatasetLoader.load_dataset(dataset_name)
        st.session_state['df'] = df
        st.session_state['feature_names'] = feature_names
        # For built-in datasets, assuming target column is "target"
        st.session_state['target_col'] = 'target'
        st.sidebar.success(f"✅ {dataset_name} dataset loaded!")

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file:
        df = DatasetLoader.load_custom_dataset(uploaded_file)
        if df is not None:
            st.session_state['df'] = df
            # Let user select the target column for custom dataset
            target_col = st.sidebar.selectbox(
                "Select the target column",
                options=df.columns,
                key='target_col_select'
            )
            st.session_state['target_col'] = target_col
            # Set feature names excluding target column
            st.session_state['feature_names'] = [col for col in df.columns if col != target_col]
            st.sidebar.success("✅ Custom dataset loaded!")
            

# Retrieve dataset from session state
if 'df' in st.session_state:
    df = st.session_state['df']

    if 'target_col' in st.session_state and st.session_state['target_col']:
        target_col = st.session_state['target_col']

        # Display dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1] - 1)

        

        # Show dataset preview
        with st.expander("📋 View Dataset"):
            st.dataframe(df.head(20))

            # Dataset statistics
            st.subheader("Statistical Summary")
            st.write(df.describe())

        # Model selection and hyperparameters
        st.sidebar.header("🤖 Model Configuration")

        model_name = st.sidebar.selectbox(
            "Select Model:",
            list(ModelRegistry.get_available_models().keys())
        )

        # Get hyperparameters for selected model
        st.sidebar.subheader("⚙️ Hyperparameters")
        hyperparams_config = ModelRegistry.get_hyperparameters(model_name)
        selected_params = {}

        for param_name, param_config in hyperparams_config.items():
            if param_config['type'] == 'slider':
                selected_params[param_name] = st.sidebar.slider(
                    param_name,
                    min_value=param_config['min'],
                    max_value=param_config['max'],
                    value=param_config['default'],
                    step=param_config.get('step', 1)
                )
            elif param_config['type'] == 'selectbox':
                selected_params[param_name] = st.sidebar.selectbox(
                    param_name,
                    options=param_config['options'],
                    index=param_config['options'].index(param_config['default'])
                )

        # Training configuration
        st.sidebar.subheader("🎯 Training Configuration")
        test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
        random_state = st.sidebar.number_input("Random State", 0, 100, 42)

        # Train button
        if st.sidebar.button("🚀 Train Model", type="primary"):
            # Prepare data
            X = df.drop(target_col, axis=1).values
            y = df[target_col].values

            # Initialize model with selected hyperparameters
            model_class = ModelRegistry.get_available_models()[model_name]
            model = model_class(**selected_params)

            # Train model
            trainer = ModelTrainer(model, X, y, test_size=test_size, random_state=random_state)

            with st.spinner("Training in progress..."):
                metrics = trainer.train()

            # Store results in session state
            st.session_state['metrics'] = metrics
            st.session_state['model'] = model
            st.session_state['trainer'] = trainer

        # Display results
        if 'metrics' in st.session_state:
            metrics = st.session_state['metrics']

            st.success(f"✅ Model trained successfully in {metrics['training_time']:.2f} seconds!")

            # Metrics display
            st.header("📊 Model Performance")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Train Accuracy", f"{metrics['train_accuracy']:.3f}")
            col2.metric("Test Accuracy", f"{metrics['test_accuracy']:.3f}")
            col3.metric("Precision", f"{metrics['precision']:.3f}")
            col4.metric("Recall", f"{metrics['recall']:.3f}")
            col5.metric("F1 Score", f"{metrics['f1_score']:.3f}")

            # Visualizations
            st.header("📈 Visualizations")

            tab1, tab2, tab3 = st.tabs(["Metrics Comparison", "Confusion Matrix", "Feature Importance"])

            with tab1:
                fig_metrics = MetricsVisualizer.plot_metrics_comparison(metrics)
                st.plotly_chart(fig_metrics, use_container_width=True)

            with tab2:
                fig_cm = MetricsVisualizer.plot_confusion_matrix(
                    metrics['confusion_matrix'],
                    class_names=[f"Class {i}" for i in range(len(np.unique(df[target_col])))]
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with tab3:
                if 'feature_names' in st.session_state and st.session_state['feature_names'] is not None:
                    fig_importance = MetricsVisualizer.plot_feature_importance(
                        st.session_state['model'],
                        st.session_state['feature_names']
                    )
                    if fig_importance:
                        st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model type")
                else:
                    st.info("Feature names not available")

    else:
        st.info("👈 Please select the target column from the sidebar to view dataset details and continue.")

else:
    st.info("👈 Please load a dataset from the sidebar to get started!")
