import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def regression_section():
    st.header("Regression Problem")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file for regression", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Select target column
        target_col = st.selectbox("Select target column", df.columns)
        
        # Select features
        feature_cols = st.multiselect("Select feature columns", [col for col in df.columns if col != target_col])
        
        if feature_cols and target_col:
            # Preprocessing options
            st.subheader("Data Preprocessing")
            handle_missing = st.radio("Handle missing values", 
                                     ["Drop rows with missing values", "Fill with mean", "Fill with median"])
            
            if handle_missing == "Drop rows with missing values":
                df = df.dropna(subset=feature_cols + [target_col])
            elif handle_missing == "Fill with mean":
                df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
                df[target_col] = df[target_col].fillna(df[target_col].mean())
            elif handle_missing == "Fill with median":
                df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
                df[target_col] = df[target_col].fillna(df[target_col].median())
            
            # Train-test split
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
            X = df[feature_cols]
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("Mean Absolute Error", f"{mae:.4f}")
            col2.metric("R² Score", f"{r2:.4f}")
            
            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted Values")
            st.pyplot(fig)
            
            # Custom prediction
            st.subheader("Make Custom Predictions")
            input_data = {}
            for feature in feature_cols:
                input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
            
            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted {target_col}: {prediction:.2f}")