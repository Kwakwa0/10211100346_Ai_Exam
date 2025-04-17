import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

def neural_net_section():
    st.header("Neural Network Classifier")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file for classification", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Select target column
        target_col = st.selectbox("Select target column", df.columns)
        
        # Select features
        feature_cols = st.multiselect("Select feature columns", [col for col in df.columns if col != target_col])
        
        if feature_cols and target_col:
            # Preprocessing
            X = df[feature_cols].values
            y = df[target_col].values
            
            # Handle numeric vs categorical target
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                num_classes = len(le.classes_)
                st.info(f"Target classes: {list(le.classes_)}")
            else:
                # For regression problems
                num_classes = 1
            
            # Train-test split
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Model configuration
            st.subheader("Model Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                hidden_layers = st.number_input("Number of hidden layers", 1, 5, 2)
            with col2:
                neurons = st.number_input("Neurons per layer", 8, 256, 64)
            with col3:
                dropout = st.slider("Dropout rate", 0.0, 0.5, 0.2)
            
            # Training configuration
            col1, col2, col3 = st.columns(3)
            with col1:
                learning_rate = st.number_input("Learning rate", 1e-5, 1e-1, 1e-3, format="%.0e")
            with col2:
                epochs = st.number_input("Epochs", 10, 200, 50)
            with col3:
                batch_size = st.number_input("Batch size", 8, 128, 32)
            
            # Build model
            model = Sequential()
            model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))
            
            for _ in range(hidden_layers - 1):
                model.add(Dense(neurons, activation='relu'))
                model.add(Dropout(dropout))
            
            if num_classes == 1:
                model.add(Dense(1))
                loss = 'mse'
                metrics = ['mae']
            elif num_classes == 2:
                model.add(Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                model.add(Dense(num_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
            
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            
            # Train model
            st.subheader("Model Training")
            if st.button("Train Model"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Callback for updating progress
                class TrainingCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch + 1}/{epochs} - loss: {logs['loss']:.4f}")
                
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[TrainingCallback(), early_stopping],
                    verbose=0
                )
                
                # Plot training history
                fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                ax[0].plot(history.history['loss'], label='Training Loss')
                ax[0].plot(history.history['val_loss'], label='Validation Loss')
                ax[0].set_title('Loss Over Epochs')
                ax[0].set_xlabel('Epoch')
                ax[0].set_ylabel('Loss')
                ax[0].legend()
                
                if 'accuracy' in history.history:
                    ax[1].plot(history.history['accuracy'], label='Training Accuracy')
                    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
                    ax[1].set_title('Accuracy Over Epochs')
                    ax[1].set_xlabel('Epoch')
                    ax[1].set_ylabel('Accuracy')
                    ax[1].legend()
                
                st.pyplot(fig)
                
                # Evaluate model
                st.subheader("Model Evaluation")
                test_loss, *test_metrics = model.evaluate(X_test, y_test, verbose=0)
                st.metric("Test Loss", f"{test_loss:.4f}")
                
                if num_classes == 1:
                    st.metric("Test MAE", f"{test_metrics[0]:.4f}")
                else:
                    st.metric("Test Accuracy", f"{test_metrics[0]:.4f}")
                
                # Save model
                model.save('models/neural_net.h5')
                st.success("Model trained and saved successfully!")
            
            # Make predictions
            st.subheader("Make Predictions")
            if st.button("Load Pre-trained Model"):
                try:
                    model = tf.keras.models.load_model('models/neural_net.h5')
                    st.success("Model loaded successfully!")
                except:
                    st.error("No trained model found. Please train a model first.")
            
            if 'model' in locals():
                input_data = {}
                for i, feature in enumerate(feature_cols):
                    input_data[feature] = st.number_input(
                        f"Enter {feature}",
                        value=float(df[feature].mean())
                    )
                
                if st.button("Predict"):
                    input_array = np.array([[input_data[col] for col in feature_cols]])
                    input_array = scaler.transform(input_array)
                    prediction = model.predict(input_array)
                    
                    if num_classes == 1:
                        st.success(f"Predicted value: {prediction[0][0]:.2f}")
                    elif num_classes == 2:
                        pred_class = "Class 1" if prediction[0][0] > 0.5 else "Class 0"
                        st.success(f"Predicted class: {pred_class} (confidence: {prediction[0][0]:.2f})")
                    else:
                        pred_class = np.argmax(prediction[0])
                        confidence = np.max(prediction[0])
                        if 'le' in locals():
                            pred_class = le.inverse_transform([pred_class])[0]
                        st.success(f"Predicted class: {pred_class} (confidence: {confidence:.2f})")