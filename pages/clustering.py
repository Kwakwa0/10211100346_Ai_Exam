import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

def clustering_section():
    st.header("Clustering Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file for clustering", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Select features
        feature_cols = st.multiselect("Select feature columns for clustering", df.columns)
        
        if len(feature_cols) >= 2:
            # Preprocessing
            scaler = StandardScaler()
            X = df[feature_cols].dropna()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal k
            st.subheader("Determine Number of Clusters")
            max_clusters = min(10, len(X_scaled))
            n_clusters = st.slider("Select number of clusters", 2, max_clusters, 3)
            
            # Apply KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Visualize clusters
            st.subheader("Cluster Visualization")
            
            if len(feature_cols) == 2:
                # 2D plot
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
                centers = kmeans.cluster_centers_
                ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
                ax.set_xlabel(feature_cols[0])
                ax.set_ylabel(feature_cols[1])
                ax.set_title("K-Means Clustering Results")
                plt.colorbar(scatter)
                st.pyplot(fig)
            else:
                # PCA for dimensionality reduction
                pca = PCA(n_components=3)
                X_pca = pca.fit_transform(X_scaled)
                
                # 3D plot
                plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
                plot_df['Cluster'] = clusters.astype(str)
                
                fig = px.scatter_3d(
                    plot_df, x='PC1', y='PC2', z='PC3', color='Cluster',
                    title="3D Cluster Visualization (PCA Reduced)",
                    hover_data={col: False for col in plot_df.columns}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Add clusters to original data
            df_clustered = df.copy()
            df_clustered['Cluster'] = clusters
            
            # Download clustered data
            st.subheader("Download Clustered Data")
            st.download_button(
                label="Download CSV with clusters",
                data=df_clustered.to_csv(index=False).encode('utf-8'),
                file_name='clustered_data.csv',
                mime='text/csv'
            )