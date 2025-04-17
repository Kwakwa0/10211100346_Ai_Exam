import streamlit as st
from pages.regression import regression_section
from pages.clustering import clustering_section
from pages.neural_network import neural_net_section
from pages.llm import llm_section

def main():
    st.set_page_config(
        page_title="AI/ML Explorer",
        page_icon="🧠",
        layout="wide"
    )
    
    st.sidebar.title("AI/ML Explorer")
    st.sidebar.markdown("Explore different machine learning and AI techniques")
    
    section = st.sidebar.radio(
        "Select Section",
        ("Regression", "Clustering", "Neural Network", "Large Language Model")
    )
    
    st.title(f"{section} Explorer")
    
    if section == "Regression":
        regression_section()
    elif section == "Clustering":
        clustering_section()
    elif section == "Neural Network":
        neural_net_section()
    elif section == "Large Language Model":
        llm_section()

if __name__ == "__main__":
    main()