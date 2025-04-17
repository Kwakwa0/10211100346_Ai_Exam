import streamlit as st
import os
import base64
import re
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

def llm_section():
    # ------------------------------
    # 1. Environment Setup
    # ------------------------------
    load_dotenv(find_dotenv())
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        st.error("🚫 Gemini API key not found. Please add GEMINI_API_KEY to your .env file.")
        st.stop()

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # ------------------------------
    # 2. Header & Instructions
    # ------------------------------
    st.title("🧠 Ask Gemini: Intelligent Q&A over PDFs")
    st.markdown("""
    Welcome to the **LLM Q&A Assistant using Gemini AI**!  
    Ask natural language questions based on any preloaded PDF document using *Retrieval-Augmented Generation (RAG)*.

    📌 **Steps to Use:**
    1. Pick a document from the dataset.
    2. Review or download the PDF.
    3. Adjust passage retrieval settings.
    4. Enter a question and hit "🔍 Ask Gemini" to get an intelligent answer!
    """)

    # ------------------------------
    # 3. PDF Utilities
    # ------------------------------
    @st.cache_data(show_spinner=False)
    def extract_text_from_pdf(file_path):
        """Extract plain text from a PDF file."""
        try:
            import PyPDF2
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                return "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            st.error(f"⚠️ PDF Read Error: {e}")
            return ""

    @st.cache_data(show_spinner=False)
    def get_pdf_base64(file_path):
        try:
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            st.error(f"⚠️ PDF Encode Error: {e}")
            return ""

    def display_pdf_preview_and_download(file_path):
        base64_pdf = get_pdf_base64(file_path)
        if base64_pdf:
            file_name = os.path.basename(file_path)
            st.download_button("⬇️ Download PDF", base64_pdf, file_name=file_name, mime="application/pdf")
            st.caption("💡 You can open the downloaded PDF locally for full view.")

    # ------------------------------
    # 4. Dataset Selection
    # ------------------------------
    st.sidebar.header("📁 Choose a PDF Dataset")
    datasets = {
        "2025 Budget Statement": "2025-Budget-Statement-and-Economic-Policy_v4.pdf",
        "Student Handbook": "handbook.pdf"
    }
    dataset_name = st.sidebar.selectbox("Available PDFs:", list(datasets.keys()))
    selected_path = datasets[dataset_name]

    st.subheader(f"📄 Preview: {dataset_name}")
    display_pdf_preview_and_download(selected_path)

    # ------------------------------
    # 5. RAG Retrieval Settings
    # ------------------------------
    with st.expander("🔧 Advanced Retrieval Settings"):
        num_passages = st.slider("How many top passages should Gemini consider?", 1, 10, 3)
        st.caption("These passages will guide Gemini to generate a more accurate answer.")

    with st.spinner("🔍 Extracting and indexing PDF..."):
        content = extract_text_from_pdf(selected_path)
        paragraphs = content.split("\n\n") if content else []

    # ------------------------------
    # 6. Ask Gemini
    # ------------------------------
    st.subheader("💬 Ask Your Question")
    query = st.text_input("Type your question below:", placeholder="e.g., What is the total expenditure for 2025?")
    ask_button = st.button("🔍 Ask Gemini")

    if ask_button and query:
        with st.spinner("🤖 Gemini is thinking..."):
            query_terms = set(re.findall(r'\w+', query.lower()))
            ranked = []
            for p in paragraphs:
                p_terms = set(re.findall(r'\w+', p.lower()))
                score = len(query_terms.intersection(p_terms))
                ranked.append((score, p))
            ranked.sort(key=lambda x: x[0], reverse=True)
            retrieval_context = "\n\n".join([p for score, p in ranked[:num_passages]])

            try:
                response = model.generate_content([retrieval_context, query])
                st.success("✅ Gemini’s Answer:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error from Gemini: {e}")

        # Optional: show the context used
        with st.expander("📚 View Source Passages"):
            for idx, (score, para) in enumerate(ranked[:num_passages]):
                st.markdown(f"**Passage {idx+1} (score: {score})**")
                st.write(para.strip())
                st.markdown("---")

    elif ask_button and not query:
        st.warning("❗ Please type a question before clicking 'Ask Gemini'.")
