# app.py
import streamlit as st
from llm_chain import create_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import os

# -----------------------
# Streamlit Page Setup
# -----------------------
st.set_page_config(page_title="🤖 AI Resume Screening Chatbot", layout="wide")
st.title("🤖 AI Resume Screening Chatbot (RAG + LLaMA 3)")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)", type=["pdf"], help="Limit 200MB per file"
)

# -----------------------
# Process Resume
# -----------------------
if uploaded_file is not None:
    # Load PDF
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()

    # Split text into chunks for vector store
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(documents)

    # Create Chroma Vector Store (in-memory)
    vector_store = Chroma.from_documents(docs, collection_name="resume_collection")

    st.success("✅ Resume uploaded and indexed successfully!")

    # -----------------------
    # Create QA Chain
    # -----------------------
    qa_chain = create_qa_chain(vector_store)

    # -----------------------
    # Ask Questions
    # -----------------------
    st.subheader("Ask a question about the resume")
    query = st.text_input("Type your question here:")

    if query:
        with st.spinner("🤖 AI is generating answer..."):
            answer = qa_chain(query)
            st.markdown(f"**Answer:** {answer}")
