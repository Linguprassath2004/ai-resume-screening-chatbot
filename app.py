import streamlit as st
from pdf_loader import extract_text_from_pdf
from vector_store import create_vector_store
from llm_chain import create_qa_chain

st.set_page_config(page_title="AI Resume Screening Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 AI Resume Screening Chatbot (RAG + LLaMA 3)")

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

with col2:
    job_description = st.text_area("Paste Job Description")

if resume_file and job_description:

    with st.spinner("Processing..."):
        resume_text = extract_text_from_pdf(resume_file)

        vector_store = create_vector_store(resume_text)
        qa_chain = create_qa_chain(vector_store)

        query = f"""
        Compare the resume with the following job description.

        Job Description:
        {job_description}
        """

        result = qa_chain.invoke(query)

    st.subheader("AI Evaluation")
    st.write(result)