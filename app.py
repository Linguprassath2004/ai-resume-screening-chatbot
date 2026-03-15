# app.py
import streamlit as st
from llm_chain import create_qa_chain
from vector_store import MyVectorStore
from pypdf import PdfReader  # Modern 2026 PDF library

st.set_page_config(page_title="AI Resume Screener", page_icon="📄")
st.title("📄 AI Resume Screening Chatbot")

# Use session_state so the data doesn't disappear when the screen refreshes
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = MyVectorStore()

# --- SIDEBAR: FILE UPLOADER ---
with st.sidebar:
    st.header("Upload Center")
    uploaded_files = st.file_uploader("Upload candidate resumes (PDF)", type="pdf", accept_multiple_files=True)
    
    if st.button("Process & Index Resumes"):
        if uploaded_files:
            new_texts = []
            for uploaded_file in uploaded_files:
                reader = PdfReader(uploaded_file)
                # Combine all pages of the PDF into one string
                text = f"Candidate: {uploaded_file.name}\n"
                for page in reader.pages:
                    text += page.extract_text()
                new_texts.append(text)
            
            st.session_state.vector_store.add_documents(new_texts)
            st.success(f"Successfully processed {len(uploaded_files)} resumes!")
        else:
            st.warning("Please upload at least one PDF.")

# --- MAIN CHAT ---
qa_chain = create_qa_chain(st.session_state.vector_store)

question = st.text_input("Ask a question about the candidates:")

if question:
    with st.spinner("Analyzing resumes..."):
        docs = qa_chain["retriever"].get_relevant_documents(question)
        
        # Guard rail: Don't call the AI if there's no data
        if "No resumes uploaded" in docs[0]:
            st.error("Please upload and process resumes in the sidebar first!")
        else:
            response = qa_chain["llm_call"](question, docs)
            st.markdown("### HR Assistant Analysis")
            st.write(response)
