import streamlit as st
import os
from pypdf import PdfReader
from llm_chain import create_qa_chain
from vector_store import MyVectorStore

# Page Configuration
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

# 1. Initialize Session State
# This ensures your vector store and documents stay in memory while you use the app
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = MyVectorStore()
    st.session_state.processed_files = []

# 2. Setup the QA Chain
# We pass the vector store from session state to the chain
qa_chain = create_qa_chain(st.session_state.vector_store)

# --- SIDEBAR: UPLOAD & MANAGEMENT ---
with st.sidebar:
    st.title("📂 Resume Management")
    st.info("Upload candidate resumes in PDF format to begin analysis.")
    
    uploaded_files = st.file_uploader(
        "Upload Resumes", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process & Index Resumes", use_container_width=True):
        if uploaded_files:
            with st.spinner("Extracting text from PDFs..."):
                new_texts = []
                for uploaded_file in uploaded_files:
                    # Avoid processing the same file twice
                    if uploaded_file.name not in st.session_state.processed_files:
                        try:
                            reader = PdfReader(uploaded_file)
                            text = f"--- START OF RESUME: {uploaded_file.name} ---\n"
                            for page in reader.pages:
                                content = page.extract_text()
                                if content:
                                    text += content + "\n"
                            text += f"--- END OF RESUME: {uploaded_file.name} ---\n"
                            
                            new_texts.append(text)
                            st.session_state.processed_files.append(uploaded_file.name)
                        except Exception as e:
                            st.error(f"Error reading {uploaded_file.name}: {e}")
                
                if new_texts:
                    # This replaces the 'dummy' data in your vector_store
                    st.session_state.vector_store.add_documents(new_texts)
                    st.success(f"Added {len(new_texts)} new resumes to knowledge base!")
                else:
                    st.warning("No new resumes were found to process.")
        else:
            st.warning("Please select PDF files first.")

    st.divider()
    st.write(f"📊 **Resumes Indexed:** {len(st.session_state.processed_files)}")
    if st.button("🗑️ Clear All Data"):
        st.session_state.vector_store.documents = []
        st.session_state.processed_files = []
        st.rerun()

# --- MAIN INTERFACE ---
st.title("📄 AI Resume Screening Chatbot")
st.markdown("""
    Ask questions about your candidates, such as:
    * *'Which candidates have more than 5 years of experience in Python?'*
    * *'Summarize the top skills for all applicants.'*
    * *'Who is the best fit for a Senior Data Scientist role?'*
""")

# User input
question = st.text_input("💬 Ask a question about the candidates:", placeholder="e.g., List candidates with AWS certification...")

if question:
    # 1. Get documents from our store
    docs = qa_chain["retriever"].get_relevant_documents(question)
    
    # 2. Check if we actually have data to work with
    if not st.session_state.processed_files:
        st.error("⚠️ No resumes found. Please upload and process PDFs in the sidebar.")
    else:
        with st.spinner("HR Assistant is thinking..."):
            # 3. Call the Groq LLM via our chain
            response = qa_chain["llm_call"](question, docs)
            
            # 4. Parse and display
            final_answer = qa_chain["output_parser"].parse(response)
            
            st.markdown("### 🤖 HR Assistant Analysis")
            st.write(final_answer)
            
            # Optional: Show what the AI "read" to get this answer
            with st.expander("🔍 View Raw Context"):
                for i, d in enumerate(docs):
                    st.text_area(f"Document Segment {i+1}", d, height=150)

# --- FOOTER ---
st.divider()
st.caption("Powered by Groq Llama 3 & Streamlit • Created for AI Resume Screening")
