# app.py
import streamlit as st
from llm_chain import create_qa_chain
from vector_store import MyVectorStore
import PyPDF2  # New: To read resumes

st.set_page_config(page_title="AI Resume Screener", page_icon="📄")
st.title("📄 AI Resume Screening Chatbot")

# Initialize our custom store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = MyVectorStore()

# --- SIDEBAR: UPLOAD SECTION ---
with st.sidebar:
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload candidate PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Resumes"):
        if uploaded_files:
            all_text_docs = []
            for file in uploaded_files:
                # Read PDF text
                pdf_reader = PyPDF2.PdfReader(file)
                text = f"--- Resume: {file.name} ---\n"
                for page in pdf_reader.pages:
                    text += page.extract_text()
                all_text_docs.append(text)
            
            # Store the real text in our vector store
            st.session_state.vector_store.add_documents(all_text_docs)
            st.success(f"Processed {len(uploaded_files)} resumes!")
        else:
            st.warning("Please upload some PDFs first.")

# --- MAIN CHAT INTERFACE ---
qa_chain = create_qa_chain(st.session_state.vector_store)

question = st.text_input("Ask a question about the candidates:")

if question:
    with st.spinner("Searching resumes..."):
        docs = qa_chain["retriever"].get_relevant_documents(question)
        
        # Check if we actually have data
        if not docs or "Example content" in docs[0]:
            st.warning("The vector store is empty or using dummy data. Please upload resumes in the sidebar!")
        else:
            response = qa_chain["llm_call"](question, docs)
            st.markdown("### Analysis")
            st.write(response)
