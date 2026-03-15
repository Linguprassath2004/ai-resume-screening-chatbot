# app.py
import streamlit as st
from llm_chain import create_qa_chain
from vector_store import MyVectorStore

st.set_page_config(page_title="Resume Screener", page_icon="📄")
st.title("📄 AI Resume Screening Chatbot")

# 1. Initialize Vector Store & Chain
@st.cache_resource
def init_chain():
    vs = MyVectorStore()
    return create_qa_chain(vs)

qa_chain = init_chain()

# 2. UI for User Input
question = st.text_input("Ask about candidate skills or experience:", placeholder="e.g., Who has experience with Python and AWS?")

if question:
    with st.spinner("Analyzing resumes..."):
        # Retrieve relevant documents
        docs = qa_chain["retriever"].get_relevant_documents(question)
        
        # Generate response
        raw_response = qa_chain["llm_call"](question, docs)
        
        # Parse and display
        final_answer = qa_chain["output_parser"].parse(raw_response)
        
        st.markdown("### Answer")
        st.write(final_answer)
        
        with st.expander("View Source Documents"):
            for i, doc in enumerate(docs):
                st.info(f"Source {i+1}: {doc}")
