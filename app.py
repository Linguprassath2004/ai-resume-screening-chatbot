# app.py
import streamlit as st
from llm_chain import create_qa_chain
from some_vector_store_module import MyVectorStore  # replace with your actual vector store class

# Initialize vector store
vector_store = MyVectorStore()

# Create the QA chain
qa_chain = create_qa_chain(vector_store)

st.title("AI Resume Screening Chatbot")

question = st.text_input("Ask a question about the resumes:")

if question:
    # Retrieve relevant documents
    docs = qa_chain["retriever"].get_relevant_documents(question)
    
    # Generate response from LLM
    raw_answer = qa_chain["llm_call"](question, context_docs=docs)
    
    # Parse and display output
    final_answer = qa_chain["output_parser"].parse(raw_answer)
    st.write(final_answer)
