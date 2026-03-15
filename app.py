# app.py
import streamlit as st
from llm_chain import create_qa_chain
from some_vector_store_module import MyVectorStore  # your vector store

# Initialize vector store
vector_store = MyVectorStore()

# Create the QA chain
qa_chain = create_qa_chain(vector_store)

st.title("AI Resume Screening Chatbot")

question = st.text_input("Ask a question about the resumes:")

if question:
    # Retrieve relevant docs
    docs = qa_chain["retriever"].get_relevant_documents(question)
    
    # Generate response
    response = qa_chain["llm"].generate(qa_chain["prompt"].format(question=question))
    
    # Optional: parse the output
    final_answer = qa_chain["output_parser"].parse(response)

    st.write(final_answer)
