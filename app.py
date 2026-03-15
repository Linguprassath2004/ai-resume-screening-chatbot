# app.py
import streamlit as st
from llm_chain import create_qa_chain
from vector_store import MyVectorStore

# Initialize vector store
vector_store = MyVectorStore()

# Create QA chain
qa_chain = create_qa_chain(vector_store)

st.title("AI Resume Screening Chatbot")

question = st.text_input("Ask a question about the resumes:")

if question:
    # Retrieve relevant documents
    docs = qa_chain["retriever"].get_relevant_documents(question)
    context = "\n".join(docs)

    # Generate response using GROQ API
    response = qa_chain["llm_call"](question, context)

    # Parse and display
    final_answer = qa_chain["output_parser"].parse(response)
    st.write(final_answer)
