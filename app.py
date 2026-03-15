import streamlit as st
from llm_chain import create_qa_chain
from some_vector_store_module import MyVectorStore  # your vector store

vector_store = MyVectorStore()
qa_chain = create_qa_chain(vector_store)

st.title("AI Resume Screening Chatbot")
question = st.text_input("Ask a question about the resumes:")

if question:
    docs = qa_chain["retriever"].get_relevant_documents(question)
    answer = qa_chain["llm_call"](question, docs)
    final_answer = qa_chain["output_parser"].parse(answer)
    st.write(final_answer)
