# app.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import Chroma
from langchain_core.document_loaders import PyPDFLoader
import os

# -----------------------
# Streamlit App Layout
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

    # Split into chunks for vector store
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(documents)

    # Create Chroma Vector Store
    vector_store = Chroma.from_documents(docs, collection_name="resume_collection")

    st.success("✅ Resume uploaded and indexed successfully!")

    # -----------------------
    # Create QA Chain
    # -----------------------
    def create_qa_chain(vector_store):
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

        retriever = vector_store.as_retriever()

        prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that answers questions about a candidate's resume. 
Use ONLY the information from the resume provided.

Resume Content:
{context}

Question:
{question}

Answer the question clearly and concisely based on the resume.
        """)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Return callable function
        def run_chain(query):
            docs = retriever.get_relevant_documents(query)
            context = format_docs(docs)
            return llm.invoke({"context": context, "question": query})

        return run_chain

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
