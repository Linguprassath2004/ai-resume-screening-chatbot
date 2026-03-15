# llm_chain.py
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

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

    def run_chain(query):
        docs = retriever.get_relevant_documents(query)
        context = format_docs(docs)
        return llm.invoke({"context": context, "question": query})

    return run_chain
