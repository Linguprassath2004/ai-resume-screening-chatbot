# llm_chain.py

# Import only necessary modules
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_qa_chain(vector_store):
    """
    Create a question-answering chain using LangChain and Ollama LLM.

    Args:
        vector_store: A retriever-compatible vector store instance (e.g., Chroma, FAISS).

    Returns:
        A QA chain object that can be called to ask questions.
    """

    # Initialize the LLM
    llm = ChatOllama(model="llama3")  # Use your model here

    # Prepare retriever from the vector store
    retriever = vector_store.as_retriever()

    # Define a prompt template
    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant designed to answer questions based on the user's uploaded resumes.

Instructions:
- Answer concisely and accurately.
- If unsure, say 'I don't know'.
- Only use information from the retrieved documents.

Question: {question}
Answer:
""")

    # Optional: define output parser (can customize if needed)
    output_parser = StrOutputParser()

    # Return a dictionary with all chain components for usage
    qa_chain = {
        "llm": llm,
        "retriever": retriever,
        "prompt": prompt,
        "output_parser": output_parser
    }

    return qa_chain
