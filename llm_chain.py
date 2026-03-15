# llm_chain.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser

def create_qa_chain(vector_store):
    """
    Creates a QA chain for retrieving answers from a vector store
    using OpenAI's LLM.
    """
    # Initialize LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # Use GPT-4 if you have access
        temperature=0
    )

    # Convert your vector store to a retriever
    retriever = vector_store.as_retriever()

    # Define a prompt template
    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant. Answer the user's question based only on the retrieved documents.

Question: {question}
Answer:
""")

    # Output parser to extract text
    output_parser = StrOutputParser()

    # Return everything in a dict (or you can wrap into LangChain Chain if preferred)
    return {
        "llm": llm,
        "retriever": retriever,
        "prompt": prompt,
        "output_parser": output_parser
    }
