# llm_chain.py
import os
import requests
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser

GROQ_API_URL = "https://api.groq.ai/v1/query"  # replace with actual endpoint
GROQ_API_KEY = os.getenv("GROQ_API_KEY")      # set in Streamlit Secrets

def create_qa_chain(vector_store):
    """
    Simple QA chain using GROQ API and a vector store
    """
    retriever = vector_store.as_retriever()

    prompt_template = """
You are an AI assistant. Answer the user's question based only on the retrieved documents.

Question: {question}
Answer:
"""

    output_parser = StrOutputParser()

    def llm_call(question, context_docs):
        """
        Send query to GROQ API with context documents.
        """
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "query": f"{prompt_template.format(question=question)}\nContext: {context_docs}"
        }

        response = requests.post(GROQ_API_URL, json=data, headers=headers)
        response.raise_for_status()
        return response.json().get("answer", "")

    return {
        "llm_call": llm_call,
        "retriever": retriever,
        "prompt_template": prompt_template,
        "output_parser": output_parser
    }
