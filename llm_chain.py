# llm_chain.py
import os
import requests
from langchain.prompts.chat import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser

# GROQ API
GROQ_API_URL = "https://api.groq.ai/v1/query"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set in Streamlit secrets

def create_qa_chain(vector_store):
    """
    Returns a QA chain object for querying resumes using GROQ LLM.
    """
    retriever = vector_store.as_retriever()

    prompt_template = ChatPromptTemplate.from_template(
        "You are an AI assistant. Answer the user's question based only on the retrieved documents.\n\n"
        "Question: {question}\nAnswer:"
    )

    output_parser = StrOutputParser()

    def llm_call(question, context_docs):
        """
        Makes a POST request to GROQ API to get answer.
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
        # Return the "answer" field from GROQ's JSON response
        return response.json().get("answer", "")

    return {
        "llm_call": llm_call,
        "retriever": retriever,
        "prompt_template": prompt_template,
        "output_parser": output_parser
    }
