# llm_chain.py
import os
import requests
from langchain.prompts.prompt import PromptTemplate  # safe import
from vector_store import MyVectorStore  # your vector store wrapper

# GROQ API
GROQ_API_URL = "https://api.groq.ai/v1/query"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # store in Streamlit secrets

class SimpleStrOutputParser:
    def parse(self, text: str) -> str:
        return text.strip()

def create_qa_chain(vector_store: MyVectorStore):
    retriever = vector_store.as_retriever()

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are an AI assistant. Answer the user's question based only on the retrieved documents.\n\n"
            "Question: {question}\nAnswer:"
        )
    )

    output_parser = SimpleStrOutputParser()

    def llm_call(question: str, context_docs: str) -> str:
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
