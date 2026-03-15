# llm_chain.py
import os
import requests
from langchain.prompts import ChatPromptTemplate

# Use environment variable for your GROQ API key
GROQ_API_URL = "https://api.groq.ai/v1/query"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # set this in Streamlit secrets

# Custom simple string parser (replaces StrOutputParser)
class SimpleStrOutputParser:
    def parse(self, text: str) -> str:
        return text.strip()

def create_qa_chain(vector_store):
    retriever = vector_store.as_retriever()

    # Prompt template
    prompt_template = ChatPromptTemplate.from_template(
        "You are an AI assistant. Answer the user's question based only on the retrieved documents.\n\n"
        "Question: {question}\nAnswer:"
    )

    output_parser = SimpleStrOutputParser()

    # LLM call using GROQ API
    def llm_call(question, context_docs):
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "query": f"{prompt_template.format(question=question)}\nContext: {context_docs}"
        }
        response = requests.post(GROQ_API_URL, json=data, headers=headers)
        response.raise_for_status()
        # GROQ returns answer in JSON key "answer" (adjust if your API is different)
        return response.json().get("answer", "")

    return {
        "llm_call": llm_call,
        "retriever": retriever,
        "prompt_template": prompt_template,
        "output_parser": output_parser
    }
