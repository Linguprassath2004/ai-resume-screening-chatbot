# llm_chain.py
import os
import requests
import streamlit as st
from langchain_core.prompts import PromptTemplate  # Modern 2026 import path
from vector_store import MyVectorStore

# Groq API Configuration (Updated for 2026 standards)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Use Streamlit Secrets for deployment, fall back to environment variable for local dev
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

class SimpleStrOutputParser:
    def parse(self, text: str) -> str:
        return text.strip() if text else "No response generated."

def create_qa_chain(vector_store: MyVectorStore):
    retriever = vector_store.as_retriever()

    # Define the instruction for the AI
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template=(
            "You are a professional HR Assistant. Use the following resume snippets to answer the question.\n"
            "If the answer isn't in the context, say you don't know.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
    )

    output_parser = SimpleStrOutputParser()

    def llm_call(question: str, context_docs: list) -> str:
        if not GROQ_API_KEY:
            return "Error: GROQ_API_KEY not found. Please add it to your Secrets."

        # Prepare the context string from the retrieved document list
        context_str = "\n".join(context_docs)
        formatted_prompt = prompt_template.format(question=question, context=context_str)

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        
        # Groq uses OpenAI-compatible chat completion payload
        # In llm_chain.py, update the payload
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": formatted_prompt}],
            "temperature": 0.2,
            "max_tokens": 1024 
        }

                
        
        try:
            response = requests.post(GROQ_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            # Extract content from the chat completion structure
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"API Error: {str(e)}"

    return {
        "llm_call": llm_call,
        "retriever": retriever,
        "output_parser": output_parser
    }
