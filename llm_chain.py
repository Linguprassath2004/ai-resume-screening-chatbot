import os
import requests
import streamlit as st
from langchain_core.prompts import PromptTemplate
from vector_store import MyVectorStore

# Groq API Configuration
# Note: Ensure this URL and Model name are correct for your current Groq tier
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

class SimpleStrOutputParser:
    def parse(self, text: str) -> str:
        return text.strip() if text else "No response generated."

def create_qa_chain(vector_store: MyVectorStore):
    retriever = vector_store.as_retriever()

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
            return "Error: GROQ_API_KEY not found in Streamlit Secrets."

        # 1. Sanitize the Context
        # Join docs and remove characters that often break JSON payloads (like specific quotes or \u00a0)
        context_str = "\n".join([str(doc) for doc in context_docs])
        context_str = context_str.replace('"', "'").replace('\xa0', ' ').strip()
        
        # 2. Limit Context Length (Safety valve for 400 Errors)
        # If the text is too long, we take the first 10,000 characters to stay within token limits
        if len(context_str) > 10000:
            context_str = context_str[:10000] + "... [Truncated for Token Limits]"

        formatted_prompt = prompt_template.format(question=question, context=context_str)

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY.strip()}",
            "Content-Type": "application/json"
        }
        
        # 3. Define Payload
        # Added max_tokens and changed model to a standard stable version
        payload = {
            "model": "llama-3.3-70b-versatile", 
            "messages": [
                {"role": "system", "content": "You are an HR assistant specialized in resume screening."},
                {"role": "user", "content": formatted_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "top_p": 1
        }

        try:
            # Use json=payload to let 'requests' handle the JSON encoding/escaping properly
            response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
            
            # 4. Detailed Error Catching
            if response.status_code != 200:
                # This will print the EXACT error from Groq (e.g., 'invalid_api_key' or 'context_length_exceeded')
                return f"Groq API Error {response.status_code}: {response.text}"
                
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.Timeout:
            return "Error: The request to Groq timed out."
        except Exception as e:
            return f"System Error: {str(e)}"

    return {
        "llm_call": llm_call,
        "retriever": retriever,
        "output_parser": output_parser
    }
