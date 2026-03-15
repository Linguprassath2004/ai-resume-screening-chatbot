# app.py
import streamlit as st
from llm_chain import create_qa_chain
from some_vector_store_module import MyVectorStore  # Replace with your actual vector store

# Initialize vector store
vector_store = MyVectorStore()

# Create the QA chain
qa_chain = create_qa_chain(vector_store)

st.title("AI Resume Screening Chatbot")
st.write("Ask questions about the resumes and get answers powered by GROQ LLM.")

# User input
question = st.text_input("Ask a question:")

if question:
    # Retrieve relevant documents from vector store
    docs = qa_chain["retriever"].get_relevant_documents(question)
    
    if not docs:
        st.warning("No relevant documents found.")
    else:
        # Combine all retrieved docs into a single context string
        context_text = "\n".join([d.page_content for d in docs])

        # Format the prompt with question and context
        formatted_prompt = qa_chain["prompt"].format(question=question, context=context_text)

        # Generate response from GROQ LLM
        response = qa_chain["llm"].generate(formatted_prompt)

        # Parse the response
        final_answer = qa_chain["output_parser"].parse(response)

        # Display the answer
        st.success(final_answer)
