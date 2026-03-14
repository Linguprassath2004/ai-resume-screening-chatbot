from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_qa_chain(vector_store):

    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0
    )

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
You are an AI resume screening assistant.

Answer the question using ONLY the provided resume context.

Resume Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain