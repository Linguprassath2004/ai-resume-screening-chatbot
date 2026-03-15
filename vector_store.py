# vector_store.py
class MyVectorStore:
    """Dummy vector store for demonstration. Replace with FAISS, Pinecone, etc."""

    def as_retriever(self):
        return self

    def get_relevant_documents(self, question: str):
        # In real use, return the top documents related to the question
        return ["Document 1: Example content.", "Document 2: More example content."]
