# vector_store.py

class MyVectorStore:
    def __init__(self):
        # Start completely empty
        self.documents = []

    def add_documents(self, text_list: list):
        """Adds real resume text to the list"""
        self.documents.extend(text_list)

    def as_retriever(self):
        return self

    def get_relevant_documents(self, question: str):
        """Returns the uploaded resumes or an empty list if none exist."""
        return self.documents
