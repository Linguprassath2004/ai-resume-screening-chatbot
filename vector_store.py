# vector_store.py
class MyVectorStore:
    def __init__(self):
        # This will hold our processed resume strings
        self.documents = []

    def add_documents(self, new_docs):
        """Replaces dummy data with real uploaded text"""
        self.documents = new_docs

    def as_retriever(self):
        return self

    def get_relevant_documents(self, question: str):
        # In a simple version, we return all uploaded resumes for the AI to scan
        # If no resumes are uploaded, return the dummy reminder
        if not self.documents:
            return ["Document 1: Example content.", "Document 2: More example content."]
        
        return self.documents
