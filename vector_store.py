# vector_store.py

class MyVectorStore:
    def __init__(self):
        self.documents = []

    def add_documents(self, text_list: list):
        self.documents.extend(text_list)

    def as_retriever(self):
        return self

    def get_relevant_documents(self, question: str):
        if not self.documents:
            return []

        # 1. Simple Keyword Filtering to prevent 400 errors (Context Overflow)
        # We split the question into words and check which resumes contain them
        keywords = question.lower().split()
        relevant = []
        
        for doc in self.documents:
            # If a resume contains any of the main words from the question, include it
            if any(word in doc.lower() for word in keywords if len(word) > 3):
                relevant.append(doc)
        
        # 2. Safety Valve: If nothing matched, just give the first 2 resumes 
        # to stay within token limits
        if not relevant:
            return self.documents[:2]
            
        # 3. Final Check: Only return the top 3 matches to keep the prompt small
        return relevant[:3]
