from langchain_community.vectorstores import FAISS

class VectorStore:

    """
    FAISS-based vector store for storing and retrieving document embeddings.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = None

    def build(self, texts):
        """Build the FAISS vector store from the provided texts."""
        self.store = FAISS.from_texts(texts=texts, embedding=self.embeddings.model)


    def search(self, query:str, k:int=3):
        """
        Docstring for search
        
        :param self: Description
        :param query: Description
        :type query: str
        :param k: Description
        :type k: int

        Retrieves the top-k most similar documents to the query from the vector store.
        """
        if self.store is None:
            raise ValueError("The vector store has not been built yet.")    
        
        docs=self.store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]