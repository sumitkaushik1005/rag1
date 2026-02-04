from typing import Iterable, List
from langchain_community.vectorstores import FAISS

class VectorStore:

    """
    FAISS-based vector store for storing and retrieving document embeddings.

    This class wraps a LangChain FAISS index. Use build() once with a list of
    text chunks to initialize the store, then call search() to retrieve the
    most similar chunks for a given query.
    """

    def __init__(self, embeddings):
        """
        Initialize the vector store.

        Parameters
        ----------
        embeddings : EmbeddingModel
            An embedding model wrapper that exposes `.model`, compatible with
            LangChain's FAISS.from_texts (e.g., HuggingFaceEmbeddings).
        """
        self.embeddings = embeddings
        self.store = None

    def build(self, texts: Iterable[str]) -> None:
        """Build the FAISS vector store from the provided text chunks.

        Parameters
        ----------
        texts : Iterable[str]
            An iterable of pre-chunked text strings used to populate the index.
        """
        texts = list(texts)
        if not texts:
            raise ValueError("No texts provided to build the vector store.")
        self.store = FAISS.from_texts(texts=texts, embedding=self.embeddings.model)

    def search(self, query: str, k: int = 3) -> List[str]:
        """Retrieve the top-k most similar documents to the query.

        Parameters
        ----------
        query : str
            The natural-language query to search for similar chunks.
        k : int, optional
            Number of most similar chunks to retrieve (default is 3).

        Returns
        -------
        List[str]
            A list of the page contents for the top-k similar documents.

        Raises
        ------
        ValueError
            If build() has not been called yet (no index), or if k < 1.
        """
        if self.store is None:
            raise ValueError("The vector store has not been built yet.")
        if k < 1:
            raise ValueError("k must be a positive integer.")

        docs = self.store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]