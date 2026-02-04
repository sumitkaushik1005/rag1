from app.core.config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME, TOP_K
from app.rag.loader import PDFLoader
from app.rag.chunker import LangchainTextChunker
from app.rag.embeddings import EmbeddingModel
from app.rag.vectorstore import VectorStore

from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv


class RAGEngine:
    """Singleton-style RAG engine to manage the components of the RAG system."""

    def __init__(self):
        self.vector_store = None
        self._initialize()

    def _initialize(self):
        load_dotenv()

        text= PDFLoader(PDF_PATH).load()
        chunks= LangchainTextChunker(CHUNK_SIZE, CHUNK_OVERLAP).chunk(text)
        embeddings=EmbeddingModel(EMBEDDING_MODEL_NAME)
        self.vector_store=VectorStore(embeddings)
        self.vector_store.build(chunks)

        self.llm= ChatGroq(model_name="llama-3.3-70b-versatile")

    def generate_answer(self, question: str):
        print("Engine.py is being executed")
        """Generate an answer using the vectore store with a grounded search
        Retrieve top k chuns and pass them to llm to generate answer"""

        contexts=self.vector_store.search(question, k=TOP_K)
        combined_texts= "\n\n".join(contexts)

        prompt_template= """Use the below context to answer the question.
        You are a helpful AI assistant. Use only the inofrmation provided in the context below to answer the question. If the answer is not present in the context, response with I'don't know.
        context: {combined_texts}
        question: {question}

        answer:
        """

        agent=create_agent(
            model=self.llm,
            system_prompt="You are a helpful AI assistant."
        )

        result=agent.invoke({
            "messages":[{
                "role":"user",
                "content": prompt_template.format(combined_texts=combined_texts, question=question)
            }]
        })
        print(result)
        return result['messages'][-1].content