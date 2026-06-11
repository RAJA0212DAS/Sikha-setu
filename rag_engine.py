from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from config import CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME

class RAGEngine:
    def __init__(self):
        self.db = None
        self.llm = Ollama(model="mistral")
        self.embeddings = HuggingFaceEmbeddings()

    def build_index(self, documents):
        splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = splitter.split_documents(documents)

        self.db = FAISS.from_documents(chunks, self.embeddings)

    def query(self, query):
        if not self.db:
            return "No documents loaded."

        docs = self.db.similarity_search(query, k=2)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are a strict academic assistant.

Rules:
- Answer ONLY from the given context
- If answer not found, say "Not found in document"
- Be clear and structured

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)
        return response