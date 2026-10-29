from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()


def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


class FastEmbedEmbeddings:
    """Lightweight embeddings using fastembed (ONNX runtime, no PyTorch).
    Uses BAAI/bge-small-en-v1.5 which produces 384-dim vectors — compatible
    with the existing Pinecone index built on all-MiniLM-L6-v2 (also 384-dim).
    threads=1 ensures zero threadpool deadlocks on Linux/Render containers.
    RAM usage: ~60MB vs ~500MB for sentence-transformers + PyTorch.
    """

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name, threads=1)

    def embed_query(self, text):
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

    def embed_documents(self, texts):
        embeddings = list(self.model.embed(texts))
        return [e.tolist() for e in embeddings]


def download_hugging_face_embeddings():
    return FastEmbedEmbeddings()
