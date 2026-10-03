from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import numpy as np


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


class HuggingFaceRemoteEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.client = InferenceClient(
            api_key=os.getenv("HF_TOKEN")
        )
        self.model_name = model_name

    def embed_query(self, text):
        result = self.client.feature_extraction(
            text,
            model=self.model_name
        )

        return np.asarray(result).reshape(-1).tolist()


def download_hugging_face_embeddings():
    return HuggingFaceRemoteEmbeddings()