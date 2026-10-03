from src.helper import (load_pdf_file,text_split,download_hugging_face_embeddings)
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()



PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()




pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

print("Pinecone index ready")





pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("medicalbot")

vectors = []

for i, chunk in enumerate(text_chunks):
    vector = embeddings.embed_query(chunk.page_content)

    vectors.append({
        "id": f"doc-{i}",
        "values": vector,
        "metadata": {
            "text": chunk.page_content
        }
    })

batch_size = 100

for i in range(0, len(vectors), batch_size):
    index.upsert(vectors=vectors[i:i + batch_size])

print(f"Uploaded {len(vectors)} vectors to Pinecone")