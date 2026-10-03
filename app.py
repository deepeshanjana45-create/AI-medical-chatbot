from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
import os

from src.prompt import *

load_dotenv()

app = Flask(__name__)


# -----------------------------
# API KEYS
# -----------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN


# -----------------------------
# EMBEDDINGS
# -----------------------------

embeddings = download_hugging_face_embeddings()


# -----------------------------
# PINECONE
# -----------------------------

index_name = "medicalbot"

pc = Pinecone(api_key=PINECONE_API_KEY)

pinecone_index = pc.Index(index_name)


# -----------------------------
# RETRIEVER
# -----------------------------

retriever = RunnableLambda(
    lambda query: pinecone_index.query(
        vector=embeddings.embed_query(query),
        top_k=3,
        include_metadata=True
    )
)


# -----------------------------
# HUGGING FACE
# -----------------------------

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="featherless-ai"
)


def hf_generate(prompt_value):

    messages = []

    for msg in prompt_value.to_messages():

        role = "system" if msg.type == "system" else "user"

        messages.append({
            "role": role,
            "content": msg.content
        })

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=messages,
        max_tokens=200,
        temperature=0.4
    )

    return response.choices[0].message.content


llm = RunnableLambda(hf_generate)


# -----------------------------
# FORMAT DOCUMENTS
# -----------------------------

def format_docs(result):

    return "\n\n".join(
        match["metadata"]["text"]
        for match in result["matches"]
    )


# -----------------------------
# RAG CHAIN
# -----------------------------

rag_chain = (
    {
        "context": RunnableLambda(
            lambda x: retriever.invoke(x["input"])
        ) | format_docs,

        "input": RunnableLambda(
            lambda x: x["input"]
        )
    }
    | prompt
    | llm
    | RunnableLambda(
        lambda x: {"answer": x}
    )
)


# -----------------------------
# HOME PAGE
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# CHAT ROUTE
# -----------------------------

@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form["msg"]

    response = rag_chain.invoke({
        "input": msg
    })

    print("Response:", response["answer"])

    return str(response["answer"])


# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )