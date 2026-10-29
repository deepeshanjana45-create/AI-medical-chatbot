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

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if HF_TOKEN:
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


import requests

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="featherless-ai"
)


def generate_response(prompt_value):
    system_text = ""
    user_text = ""
    for msg in prompt_value.to_messages():
        if msg.type == "system":
            system_text += msg.content + "\n"
        else:
            user_text += msg.content + "\n"

    # 1. Use clean conversational Gemini model
    if GEMINI_API_KEY:
        models_to_try = [
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
            "gemini-pro-latest"
        ]
        for model_name in models_to_try:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
                payload = {
                    "system_instruction": {
                        "parts": [{"text": system_text.strip()}]
                    },
                    "contents": [
                        {"parts": [{"text": user_text.strip()}]}
                    ],
                    "generationConfig": {
                        "maxOutputTokens": 1000,
                        "temperature": 0.3
                    }
                }
                res = requests.post(url, json=payload, timeout=20)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"].strip()
            except Exception as err:
                print(f"Error calling {model_name}:", err)

    # 2. Fallback to Hugging Face if configured
    if HF_TOKEN:
        try:
            combined = f"System: {system_text}\nUser: {user_text}"
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-1.5B-Instruct",
                messages=[{"role": "user", "content": combined}],
                max_tokens=600,
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as err:
            print("Hugging Face error:", err)

    return "Sorry, I could not generate a response. Please try again."


llm = RunnableLambda(generate_response)


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
    try:
        msg = request.form["msg"]
        response = rag_chain.invoke({
            "input": msg
        })
        answer = str(response["answer"])
        try:
            print("Response:", answer[:150].encode("ascii", errors="replace").decode("ascii"))
        except Exception:
            pass
        return answer
    except Exception as e:
        print("Error during chat processing:", e)
        return "I apologize, but an error occurred while processing your request. Please try again."


# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )