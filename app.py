from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
import os
import requests
import threading

from src.prompt import *

load_dotenv()

app = Flask(__name__)


# -----------------------------
# API KEYS (handles typo PINECONE_APT_KEY as fallback)
# -----------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_APT_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


# -----------------------------
# EMBEDDINGS & PINECONE (THREAD-SAFE LAZY LOAD + BACKGROUND PRE-WARM)
# -----------------------------

index_name = "medicalbot"
_embeddings = None
_pinecone_index = None
_lock = threading.Lock()

def get_embeddings():
    global _embeddings
    with _lock:
        if _embeddings is None:
            _embeddings = download_hugging_face_embeddings()
    return _embeddings

def get_pinecone_index():
    global _pinecone_index
    with _lock:
        if _pinecone_index is None:
            key = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_APT_KEY")
            pc = Pinecone(api_key=key)
            _pinecone_index = pc.Index(index_name)
    return _pinecone_index

# Pre-warm in background so first request is instant
def _prewarm():
    try:
        get_embeddings()
        get_pinecone_index()
    except Exception as e:
        print("Pre-warm background error:", e)

threading.Thread(target=_prewarm, daemon=True).start()


# -----------------------------
# RETRIEVER
# -----------------------------

retriever = RunnableLambda(
    lambda query: get_pinecone_index().query(
        vector=get_embeddings().embed_query(query),
        top_k=3,
        include_metadata=True
    )
)


def generate_response(prompt_value):
    system_text = ""
    user_text = ""
    for msg in prompt_value.to_messages():
        if msg.type == "system":
            system_text += msg.content + "\n"
        else:
            user_text += msg.content + "\n"

    # 1. Use Gemini (primary)
    api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if api_key:
        models_to_try = [
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
            "gemini-pro-latest"
        ]
        for model_name in models_to_try:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
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
                else:
                    print(f"Gemini API {model_name} returned status {res.status_code}: {res.text[:100]}")
            except Exception as err:
                print(f"Error calling {model_name}:", err)

    # 2. Fallback to Hugging Face if configured
    hf_tok = os.getenv("HF_TOKEN") or HF_TOKEN
    if hf_tok:
        try:
            from huggingface_hub import InferenceClient
            hf_client = InferenceClient(api_key=hf_tok, provider="featherless-ai")
            combined = f"System: {system_text}\nUser: {user_text}"
            response = hf_client.chat.completions.create(
                model="Qwen/Qwen2.5-1.5B-Instruct",
                messages=[{"role": "user", "content": combined}],
                max_tokens=600,
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as err:
            print("Hugging Face error:", err)

    return "I am currently unable to reach the medical language model. Please verify your GEMINI_API_KEY."


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
        msg = request.form.get("msg") or (request.get_json(silent=True) or {}).get("msg", "")
        if not msg:
            return "Please enter a question."
        response = rag_chain.invoke({
            "input": msg
        })
        answer = str(response["answer"])
        return answer
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print("Error during chat processing:\n", err_msg)
        return "I apologize, but an error occurred while processing your request. Please check the server logs."


# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
