from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
import os
import requests
import threading
import re

from src.prompt import *

load_dotenv()

app = Flask(__name__)

# Persistent HTTP session for fast connection reuse
http_session = requests.Session()


# -----------------------------
# API KEYS
# -----------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_APT_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# -----------------------------
# EMBEDDINGS & PINECONE (BACKGROUND WARM-UP)
# -----------------------------

index_name = "medicalbot"
_embeddings = None
_pinecone_index = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = download_hugging_face_embeddings()
    return _embeddings

def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        key = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_APT_KEY")
        pc = Pinecone(api_key=key)
        _pinecone_index = pc.Index(index_name)
    return _pinecone_index

# Pre-warm embeddings & index in a non-blocking background thread
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
        top_k=2,
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

    # 1. Groq (Ultra-Fast ~0.8s) if key configured
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        groq_models = [
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b"
        ]
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        for model in groq_models:
            try:
                groq_payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_text.strip()},
                        {"role": "user", "content": user_text.strip()}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.2
                }
                res = http_session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=groq_payload,
                    timeout=10
                )
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0]:
                        return choices[0]["message"].get("content", "").strip()
            except Exception as err:
                print(f"Groq {model} error:", err)

    # 2. Google Gemini fallback
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        models_to_try = [
            "gemini-flash-lite-latest",
            "gemini-flash-latest"
        ]
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        payload = {
            "system_instruction": {"parts": [{"text": system_text.strip()}]},
            "contents": [{"parts": [{"text": user_text.strip()}]}],
            "safetySettings": safety_settings,
            "generationConfig": {"maxOutputTokens": 300, "temperature": 0.2}
        }
        for model_name in models_to_try:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
                res = http_session.post(url, json=payload, timeout=25)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"].strip()
            except Exception as err:
                print(f"Gemini {model_name} error:", err)

    return "I am currently experiencing high traffic. Please check that GROQ_API_KEY or GEMINI_API_KEY is configured in your Render Environment settings."


llm = RunnableLambda(generate_response)


# -----------------------------
# FORMAT DOCUMENTS
# -----------------------------

def format_docs(result):
    return "\n\n".join(
        match["metadata"]["text"]
        for match in result.get("matches", [])
        if "metadata" in match and "text" in match["metadata"]
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
# GREETING PATTERNS (INSTANT 0.005s RESPONSE)
# -----------------------------

GREETING_WORDS = {
    "hi", "hii", "hiii", "hello", "helloo", "hey", "heyy", "heya",
    "good morning", "good evening", "good afternoon", "greetings", "help"
}

def is_greeting(text):
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    return cleaned in GREETING_WORDS


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

        if is_greeting(msg):
            return "Hello! How can I help you with your health today?"

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
