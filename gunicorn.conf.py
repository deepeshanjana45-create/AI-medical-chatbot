import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/fastembed_cache"

port = os.environ.get("PORT", "8080")
bind = f"0.0.0.0:{port}"

workers = 1
threads = 1   # Single thread prevents ONNX runtime deadlocks on Render/Linux containers
timeout = 300  # 5 minutes — covers cold start model download + first inference
keepalive = 5
graceful_timeout = 30
