import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

port = os.environ.get("PORT", "8080")
bind = f"0.0.0.0:{port}"

workers = 1
threads = 1  # Single thread prevents ONNX runtime deadlocks on virtualized Linux containers
timeout = 180
keepalive = 5
graceful_timeout = 30
