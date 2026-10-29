import os

# Gunicorn configuration file
# Loaded automatically by Gunicorn even if called as `gunicorn app:app`

port = os.environ.get("PORT", "8080")
bind = f"0.0.0.0:{port}"

workers = 1
threads = 4
timeout = 180  # 3 minutes: prevents premature worker termination
keepalive = 5
graceful_timeout = 30
