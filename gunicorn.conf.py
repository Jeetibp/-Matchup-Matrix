bind = "0.0.0.0:10000"
workers = 4  # 4 workers for ~20 concurrent users (HF free: 16GB RAM)
threads = 2  # 2 threads per worker = 8 total request handlers
timeout = 120
keepalive = 5
max_requests = 500
max_requests_jitter = 50
worker_class = "sync"
worker_connections = 1000
preload_app = True  # Share memory between workers (saves RAM)
