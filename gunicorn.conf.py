bind = "0.0.0.0:10000"
workers = 1  # Reduce from 2 to 1
timeout = 120
keepalive = 5
max_requests = 100  # Reduce from 1000
max_requests_jitter = 10  # Reduce from 100
worker_class = "sync"
worker_connections = 1000
preload_app = True  # Share memory between workers
