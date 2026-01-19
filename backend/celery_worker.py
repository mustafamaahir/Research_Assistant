#!/usr/bin/env python
"""
Celery worker entry point for background task processing
"""
from main import celery_app

if __name__ == '__main__':
    celery_app.worker_main([
        'worker',
        '--loglevel=info',
        '--concurrency=4',  # Number of concurrent tasks
        '--max-tasks-per-child=100',  # Restart worker after 100 tasks (prevents memory leaks)
    ])