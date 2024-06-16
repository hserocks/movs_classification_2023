#!/bin/sh

# Start bot
nohup python scripts/bot_new.py > /dev/null 2>&1 &

# Start API app
exec gunicorn scripts.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000
# exec gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:9999
