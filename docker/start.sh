#!/bin/sh

# Start bot
nohup python bot_new.py > /dev/null 2>&1 &

# Start API app
exec gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000
