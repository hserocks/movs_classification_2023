#!/bin/sh

# stop api processes manually
echo "Stopping API processes..."
pkill -f "main:app"
pkill -f "uvicorn.workers.UvicornWorker"
pkill redis
pkill celery
pkill flower
pkill gunicorn
echo "API processes stopped."
