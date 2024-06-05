#!/bin/bash

# Load environment variables
source .env-non-dev

# Start PostgreSQL
echo "Starting PostgreSQL..."
sudo service postgresql start

# Start Redis
echo "Starting Redis..."
sudo service redis-server start

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Run PostgreSQL command (if needed to set custom port)
sudo -u postgres psql -c "ALTER SYSTEM SET port = 1221;"
sudo service postgresql restart

# Check if PostgreSQL is running
echo "Checking if PostgreSQL is running..."
docker/postgres.sh &

# Run Alembic migrations
echo "Running Alembic migrations..."
docker/migration.sh &

# Run Redis command (if needed to set custom port)
redis-server --port 5370 &

# Run FastAPI application
echo "Starting FastAPI application..."
docker/app.sh &

# Run Celery worker
echo "Starting Celery worker..."
docker/celery.sh celery &

# Run Flower
echo "Starting Flower..."
docker/celery.sh flower --port=5555 &

# Wait for background processes to finish
wait
