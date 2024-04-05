#!/bin/sh
# entrypoint.sh

echo "Generating new Alembic revision (if necessary)..."
alembic revision --autogenerate -m "Database creation"

echo "Applying Alembic migrations..."
alembic upgrade head

echo "Starting FastAPI application..."
exec "$@"