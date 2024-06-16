#!/bin/bash

alembic revision --autogenerate -m "Database creation"
alembic upgrade head

# add role = 1

#DB_HOST="db"
DB_HOST="localhost"
#DB_PORT="1221"
DB_PORT="5432"
DB_NAME="postgres"
DB_USER="postgres"
# DB_PASS="postgres"
DB_PASS="GWtfrb12!"

PGPASSWORD=$DB_PASS psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "INSERT INTO role (id, column1, column2) VALUES (1, 'gg', 'gg') ON CONFLICT (id) DO NOTHING;"

#cd src

gunicorn scripts/main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000