set -e

DB_HOST="db"
#DB_PORT="1221"
DB_PORT="5432"
DB_NAME="postgres"
DB_USER="postgres"
# DB_PASS="postgres"
DB_PASS="GWtfrb12!"

echo "Waiting for PostgreSQL to become available..."

# Loop until we can connect to the PostgreSQL server or we reach a timeout
timeout=60  # Timeout after 60 seconds
while ! PGPASSWORD=$DB_PASS psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c '\q' > /dev/null 2>&1
do
  timeout=$(($timeout - 1))
  if [ $timeout -le 0 ]; then
    echo "Timed out waiting for PostgreSQL to become available"
    exit 1
  fi
  echo "PostgreSQL is unavailable - retrying in 1 second..."
  sleep 1
done

echo "PostgreSQL is up - executing command"
exec "$@"
