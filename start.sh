#!/bin/sh
echo "Starting application..."

# Generate Prisma client
echo "Generating Prisma client..."
prisma generate

# Push database schema
echo "Pushing database schema..."
prisma db push

# Use the PORT env var if set, otherwise use 8000
PORT_TO_USE=${PORT:-8000}
echo "Starting application on port ${PORT_TO_USE}..."

# Check if we're in development mode
if [ "$ENVIRONMENT" = "development" ]; then
    echo "Running in DEVELOPMENT mode with hot reload..."
    exec uvicorn main:app \
        --host 0.0.0.0 \
        --port ${PORT_TO_USE} \
        --reload \
        --reload-dir /app \
        --workers 1
else
    echo "Running in PRODUCTION mode..."
    exec uvicorn main:app \
        --host 0.0.0.0 \
        --port ${PORT_TO_USE} \
        --workers 1
fi