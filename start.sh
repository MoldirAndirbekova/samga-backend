#!/bin/sh

echo "Starting application on Railway..."

# Generate Prisma client
echo "Generating Prisma client..."
prisma generate

# Push database schema
echo "Pushing database schema..."
prisma db push

# Use the PORT env var if set, otherwise use 8000 (Railway's expected port)
PORT_TO_USE=${PORT:-8000}
echo "Starting application on port ${PORT_TO_USE}..."

# Start the application
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT_TO_USE} \
    --workers 1