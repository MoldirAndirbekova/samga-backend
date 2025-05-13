#!/bin/sh

echo "Starting application on Railway..."

# Generate Prisma client
echo "Generating Prisma client..."
prisma generate

# Push database schema
echo "Pushing database schema..."
prisma db push

# Start with uvicorn using Railway's PORT
echo "Starting the application on port ${PORT}..."
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1