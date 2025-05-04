#!/bin/sh

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
while ! nc -z postgres 5432; do
  sleep 0.1
done
echo "PostgreSQL is ready!"

# Generate Prisma client
echo "Generating Prisma client..."
cd /app
prisma generate

# Push database schema
echo "Pushing database schema..."
prisma db push

# Start the application
echo "Starting the application..."
cd /app
python -c "
import asyncio
from prisma import Prisma
from main import app
import uvicorn

async def main():
    prisma = Prisma()
    await prisma.connect()
    config = uvicorn.Config(app, host='0.0.0.0', port=8000, reload=True, workers=1, loop='asyncio')
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == '__main__':
    asyncio.run(main())
" 