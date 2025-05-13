#!/bin/sh

echo "Starting application on Railway..."

# Generate Prisma client
echo "Generating Prisma client..."
prisma generate

# Push database schema
echo "Pushing database schema..."
prisma db push

# Start the application with Python
echo "Starting the application..."
python -c "
import asyncio
import os
from prisma import Prisma
from main import app
import uvicorn

async def main():
    # Prisma will use DATABASE_URL from environment
    prisma = Prisma()
    await prisma.connect()
    print('Connected to database')
    
    port = int(os.getenv('PORT', 8000))
    config = uvicorn.Config(
        app, 
        host='0.0.0.0', 
        port=port, 
        workers=1, 
        loop='asyncio'
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == '__main__':
    asyncio.run(main())
"