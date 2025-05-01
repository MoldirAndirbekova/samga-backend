from prisma import Prisma
from prisma.models import User, Child, Game, Category
from contextlib import asynccontextmanager

# Create a single Prisma instance to be used throughout the application
prisma = Prisma()

async def connect():
    if not prisma.is_connected():
        await prisma.connect()

async def disconnect():
    if prisma.is_connected():
        await prisma.disconnect()

@asynccontextmanager
async def get_db():
    try:
        await connect()
        yield prisma
    finally:
        await disconnect() 