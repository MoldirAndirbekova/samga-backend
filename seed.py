from database import prisma
import asyncio
import uuid
import os
from datetime import datetime

async def seed_database():
    print("Checking database for existing seed data...")
    
    # Check if categories exist
    motion_category = await prisma.category.find_first(
        where={"name": "Motion Games"}
    )
    
    # If motion_category doesn't exist, create it
    if not motion_category:
        print("Creating Motion Games category...")
        motion_category = await prisma.category.create(
            data={
                "id": str(uuid.uuid4()),
                "name": "Motion Games"
            }
        )
    else:
        print("Motion Games category already exists")
    
    # Check if games exist
    ping_pong_game = await prisma.game.find_unique(
        where={"id": "ping-pong"}
    )
    
    bubble_pop_game = await prisma.game.find_unique(
        where={"id": "bubble-pop"}
    )

    letter_tracing_game = await prisma.game.find_unique(
        where={"id": "letter-tracing"}
    )
    
    # Create games if they don't exist - using the correct schema structure
    if not ping_pong_game:
        print("Creating Ping Pong game...")
        await prisma.game.create(
            data={
                "id": "ping-pong",
                "name": "Ping Pong",
                "category": {
                    "connect": {
                        "id": motion_category.id
                    }
                }
            }
        )
    else:
        print("Ping Pong game already exists")
    
    if not bubble_pop_game:
        print("Creating Bubble Pop game...")
        await prisma.game.create(
            data={
                "id": "bubble-pop",
                "name": "Bubble Pop",
                "category": {
                    "connect": {
                        "id": motion_category.id
                    }
                }
            }
        )
    else:
        print("Bubble Pop game already exists")

    if not letter_tracing_game:
        print("Creating Letter Tracing game...")
        await prisma.game.create(
            data={
                "id": "letter-tracing",
                "name": "Letter Tracing",
                "category": {
                    "connect": {
                        "id": motion_category.id
                    }
                }
            }
        )
    else:
        print("Letter Tracing game already exists")

    print("Database seeding complete")

if __name__ == "__main__":
    asyncio.run(seed_database()) 