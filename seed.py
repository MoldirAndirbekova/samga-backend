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
    cognitive_category = await prisma.category.find_first(
        where={"name": "Cognitive Games"}
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

    if not cognitive_category:
        print("Creating Cognitive Games category...")
        cognitive_category = await prisma.category.create(
            data={
                "id": str(uuid.uuid4()),
                "name": "Cognitive Games"
            }
        )
    else:
        print("Cognitive Games category already exists")
    
    # Check if games exist
    ping_pong_game = await prisma.game.find_unique(
        where={"id": "ping-pong"}
    )
    
    bubble_pop_game = await prisma.game.find_unique(
        where={"id": "bubble-pop"}
    )

    letter_tracing_game = await prisma.game.find_unique(
        where={"id": "letter-tracing"},
        include={"category": True}
    )

    fruit_slicer_game = await prisma.game.find_unique(
        where={"id": "fruit-slicer"}
    )

    snake_game = await prisma.game.find_unique(
        where={"id": "snake"}
    )

    flappy_bird_game = await prisma.game.find_unique(
        where={"id": "flappy-bird"}
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
    
    if letter_tracing_game and letter_tracing_game.category.id != cognitive_category.id:
        print("Updating letter tracing category to cognitive...")
        await prisma.game.update(
            where={"id": "letter-tracing"},
            data={
                "category": {
                    "connect": {
                        "id": cognitive_category.id
                    }
                }
            }
        )
    else:
        print("Category is already cognitive. No update needed.")

    if fruit_slicer_game and fruit_slicer_game.name == "Fruit Slacer":
        print("Fixing Fruit Slicer game name...")
        await prisma.game.update(
            where={"id": "fruit-slicer"},
            data={"name": "Fruit Slicer"}  # Correct spelling
        )

    if not fruit_slicer_game:
        print("Creating Fruit Slicer game...")
        await prisma.game.create(
            data={
                "id": "fruit-slicer",
                "name": "Fruit Slicer",
                "category": {
                    "connect": {
                        "id": motion_category.id
                    }
                }
            }
        )
    else:
        print("Fruit Slicer game already exists")

    if not snake_game:
        print("Creating Snake game...")
        await prisma.game.create(
            data={
                "id": "snake",
                "name": "Snake",
                "category": {
                    "connect": {
                        "id": motion_category.id
                    }
                }
            }
        )
    else:
        print("Snake game already exists")

    if not flappy_bird_game:
        print("Creating Flappy bird game")
        await prisma.game.create(
            data={
                "id": "flappy-bird",
                "name": "Flappy Bird",
                "category": {
                    "connect": {
                        "id": motion_category.id
                    }
                }
            }
        )
    else:
        print("Flappy Bird game already exists")

    print("Database seeding complete")

if __name__ == "__main__":
    asyncio.run(seed_database()) 