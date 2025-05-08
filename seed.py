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
        where={"id": "fruit-slicer"},
        include={"category": True}  # Include category to check current category
    )

    snake_game = await prisma.game.find_unique(
        where={"id": "snake"}
    )
    
    constructor_game = await prisma.game.find_unique(
        where={"id": "constructor"},
        include={"category": True}  # Include category to check current category
    )
    
    # Check if Rock Paper Scissors game exists
    rock_paper_scissors_game = await prisma.game.find_unique(
        where={"id": "rock-paper-scissors"}
    )

    # Create Rock Paper Scissors game if it doesn't exist
    if not rock_paper_scissors_game:
        print("Creating Rock Paper Scissors game...")
        await prisma.game.create(
            data={
                "id": "rock-paper-scissors",
                "name": "Rock Paper Scissors",
                "category": {
                    "connect": {
                        "id": motion_category.id  # Connect to Motion Games category
                    }
                }
            }
        )
    else:
        print("Rock Paper Scissors game already exists")
        
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
        print("Creating Balloon Pop game...")
        await prisma.game.create(
            data={
                "id": "bubble-pop",
                "name": "Balloon Pop",  # Using the new name
                "category": {
                    "connect": {
                        "id": motion_category.id
                    }
                }
            }
        )
    else:
        print("Bubble/Balloon Pop game already exists")
        # Check if the game needs name update
        if hasattr(bubble_pop_game, 'name') and bubble_pop_game.name == "Bubble Pop":
            print("Updating Bubble Pop game name to Balloon Pop...")
            await prisma.game.update(
                where={"id": "bubble-pop"},
                data={"name": "Balloon Pop"}  # Update the name
            )
        else:
            print("Game name already updated or couldn't determine current name")

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

    if not fruit_slicer_game:
        print("Creating Fruit Slicer game...")
        await prisma.game.create(
            data={
                "id": "fruit-slicer",
                "name": "Fruit Slicer",
                "category": {
                    "connect": {
                        "id": cognitive_category.id  # Setting to cognitive from the start
                    }
                }
            }
        )
    else:
        print("Fruit Slicer game already exists")
        # Check if the game needs name correction
        if fruit_slicer_game.name == "Fruit Slacer":
            print("Fixing Fruit Slicer game name...")
            await prisma.game.update(
                where={"id": "fruit-slicer"},
                data={"name": "Fruit Slicer"}  # Correct spelling
            )
        
        # Check if it needs category update (from motion to cognitive)
        if hasattr(fruit_slicer_game, 'category') and fruit_slicer_game.category and fruit_slicer_game.category.id == motion_category.id:
            print("Updating Fruit Slicer category from motion to cognitive...")
            await prisma.game.update(
                where={"id": "fruit-slicer"},
                data={
                    "category": {
                        "connect": {
                            "id": cognitive_category.id
                        }
                    }
                }
            )
        else:
            print("Fruit Slicer already has the correct category or couldn't determine current category")

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

    if not constructor_game:
        print("Creating Constructor game...")
        await prisma.game.create(
            data={
                "id": "constructor",
                "name": "Constructor",
                "category": {
                    "connect": {
                        "id": cognitive_category.id
                    }
                }
            }
        )
    else:               
        print("Constructor game already exists")
        # Check if it needs category update (from motion to cognitive)
        if hasattr(constructor_game, 'category') and constructor_game.category and constructor_game.category.id == motion_category.id:
            print("Updating Constructor category from motion to cognitive...")
            await prisma.game.update(
                where={"id": "constructor"},
                data={
                    "category": {
                        "connect": {
                            "id": cognitive_category.id
                        }
                    }
                }
            )
        else:
            print("Constructor already has the correct category or couldn't determine current category")
                      
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