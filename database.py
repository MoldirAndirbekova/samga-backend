from contextlib import asynccontextmanager
from src.prisma import prisma
from prisma.enums import Category

async def init_db():
    default_games = [
        {"name": "Bubble pop", "category": Category.motor},
        {"name": "Tennis", "category": Category.motor},
        {"name": "Draw it", "category": Category.cognitive},
        {"name": "Fruit slice", "category": Category.motor},
    ]

    for game in default_games:
        existing_game = await prisma.game.find_first(
            where={"name": {"equals": game["name"], "mode": "insensitive"}}
        )
        if not existing_game:
            await prisma.game.create(data=game)

@asynccontextmanager
async def lifespan(app):
    await prisma.connect()
    await init_db()  # Ensure default games exist
    yield
    await prisma.disconnect()
