from contextlib import asynccontextmanager
from src.prisma import prisma

async def init_db():
    categories = {"motor": None, "cognitive": None}
    
    for category_name in categories.keys():
        category = await prisma.category.find_first(where={"name": category_name})
        if not category:
            category = await prisma.category.create(data={"name": category_name})
        categories[category_name] = category.id
    
    default_games = [
        {"name": "Bubble pop", "category_id": categories["motor"], "image_url": "/bubble_pop.png"},
        {"name": "Tennis", "category_id": categories["motor"], "image_url": "/tennis.png"},
        {"name": "Draw it", "category_id": categories["cognitive"], "image_url": "/draw_it.png"},
        {"name": "Fruit slice", "category_id": categories["motor"], "image_url": "/friut_slice.png"},
    ]

    for game in default_games:
        existing_game = await prisma.game.find_first(where={"name": game["name"]})
        if not existing_game:
            await prisma.game.create(data=game)

@asynccontextmanager
async def lifespan(app):
    await prisma.connect()
    await init_db()  # Ensure default games exist
    yield
    await prisma.disconnect()
