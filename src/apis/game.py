from fastapi import APIRouter
from src.prisma import prisma

router = APIRouter()

@router.get("/categories", tags=["game"])
async def get_categories():
    categories = await prisma.category.find_many(include={"games": True})
    return categories

@router.get("/categories/{category_id}/games", tags=["game"])
async def get_games_by_category(category_id: str):
    games = await prisma.game.find_many(
        where={"category_id": category_id},
    )
    return games

