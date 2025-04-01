from fastapi import APIRouter, Depends
from src.prisma import prisma
from src.utils.auth import JWTBearer, decodeJWT
from fastapi import HTTPException

router = APIRouter()

@router.get("/categories", tags=["game"])
async def get_categories():
    categories = await prisma.category.find_many(include={"games": True})
    return categories

@router.get("/categories/{category_id}/games", tags=["game"], dependencies=[Depends(JWTBearer())])
async def get_games_by_category(category_id: str, token: str = Depends(JWTBearer())):
    decoded = decodeJWT(token)

    if not decoded or "userId" not in decoded:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    games = await prisma.game.find_many(
        where={"category_id": category_id},
    )
    
    return games

@router.get("/games", tags=["game"], dependencies=[Depends(JWTBearer())])
async def get_game(token: str = Depends(JWTBearer())):
    decoded = decodeJWT(token)

    if not decoded or "userId" not in decoded:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    game = await prisma.game.find_many()
    return game

