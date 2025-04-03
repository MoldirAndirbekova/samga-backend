from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.prisma import prisma
from src.utils.auth import JWTBearer, decodeJWT
from fastapi import HTTPException

router = APIRouter()

class Game(BaseModel):
    id: str
    name: str
    image_url: str
    category_id: str

@router.get("/categories", tags=["game"])
async def get_categories():
    categories = await prisma.category.find_many()
    return categories

@router.get("/categories/{category_id}/games", tags=["game"], dependencies=[Depends(JWTBearer())])
async def get_games_by_category(category_id: str, token: str = Depends(JWTBearer())):
    decoded = decodeJWT(token)

    if not decoded or "userId" not in decoded:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    games = await prisma.game.find_many(
        where={"category_id": category_id},
    )
    if games:
        return games
    
    return HTTPException(status_code=404, detail="Games not found")

@router.get("/games", tags=["game"], dependencies=[Depends(JWTBearer())])
async def get_game(token: str = Depends(JWTBearer())):
    decoded = decodeJWT(token)

    if not decoded or "userId" not in decoded:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    game = await prisma.game.find_many()
    return game

@router.get("/games/{game_id}", tags=["game"], dependencies=[Depends(JWTBearer())])
async def get_game_by_id(game_id: str, token: str = Depends(JWTBearer()), response_model=Game):
    decoded = decodeJWT(token)

    if not decoded or "userId" not in decoded:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    game = await prisma.game.find_first(
        where={"id": game_id},
    )
    if game:
        return game
    
    return HTTPException(status_code=404, detail="Game not found")
