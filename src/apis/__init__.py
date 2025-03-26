from fastapi import APIRouter

from src.apis.auth import router as authRouter
from src.apis.game import router as gameRouter

apis = APIRouter()
apis.include_router(authRouter)
apis.include_router(gameRouter)

__all__ = ["apis"]