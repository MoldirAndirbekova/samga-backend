from fastapi import APIRouter

from src.apis.auth import router as authRouter
from src.apis.game import router as gameRouter
from src.apis.users import router as usersRouter
from src.apis.feedback import router as feedbackRouter

apis = APIRouter()

apis.include_router(authRouter)
apis.include_router(gameRouter)
apis.include_router(usersRouter)
apis.include_router(feedbackRouter)

__all__ = ["apis"]