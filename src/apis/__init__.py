from fastapi import APIRouter

from src.apis.auth import router as authRouter

apis = APIRouter()
apis.include_router(authRouter)

__all__ = ["apis"]