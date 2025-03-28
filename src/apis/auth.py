import datetime
from fastapi import APIRouter, HTTPException, Depends
from prisma.models import User
from pydantic import BaseModel
from src.prisma import prisma
from src.utils.auth import (
    encryptPassword,
    signJWT,
    validatePassword,
    decodeJWT,
)

router = APIRouter()

class SignIn(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    user: User

@router.post("/auth/sign-in", tags=["auth"])
async def sign_in(signIn: SignIn):
    user = await prisma.user.find_first(where={"email": signIn.email})

    if not user or not validatePassword(signIn.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    del user.password

    return TokenResponse(
        access_token=signJWT(user.id),
        refresh_token=signJWT(user.id, is_refresh=True),
        user=user
    )

class SignUp(BaseModel):
    email: str
    password: str
    full_name: str

@router.post("/auth/sign-up", tags=["auth"])
async def sign_up(user: SignUp):
    encrypted_password = encryptPassword(user.password)

    created_user = await prisma.user.create(
        {
            "email": user.email,
            "password": encrypted_password,
            "full_name": user.full_name
        }
    )

    return created_user

class RefreshRequest(BaseModel):
    refresh_token: str

@router.post("/auth/refresh", tags=["auth"])
async def refresh_token(request: RefreshRequest):
    decoded = decodeJWT(request.refresh_token, is_refresh=True)

    if not decoded:
        raise HTTPException(status_code=403, detail="Invalid or expired refresh token")

    new_access_token = signJWT(decoded["userId"])
    return {"access_token": new_access_token}

@router.get("/auth/", tags=["auth"])
async def auth():
    users = await prisma.user.find_many()

    for user in users:
        del user.password

    return users
