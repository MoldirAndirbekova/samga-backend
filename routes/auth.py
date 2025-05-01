from datetime import datetime, timedelta
from typing import Optional
import random
import secrets
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from database import prisma

router = APIRouter()

# Random names for children
CHILD_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "Lucas", "Mia", "Jackson", "Amelia", "Aiden", "Harper",
    "Elijah", "Evelyn", "Grayson", "Abigail", "Benjamin", "Emily", "Carter",
    "Elizabeth", "Michael", "Mila", "Sebastian", "Ella", "James", "Scarlett",
    "Alexander", "Victoria", "William", "Madison", "Daniel", "Luna", "Matthew",
    "Grace", "Henry", "Chloe", "Joseph", "Penelope", "Samuel", "Layla", "David",
    "Riley", "Wyatt", "Zoey", "John", "Nora", "Owen", "Lily", "Dylan", "Eleanor"
]

def generate_random_child_name():
    return random.choice(CHILD_NAMES)

# Security configuration
SECRET_KEY = "your-secret-key"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/signin")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class PasswordRecovery(BaseModel):
    email: EmailStr

class PasswordReset(BaseModel):
    token: str
    new_password: str

def generate_reset_token():
    return secrets.token_urlsafe(32)

async def create_reset_token(user_id: str, expires_in_minutes: int = 30):
    token = generate_reset_token()
    expires_at = datetime.utcnow() + timedelta(minutes=expires_in_minutes)
    
    # Invalidate any existing tokens for this user
    await prisma.passwordresettoken.update_many(
        where={
            "user_id": user_id,
            "used": False
        },
        data={"used": True}
    )
    
    # Create new token
    reset_token = await prisma.passwordresettoken.create(
        data={
            "token": token,
            "user_id": user_id,
            "expires_at": expires_at
        }
    )
    return reset_token

async def verify_reset_token(token: str):
    reset_token = await prisma.passwordresettoken.find_unique(
        where={"token": token}
    )
    
    if not reset_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reset token"
        )
    
    if reset_token.used:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has already been used"
        )
    
    if reset_token.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has expired"
        )
    
    return reset_token

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

async def authenticate_user(email: str, password: str):
    user = await prisma.user.find_unique(where={"email": email})
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = await prisma.user.find_unique(where={"email": token_data.email})
    if user is None:
        raise credentials_exception
    return user

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user: UserRegister):
    existing_user = await prisma.user.find_unique(where={"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    hashed_password = get_password_hash(user.password)
    
    # Create user and child in a transaction
    async with prisma.tx() as transaction:
        new_user = await transaction.user.create(
            data={
                "email": user.email,
                "password": hashed_password,
                "full_name": user.full_name
            }
        )
        
        # Create a child with random name
        await transaction.child.create(
            data={
                "full_name": generate_random_child_name(),
                "user_id": new_user.id
            }
        )
    
    return {
        "message": "User registered successfully",
        "child_created": True
    }

@router.post("/signin", response_model=Token)
async def signin(user: UserLogin):
    authenticated_user = await authenticate_user(user.email, user.password)
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": authenticated_user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/recover-pass")
async def recover_password(recovery: PasswordRecovery):
    user = await prisma.user.find_unique(where={"email": recovery.email})
    if not user:
        # For security reasons, we don't reveal if the email exists
        return {"message": "If your email is registered, you will receive a password reset link"}
    
    # Create reset token
    reset_token = await create_reset_token(user.id)
    
    # In a real application, you would send an email here
    # For now, we'll return the token in the response for testing
    reset_link = f"/auth/reset-password?token={reset_token.token}"
    
    return {
        "message": "If your email is registered, you will receive a password reset link",
        "reset_link": reset_link  # Remove this in production
    }

@router.post("/reset-password")
async def reset_password(reset: PasswordReset):
    # Verify the reset token
    reset_token = await verify_reset_token(reset.token)
    
    # Update user's password
    hashed_password = get_password_hash(reset.new_password)
    await prisma.user.update(
        where={"id": reset_token.user_id},
        data={"password": hashed_password}
    )
    
    # Mark token as used
    await prisma.passwordresettoken.update(
        where={"id": reset_token.id},
        data={"used": True}
    )
    
    return {"message": "Password has been reset successfully"}

# Keep the old token endpoint for backward compatibility
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"} 