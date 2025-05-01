from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from database import prisma
from routes.auth import get_current_user, get_password_hash

router = APIRouter()

class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str

class ChildResponse(BaseModel):
    id: str
    full_name: str
    user_id: str

    class Config:
        from_attributes = True

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    children: Optional[List[ChildResponse]] = None

    class Config:
        from_attributes = True

class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None

@router.post("/", response_model=UserResponse)
async def create_user(user: UserCreate):
    existing_user = await prisma.user.find_unique(where={"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    hashed_password = get_password_hash(user.password)
    new_user = await prisma.user.create(
        data={
            "email": user.email,
            "password": hashed_password,
            "full_name": user.full_name
        }
    )
    return UserResponse(
        id=new_user.id,
        email=new_user.email,
        full_name=new_user.full_name
    )

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user = Depends(get_current_user)):
    # Fetch user with children
    user_with_children = await prisma.user.find_unique(
        where={"id": current_user.id},
        include={"children": True}
    )
    
    children_response = []
    if user_with_children.children:
        for child in user_with_children.children:
            children_response.append(
                ChildResponse(
                    id=child.id,
                    full_name=child.full_name,
                    user_id=child.user_id
                )
            )
    
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        children=children_response
    )

@router.put("/me", response_model=UserResponse)
async def update_user(
    user_data: UserUpdateRequest,
    current_user = Depends(get_current_user)
):
    # Build update data based on provided fields
    update_data = {}
    if user_data.full_name is not None:
        update_data["full_name"] = user_data.full_name
    
    if user_data.email is not None and user_data.email != current_user.email:
        # Check if email is already in use
        existing_user = await prisma.user.find_unique(where={"email": user_data.email})
        if existing_user and existing_user.id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered by another user"
            )
        update_data["email"] = user_data.email
    
    # Only update if there's data to update
    if update_data:
        updated_user = await prisma.user.update(
            where={"id": current_user.id},
            data=update_data,
            include={"children": True}
        )
        
        children_response = []
        if updated_user.children:
            for child in updated_user.children:
                children_response.append(
                    ChildResponse(
                        id=child.id,
                        full_name=child.full_name,
                        user_id=child.user_id
                    )
                )
        
        return UserResponse(
            id=updated_user.id,
            email=updated_user.email,
            full_name=updated_user.full_name,
            children=children_response
        )
    
    # If no updates, return current user data
    return await read_users_me(current_user)

@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(current_user = Depends(get_current_user)):
    await prisma.user.delete(where={"id": current_user.id})
    return None 