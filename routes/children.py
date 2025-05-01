from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from database import prisma
from routes.auth import get_current_user

router = APIRouter()

class ChildCreate(BaseModel):
    full_name: str

class ChildResponse(BaseModel):
    id: str
    full_name: str
    user_id: str

    class Config:
        from_attributes = True

class ChildUpdate(BaseModel):
    full_name: str

@router.post("/", response_model=ChildResponse, status_code=status.HTTP_201_CREATED)
async def create_child(
    child: ChildCreate,
    current_user = Depends(get_current_user)
):
    new_child = await prisma.child.create(
        data={
            "full_name": child.full_name,
            "user_id": current_user.id
        }
    )
    return ChildResponse(
        id=new_child.id,
        full_name=new_child.full_name,
        user_id=new_child.user_id
    )

@router.get("/", response_model=List[ChildResponse])
async def get_children(current_user = Depends(get_current_user)):
    children = await prisma.child.find_many(
        where={"user_id": current_user.id}
    )
    return [
        ChildResponse(
            id=child.id,
            full_name=child.full_name,
            user_id=child.user_id
        )
        for child in children
    ]

@router.get("/{child_id}", response_model=ChildResponse)
async def get_child(
    child_id: str,
    current_user = Depends(get_current_user)
):
    child = await prisma.child.find_unique(
        where={"id": child_id}
    )
    if not child or child.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Child not found"
        )
    return ChildResponse(
        id=child.id,
        full_name=child.full_name,
        user_id=child.user_id
    )

@router.put("/{child_id}", response_model=ChildResponse)
async def update_child(
    child_id: str,
    child_data: ChildUpdate,
    current_user = Depends(get_current_user)
):
    child = await prisma.child.find_unique(
        where={"id": child_id}
    )
    if not child or child.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Child not found"
        )
    
    updated_child = await prisma.child.update(
        where={"id": child_id},
        data={"full_name": child_data.full_name}
    )
    return ChildResponse(
        id=updated_child.id,
        full_name=updated_child.full_name,
        user_id=updated_child.user_id
    )

@router.delete("/{child_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_child(
    child_id: str,
    current_user = Depends(get_current_user)
):
    child = await prisma.child.find_unique(
        where={"id": child_id}
    )
    if not child or child.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Child not found"
        )
    
    await prisma.child.delete(where={"id": child_id})
    return None