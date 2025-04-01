from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.prisma import prisma
from fastapi import HTTPException

router = APIRouter()

class FeedbackCreate(BaseModel):
    rating: int
    comment: str
    user_id: str

class FeedbackResponse(BaseModel):
    rating: float

@router.get("/feedback", response_model=FeedbackResponse, tags=["feedback"])
async def get_average_feedback():
    feedbacks = await prisma.feedback.find_many()
    if not feedbacks:
        raise HTTPException(status_code=404, detail="No feedbacks found")
    avg_rating = sum(f.rating for f in feedbacks) / len(feedbacks)
    return {"rating": avg_rating}

@router.get("/feedback/{user_id}", response_model=list[FeedbackCreate], tags=["feedback"])
async def get_user_feedback(user_id: str):
    feedbacks = await prisma.feedback.find_many(where={"user_id": user_id})
    if not feedbacks:
        raise HTTPException(status_code=404, detail="No feedbacks found for this user")
    return feedbacks

@router.post("/feedback", tags=["feedback"])
async def add_feedback(feedback: FeedbackCreate):
    new_feedback = await prisma.feedback.create(data=feedback.dict())
    return new_feedback
