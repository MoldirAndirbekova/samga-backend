from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import auth, users, children, games
from database import prisma
from seed import seed_database  # Import the seed script

app = FastAPI(title="Children Games API")

@app.on_event("startup")
async def startup():
    await prisma.connect()
    # Run the seed script to ensure the database has initial data
    await seed_database()

@app.on_event("shutdown")
async def shutdown():
    await prisma.disconnect()

origins = [
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
    "https://samga-frontend-production.up.railway.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(children.router, prefix="/children", tags=["Children"])
app.include_router(games.router, prefix="/games", tags=["Games"])

@app.get("/")
async def root():
    return {"message": "Welcome to Children Games API"} 