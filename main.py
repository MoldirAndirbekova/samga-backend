from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from src.apis import apis
from src.prisma import prisma

@asynccontextmanager
async def lifespan(app: FastAPI):
    await prisma.connect()
    yield
    await prisma.disconnect()

app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.include_router(apis, prefix="/apis")

@app.get("/")
def read_root():
    return {"version": "1.0.0"}
