from fastapi import APIRouter

router = APIRouter(prefix="/video", tags=["video"])

@router.get("/")
async def get_video():
    return "Hello, World!"

