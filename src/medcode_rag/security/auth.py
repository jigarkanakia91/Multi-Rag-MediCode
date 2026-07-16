from fastapi import Header, HTTPException, status
from medcode_rag.config import settings


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    if not settings.valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication not configured",
        )
    if x_api_key not in settings.valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key
