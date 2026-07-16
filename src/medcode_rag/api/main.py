from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from sqlalchemy.ext.asyncio import AsyncSession

from medcode_rag.api.middleware import RequestContextMiddleware, limiter
from medcode_rag.audit.db import EncodingAuditLog, FeedbackAuditLog, get_session, init_db
from medcode_rag.feedback.logger import log_correction
from medcode_rag.logging_config import configure_logging, get_logger
from medcode_rag.models.schemas import EncodingResult, FeedbackRecord, TranscriptInput
from medcode_rag.pipeline import run_encoding_pipeline
from medcode_rag.retrieval.vector_store import vector_store
from medcode_rag.security.auth import verify_api_key

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup")
    vector_store.ensure_collections()
    await init_db()
    yield
    logger.info("shutdown")


app = FastAPI(title="MedCode RAG", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(RequestContextMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


@app.post("/encode", response_model=EncodingResult)  #, dependencies=[Depends(verify_api_key)]
@limiter.limit("30/minute")
async def encode(
    request,
    transcript: TranscriptInput,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = run_encoding_pipeline(transcript)
    except Exception as e:
        logger.exception("encoding_pipeline_failed", extra={"transcript_id": transcript.transcript_id})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    session.add(
        EncodingAuditLog(
            transcript_id=result.transcript_id,
            entries=[e.model_dump() for e in result.entries],
            flagged_for_review=result.flagged_for_review,
        )
    )
    await session.commit()

    return result


@app.post("/feedback") #, dependencies=[Depends(verify_api_key)]
@limiter.limit("30/minute")
async def feedback(
    request,
    record: FeedbackRecord,
    session: AsyncSession = Depends(get_session),
):
    try:
        log_correction(record)
    except Exception as e:
        logger.exception("feedback_logging_failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    session.add(FeedbackAuditLog(**record.model_dump()))
    await session.commit()

    return {"status": "logged"}


@app.get("/health")
async def health():
    qdrant_ok = vector_store.health_check()
    return {
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": qdrant_ok,
    }
