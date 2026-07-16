import uuid

from qdrant_client.models import PointStruct

from medcode_rag.embeddings.embedder import embed_text
from medcode_rag.logging_config import get_logger
from medcode_rag.models.schemas import FeedbackRecord
from medcode_rag.retrieval.vector_store import vector_store

logger = get_logger(__name__)


def log_correction(feedback: FeedbackRecord) -> None:
    text = (
        f"Context: {feedback.context_snippet} | Wrong: {feedback.original_code} "
        f"-> Correct: {feedback.corrected_code} because {feedback.reason}"
    )
    vec = embed_text(text)
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vec,
        payload={**feedback.model_dump(), "chunk_text": text},
    )
    vector_store.upsert("error_patterns", [point])
    logger.info(
        "feedback_logged",
        extra={"transcript_id": feedback.transcript_id, "corrected_code": feedback.corrected_code},
    )
