from medcode_rag.config import settings
from medcode_rag.embeddings.embedder import embed_query as embed_text
from medcode_rag.logging_config import get_logger
from medcode_rag.models.schemas import CodeCandidate, ErrorPattern
from medcode_rag.retrieval.vector_store import vector_store

logger = get_logger(__name__)


def retrieve_code_candidates(entity_text: str) -> list[CodeCandidate]:
    try:
        vec = embed_text(entity_text)
        hits = vector_store.search("code_kb", vec, settings.top_k_codes)
        return [
            CodeCandidate(
                code=h.payload["code"],
                code_system=h.payload["code_system"],
                description=h.payload["description"],
                score=h.score,
                source_chunk=h.payload["chunk_text"],
            )
            for h in hits
        ]
    except Exception as e:
        logger.error("code_candidate_retrieval_failed", extra={"error": str(e)})
        return []


def retrieve_similar_cases(entity_text: str) -> list[dict]:
    try:
        vec = embed_text(entity_text)
        hits = vector_store.search("transcripts", vec, settings.top_k_similar_cases)
        return [h.payload for h in hits]
    except Exception as e:
        logger.error("similar_case_retrieval_failed", extra={"error": str(e)})
        return []


def retrieve_error_patterns(entity_text: str) -> list[ErrorPattern]:
    try:
        vec = embed_text(entity_text)
        hits = vector_store.search("error_patterns", vec, settings.top_k_error_patterns)
        return [
            ErrorPattern(
                original_code=h.payload["original_code"],
                corrected_code=h.payload["corrected_code"],
                reason=h.payload["reason"],
                context_snippet=h.payload["context_snippet"],
                score=h.score,
            )
            for h in hits
        ]
    except Exception as e:
        logger.error("error_pattern_retrieval_failed", extra={"error": str(e)})
        return []
