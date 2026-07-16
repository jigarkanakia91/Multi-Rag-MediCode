import json
import uuid

from qdrant_client.models import PointStruct

from medcode_rag.embeddings.embedder import embed_texts
from medcode_rag.logging_config import get_logger
from medcode_rag.retrieval.vector_store import vector_store

logger = get_logger(__name__)


def load_error_logs(jsonl_path: str, batch_size: int = 128) -> int:
    """
    Each line: {original_code, corrected_code, reason, context_snippet}
    """
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        texts = [
            f"Context: {r['context_snippet']} | Wrong: {r['original_code']} "
            f"-> Correct: {r['corrected_code']} because {r['reason']}"
            for r in batch
        ]
        vectors = embed_texts(texts)
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vec, payload={**r, "chunk_text": text})
            for r, vec, text in zip(batch, vectors, texts)
        ]
        vector_store.upsert("error_patterns", points)
        total += len(points)

    logger.info("error_logs_loaded", extra={"count": total})
    return total
