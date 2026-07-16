import json
import uuid

from qdrant_client.models import PointStruct

from medcode_rag.embeddings.embedder import embed_texts
from medcode_rag.logging_config import get_logger
from medcode_rag.retrieval.vector_store import vector_store

logger = get_logger(__name__)


def load_historical_transcripts(jsonl_path: str, batch_size: int = 128) -> int:
    """
    Each line: {transcript_id, text, codes: [{"code": "...", "code_system": "..."}]}
    """
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        texts = [r["text"] for r in batch]
        vectors = embed_texts(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "transcript_id": r["transcript_id"],
                    "codes": r.get("codes", []),
                    "chunk_text": text,
                },
            )
            for r, vec, text in zip(batch, vectors, texts)
        ]
        vector_store.upsert("transcripts", points)
        total += len(points)

    logger.info("transcripts_loaded", extra={"count": total})
    return total
