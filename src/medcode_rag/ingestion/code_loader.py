import json
import uuid

from qdrant_client.models import PointStruct

from medcode_rag.embeddings.embedder import embed_texts
from medcode_rag.logging_config import get_logger
from medcode_rag.retrieval.vector_store import vector_store

logger = get_logger(__name__)


def load_code_kb(json_path: str, code_system: str, batch_size: int = 256) -> int:
    """
    Expects JSON array of objects with fields: code, description, guideline (optional)
    """
    with open(json_path, encoding="utf-8") as f:
        rows = json.load(f)

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [f"{r['code']}: {r['description']} {r.get('guideline', '')}" for r in batch]
        vectors = embed_texts(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "code": r["code"],
                    "code_system": code_system,
                    "description": r["description"],
                    "guideline": r.get("guideline", ""),
                    "chunk_text": text,
                },
            )
            for r, vec, text in zip(batch, vectors, texts)
        ]
        vector_store.upsert("code_kb", points)
        total += len(points)

    logger.info("code_kb_loaded", extra={"code_system": code_system, "count": total})
    return total


def export_target_rules(json_path: str, category: str, output_path: str, append: bool = True) -> int:
    """
    Generates medspaCy TargetRule JSON entries from code descriptions,
    so NER matching stays aligned with the coding knowledge base.
    """
    rules = []
    if append:
        try:
            with open(output_path, encoding="utf-8") as f:
                rules = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            rules = []

    with open(json_path, encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        description = row["description"].strip().lower()
        if description:
            rules.append({"literal": description, "category": category})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)

    logger.info("target_rules_exported", extra={"count": len(rules), "output": output_path})
    return len(rules)
