from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams
from tenacity import retry, stop_after_attempt, wait_exponential

from medcode_rag.config import settings
from medcode_rag.logging_config import get_logger

logger = get_logger(__name__)

EMBEDDING_DIM = 768  # matches pritamdeka/S-PubMedBert-MS-MARCO

COLLECTIONS = {
    "code_kb": EMBEDDING_DIM,
    "transcripts": EMBEDDING_DIM,
    "error_patterns": EMBEDDING_DIM,
}


class VectorStore:
    def __init__(self) -> None:
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
            timeout=10,
        )

    def ensure_collections(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        for name, dim in COLLECTIONS.items():
            if name not in existing:
                logger.info("creating_collection", extra={"collection": name})
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def upsert(self, collection: str, points: list[PointStruct]) -> None:
        if not points:
            return
        self.client.upsert(collection_name=collection, points=points)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        flt: Filter | None = None,
    ):
        return self.client.search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k,
            query_filter=flt,
        )

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error("qdrant_health_check_failed", extra={"error": str(e)})
            return False


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()


vector_store = get_vector_store()
