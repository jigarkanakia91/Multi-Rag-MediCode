from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from medcode_rag.config import settings
from medcode_rag.logging_config import get_logger

logger = get_logger(__name__)

_client = OpenAI(
    api_key=settings.nvidia_api_key,
    base_url=settings.nvidia_base_url,
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def embed_texts(texts: list[str], input_type: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    itype = input_type or settings.embedding_input_type
    response = _client.embeddings.create(
        input=texts,
        model=settings.embedding_model,
        encoding_format="float",
        extra_body={"input_type": itype, "truncate": "NONE"},
    )
    return [item.embedding for item in response.data]


def embed_text(text: str, input_type: str | None = None) -> list[float]:
    return embed_texts([text], input_type=input_type)[0]


def embed_query(text: str) -> list[float]:
    return embed_text(text, input_type="query")


def embed_passage(text: str) -> list[float]:
    return embed_text(text, input_type="passage")
