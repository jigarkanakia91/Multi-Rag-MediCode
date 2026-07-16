import json

from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential

from medcode_rag.config import settings
from medcode_rag.generation.prompts import ENCODING_SYSTEM_PROMPT, build_user_prompt
from medcode_rag.logging_config import get_logger
from medcode_rag.models.schemas import CodeCandidate, EncodedEntry

logger = get_logger(__name__)

_client = OpenAI(
    api_key=settings.nvidia_api_key,
    base_url=settings.nvidia_base_url,
    timeout=settings.llm_timeout_seconds,
)


def _uncoded_entry(entity_text: str, section: str, reason: str) -> EncodedEntry:
    return EncodedEntry(
        code="UNCODED",
        code_system="NONE",
        description=reason,
        source_text_span=entity_text,
        section=section,
        confidence=0.0,
        warnings=[reason],
    )


@retry(stop=stop_after_attempt(settings.llm_max_retries), wait=wait_exponential(min=1, max=10))
def _call_llm(system_prompt: str, user_prompt: str) -> dict:
    response = _client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def encode_entity(
    entity_text: str,
    section: str,
    candidates: list[CodeCandidate],
    similar_cases: list[dict],
    error_patterns: list,
) -> EncodedEntry:
    if not candidates:
        return _uncoded_entry(entity_text, section, "No candidates retrieved — needs manual coding")

    user_prompt = build_user_prompt(entity_text, candidates, similar_cases, error_patterns)

    try:
        parsed = _call_llm(ENCODING_SYSTEM_PROMPT, user_prompt)
    except (OpenAIError, json.JSONDecodeError) as e:
        logger.error("llm_call_failed", extra={"error": str(e)})
        return _uncoded_entry(entity_text, section, f"LLM call failed: {e}")

    try:
        return EncodedEntry(
            code=parsed["code"],
            code_system=parsed["code_system"],
            description=parsed["description"],
            source_text_span=entity_text,
            section=section,
            confidence=float(parsed.get("confidence", 0.0)),
            alternatives=candidates,
            warnings=parsed.get("warnings", []),
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.error("llm_response_malformed", extra={"error": str(e), "raw": parsed})
        return _uncoded_entry(entity_text, section, "Malformed LLM response — needs manual coding")
