from medcode_rag.extraction.ner import extract_clinical_entities
from medcode_rag.generation.encoder import encode_entity
from medcode_rag.logging_config import get_logger
from medcode_rag.models.schemas import (
    EncodedEntry,
    EncodingResult,
    SkippedEntity,
    TranscriptInput,
)
from medcode_rag.retrieval.retriever import (
    retrieve_code_candidates,
    retrieve_error_patterns,
    retrieve_similar_cases,
)
from medcode_rag.security.phi import redact_for_logging
from medcode_rag.validation.rules import validate_entry

logger = get_logger(__name__)


def run_encoding_pipeline(transcript: TranscriptInput) -> EncodingResult:
    logger.info(
        "pipeline_start",
        extra={
            "transcript_id": transcript.transcript_id,
            "preview": redact_for_logging(transcript.raw_text[:120]),
        },
    )

    entities = extract_clinical_entities(transcript.raw_text)

    entries: list[EncodedEntry] = []
    skipped: list[SkippedEntity] = []

    for entity in entities:
        if not entity.is_codable:
            reason = "negated" if entity.is_negated else (
                "family history" if entity.is_family else "hypothetical"
            )
            skipped.append(SkippedEntity(text=entity.text, section=entity.section, reason=reason))
            continue

        candidates = retrieve_code_candidates(entity.text)
        similar_cases = retrieve_similar_cases(entity.text)
        error_patterns = retrieve_error_patterns(entity.text)

        entry = encode_entity(entity.text, entity.section, candidates, similar_cases, error_patterns)

        if entity.is_uncertain:
            entry.warnings.append("Source text marked as uncertain/possible finding")
        if entity.is_historical:
            entry.warnings.append("Source text refers to historical (past) condition")

        entries.append(entry)

    for entry in entries:
        entry.warnings = validate_entry(entry, entries)

    flagged = any(e.warnings or e.code == "UNCODED" for e in entries)

    result = EncodingResult(
        transcript_id=transcript.transcript_id,
        entries=entries,
        skipped_entities=skipped,
        flagged_for_review=flagged,
        review_reasons=[w for e in entries for w in e.warnings],
    )

    logger.info(
        "pipeline_complete",
        extra={
            "transcript_id": transcript.transcript_id,
            "entries": len(entries),
            "skipped": len(skipped),
            "flagged": flagged,
        },
    )
    return result
