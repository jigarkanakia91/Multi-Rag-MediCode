import json
import os
from functools import lru_cache

import medspacy
from medspacy.ner import TargetRule
from spacy.language import Language

from medcode_rag.config import settings
from medcode_rag.logging_config import get_logger
from medcode_rag.models.schemas import ClinicalEntity

logger = get_logger(__name__)

# Fallback base rules if target_rules.json is missing/empty.
_BASE_TARGET_RULES = [
    TargetRule("diabetes", "PROBLEM"),
    TargetRule("hypertension", "PROBLEM"),
    TargetRule("hyperglycemia", "PROBLEM"),
    TargetRule("pneumonia", "PROBLEM"),
    TargetRule("chest pain", "PROBLEM"),
    TargetRule("shortness of breath", "PROBLEM"),
    TargetRule("appendectomy", "PROCEDURE"),
    TargetRule("colonoscopy", "PROCEDURE"),
    TargetRule("x-ray", "PROCEDURE"),
    TargetRule("MRI", "PROCEDURE"),
    TargetRule("metformin", "MEDICATION"),
    TargetRule("lisinopril", "MEDICATION"),
]


def _load_target_rules_from_file(path: str) -> list[TargetRule]:
    """
    Loads target rules generated from the code knowledge base
    (see ingestion/code_loader.py -> export_target_rules) plus
    any manually curated rules. Expected JSON schema:
    [{"literal": "type 2 diabetes mellitus", "category": "PROBLEM"}, ...]
    """
    if not os.path.exists(path):
        logger.warning("target_rules_file_missing", extra={"path": path})
        return []

    try:
        with open(path, encoding="utf-8") as f:
            raw_rules = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("target_rules_load_failed", extra={"error": str(e)})
        return []

    rules = []
    for r in raw_rules:
        try:
            rules.append(TargetRule(literal=r["literal"], category=r["category"]))
        except (KeyError, TypeError):
            continue
    logger.info("target_rules_loaded", extra={"count": len(rules)})
    return rules


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """
    Builds and caches the medspaCy pipeline once per process.
    medspaCy's default pipeline is rule-based (blank spaCy model +
    PyRuSH sentence splitter + target matcher + ConText), so it is
    lightweight and does not require downloading a large statistical model.
    """
    nlp = medspacy.load()

    if "medspacy_sectionizer" not in nlp.pipe_names:
        nlp.add_pipe("medspacy_sectionizer")

    if "medspacy_target_matcher" not in nlp.pipe_names:
        nlp.add_pipe("medspacy_target_matcher")

    target_matcher = nlp.get_pipe("medspacy_target_matcher")

    file_rules = _load_target_rules_from_file(settings.target_rules_path)
    combined_rules = file_rules if file_rules else _BASE_TARGET_RULES
    target_matcher.add(combined_rules)

    logger.info(
        "medspacy_pipeline_ready",
        extra={"pipe_names": nlp.pipe_names, "rule_count": len(combined_rules)},
    )
    return nlp


def _get_section_category(doc, ent) -> str:
    """
    Safely resolves the section category for an entity span,
    tolerating minor API differences across medspaCy versions.
    """
    try:
        sections = doc._.sections
    except AttributeError:
        return "unknown"

    for section in sections:
        body = getattr(section, "body_span", None) or getattr(section, "body", None)
        if body is not None and body.start <= ent.start and ent.end <= body.end:
            category = getattr(section, "category", None)
            return category or "unknown"
    return "unknown"


def extract_clinical_entities(text: str) -> list[ClinicalEntity]:
    """
    Runs the full medspaCy pipeline and returns entities annotated with
    section context and ConText assertion attributes (negation, family
    history, hypothetical, historical, uncertainty). Downstream coding
    logic should only act on entities where `is_codable` is True.
    """
    nlp = get_nlp()
    doc = nlp(text)

    entities: list[ClinicalEntity] = []
    for ent in doc.ents:
        entities.append(
            ClinicalEntity(
                text=ent.text,
                label=ent.label_,
                section=_get_section_category(doc, ent),
                start_char=ent.start_char,
                end_char=ent.end_char,
                is_negated=bool(getattr(ent._, "is_negated", False)),
                is_historical=bool(getattr(ent._, "is_historical", False)),
                is_hypothetical=bool(getattr(ent._, "is_hypothetical", False)),
                is_family=bool(getattr(ent._, "is_family", False)),
                is_uncertain=bool(getattr(ent._, "is_uncertain", False)),
            )
        )

    logger.info(
        "entities_extracted",
        extra={"count": len(entities), "codable": sum(e.is_codable for e in entities)},
    )
    return entities
