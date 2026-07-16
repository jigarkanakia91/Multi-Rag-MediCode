from medcode_rag.config import settings
from medcode_rag.models.schemas import EncodedEntry

# Populate from payer/coding guidelines
MUTUALLY_EXCLUSIVE: dict[tuple[str, str], str] = {
    ("E11.9", "E11.65"): "Diabetes without complications vs with hyperglycemia — check specificity",
}


def validate_entry(entry: EncodedEntry, all_entries: list[EncodedEntry]) -> list[str]:
    warnings = list(entry.warnings)

    if entry.code == "UNCODED":
        return warnings

    if entry.confidence < settings.low_confidence_threshold:
        warnings.append(
            f"Low confidence ({entry.confidence:.2f}) — flagged for human review"
        )

    codes_in_result = {e.code for e in all_entries}
    for (a, b), msg in MUTUALLY_EXCLUSIVE.items():
        if a in codes_in_result and b in codes_in_result:
            warnings.append(f"Conflict: {a} and {b} — {msg}")

    return warnings
