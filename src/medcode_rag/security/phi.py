import re

# Patterns used ONLY for redacting log output / audit display text.
# The raw transcript itself is still processed in full by the pipeline,
# since real PHI context (dates, IDs) can be clinically relevant to coding.
_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "MRN": re.compile(r"\bMRN[:#]?\s*\d{4,10}\b", re.IGNORECASE),
    "DOB": re.compile(
        r"\b(0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])[/-](19|20)\d{2}\b"
    ),
}


def redact_for_logging(text: str) -> str:
    redacted = text
    for label, pattern in _PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return redacted
