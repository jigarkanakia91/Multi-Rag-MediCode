from pydantic import BaseModel, Field, field_validator


class TranscriptInput(BaseModel):
    transcript_id: str = Field(..., min_length=1, max_length=128)
    patient_id: str | None = Field(default=None, max_length=128)
    raw_text: str = Field(..., min_length=1, max_length=50_000)

    @field_validator("raw_text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("raw_text cannot be empty")
        return v


class ClinicalEntity(BaseModel):
    text: str
    label: str
    section: str
    start_char: int
    end_char: int
    is_negated: bool = False
    is_historical: bool = False
    is_hypothetical: bool = False
    is_family: bool = False
    is_uncertain: bool = False

    @property
    def is_codable(self) -> bool:
        return not (
            self.is_negated
            or self.is_hypothetical
            or self.is_family
        )


class CodeCandidate(BaseModel):
    code: str
    code_system: str
    description: str
    score: float
    source_chunk: str


class ErrorPattern(BaseModel):
    original_code: str
    corrected_code: str
    reason: str
    context_snippet: str
    score: float


class EncodedEntry(BaseModel):
    code: str
    code_system: str
    description: str
    source_text_span: str
    section: str
    confidence: float
    alternatives: list[CodeCandidate] = []
    warnings: list[str] = []


class SkippedEntity(BaseModel):
    text: str
    section: str
    reason: str


class EncodingResult(BaseModel):
    transcript_id: str
    entries: list[EncodedEntry]
    skipped_entities: list[SkippedEntity] = []
    flagged_for_review: bool
    review_reasons: list[str] = []


class FeedbackRecord(BaseModel):
    transcript_id: str
    original_code: str
    corrected_code: str
    reason: str = Field(..., min_length=1, max_length=1000)
    context_snippet: str = Field(..., min_length=1, max_length=2000)
