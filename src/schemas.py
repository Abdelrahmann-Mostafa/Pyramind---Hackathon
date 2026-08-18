from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

class Citation(BaseModel):
    """Granular source reference linking a recommendation to its exact origin."""
    model_config = ConfigDict(frozen=True)

    document_name: str = Field(description="Exact document filename (e.g., NICE_CG124.pdf)")
    section_number: str = Field(description="Numerical guideline section identifier (e.g., '1.3')")
    section_title: str = Field(description="Title heading of the guideline section")
    page_number: int = Field(description="Physical page index in the source PDF (1-indexed)")
    evidence_grade: str = Field(default="N/A", description="USPSTF grade or NICE marker")
    target_population: Optional[str] = Field(default="Adults", description="Demographic scope")


class ProcessedChunk(BaseModel):
    """A single section-aware chunk (child or parent) with clinical metadata."""
    model_config = ConfigDict(json_schema_extra={
        "description": "Structured chunk produced by the ingestion pipeline"
    })

    chunk_id: str = Field(description="Unique identifier (e.g., 'NICE_CG124_p06_c001')")
    content: str = Field(description="Raw clinical text content of the chunk")
    embedding_text: str = Field(
        description="Section-prefixed text optimized for embedding similarity search. "
                    "Format: '[Section X.Y: Title] content...'"
    )
    parent_id: Optional[str] = Field(default=None, description="ID of the parent chunk (only for child chunks)")
    section_number: str = Field(description="Hierarchical section identifier (e.g., '1.6')")
    section_title: str = Field(description="Human-readable section heading")
    section_type: str = Field(default="General", description="Type: Diagnosis, Treatment, Screening, etc.")
    page_number: int = Field(ge=1, description="Physical page index in the source PDF (1-indexed)")
    document_name: str = Field(description="Source PDF filename")
    evidence_grade: str = Field(default="N/A", description="USPSTF grade or NICE recommendation marker")
    target_population: str = Field(default="Adults", description="Clinical demographic scope")
    char_count: int = Field(ge=0, description="Character count of the raw content field")
    is_parent: bool = Field(default=False, description="True if this is a parent chunk (for context generation)")


class RetrievalResult(BaseModel):
    """A single retrieved chunk with similarity score and all metadata."""
    rank: int
    chunk_id: str
    content: str
    similarity_score: float
    parent_chunk: Optional[ProcessedChunk] = None  # full parent chunk for generation context
    document_name: str
    section_number: str
    section_title: str
    page_number: int
    evidence_grade: str
    target_population: str


class ClinicalRecommendationItem(BaseModel):
    """A single grounded clinical recommendation with verbatim evidence and citation."""
    clinical_guidance: str = Field(description="Synthesized clinical directive grounded strictly in context")
    supporting_direct_excerpt: str = Field(description="Verbatim text quote from the retrieved document")
    citation: Citation


class CDSResponseSchema(BaseModel):
    """Top-level response schema for the Clinical Decision Support system."""
    query_status: Literal["SUCCESS", "REFUSED"] = Field(description="Status: 'SUCCESS' or 'REFUSED'")
    refusal_reason: Optional[str] = Field(default=None, description="Detailed reason if query was refused")
    confidence_score: Optional[float] = Field(default=None, description="Calibrated retrieval confidence score (0.0 to 1.0)")
    recommendations: List[ClinicalRecommendationItem] = Field(default_factory=list)