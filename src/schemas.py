from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class Citation(BaseModel):
    """Granular source reference linking a recommendation back to its exact origin."""
    model_config = ConfigDict(frozen=True)

    document_name: str = Field(description="Exact document filename (e.g., NICE_CG124.pdf, USPSTF_Osteoporosis.pdf)")
    section_number: str = Field(description="Numerical guideline section identifier (e.g., '1.3', '1.6', '2.1')")
    section_title: str = Field(description="Title heading of the guideline section")
    page_number: int = Field(description="Physical page index in the source PDF (1-indexed)")
    evidence_grade: str = Field(
        default="N/A", 
        description="Official USPSTF recommendation grade (Grade A, B, C, D, I statement) or NICE recommendation marker"
    )
    target_population: Optional[str] = Field(default="Adults", description="Demographic scope specified by the snippet")


class ProcessedChunk(BaseModel):
    """A single section-aware chunk with clinical metadata and search-optimized text."""
    model_config = ConfigDict(json_schema_extra={
        "description": "Structured chunk produced by the ingestion pipeline"
    })

    chunk_id: str = Field(description="Deterministic unique identifier: {doc_stem}_p{page}_c{counter}")
    content: str = Field(description="Raw clinical text content of the chunk")
    embedding_text: str = Field(
        description="Section-prefixed text optimized for embedding similarity search. "
                    "Format: '[Section X.Y: Title] content...'"
    )
    section_number: str = Field(description="Hierarchical section identifier (e.g., '1.6', '3.5')")
    section_title: str = Field(description="Human-readable section heading")
    page_number: int = Field(ge=1, description="Physical page index in the source PDF (1-indexed)")
    document_name: str = Field(description="Source PDF filename")
    evidence_grade: str = Field(default="N/A", description="USPSTF grade or NICE recommendation marker")
    target_population: str = Field(default="Adults", description="Clinical demographic scope")
    char_count: int = Field(ge=0, description="Character count of the raw content field")


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
