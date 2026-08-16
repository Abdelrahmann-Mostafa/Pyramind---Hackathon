from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_name: str = Field(description="Exact document filename (e.g., NICE_CG124.pdf, USPSTF_Osteoporosis.pdf)")
    section_number: str = Field(description="Numerical guideline section identifier (e.g., '1.3', '1.6', '2.1')")
    section_title: str = Field(description="Title heading of the guideline section")
    page_number: int = Field(description="Physical page index in the source PDF (1-indexed)")
    evidence_grade: str = Field(
        default="N/A", 
        description="Official USPSTF recommendation grade (Grade A, B, C, D, I statement) or NICE recommendation marker"
    )
    target_population: Optional[str] = Field(default="Adults", description="Demographic scope specified by the snippet")


class ChunkMetadata(BaseModel):
    document_name: str
    section_number: str
    section_title: str
    page_number: int
    evidence_grade: str = "N/A"
    target_population: str = "General / Adults"
    chunk_index: int = 0


class ProcessedChunk(BaseModel):
    chunk_id: str
    content: str
    section_number: str
    section_title: str
    page_number: int
    document_name: str
    evidence_grade: str = "N/A"
    target_population: str = "General / Adults"
    char_count: int


class ClinicalRecommendationItem(BaseModel):
    clinical_guidance: str = Field(description="Synthesized clinical directive grounded strictly in context")
    supporting_direct_excerpt: str = Field(description="Verbatim text quote from the retrieved document")
    citation: Citation


class CDSResponseSchema(BaseModel):
    query_status: Literal["SUCCESS", "REFUSED"] = Field(description="Status: 'SUCCESS' or 'REFUSED'")
    refusal_reason: Optional[str] = Field(default=None, description="Detailed reason if query was refused")
    confidence_score: Optional[float] = Field(default=None, description="Calibrated retrieval confidence score (0.0 to 1.0)")
    recommendations: List[ClinicalRecommendationItem] = Field(default_factory=list)
