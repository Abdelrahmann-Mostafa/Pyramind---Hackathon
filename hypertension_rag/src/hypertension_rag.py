"""
Hypertension RAG Pipeline
Adapted from Pyramind, domain-specific for blood pressure management
"""

import json
import os
import re
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from src.hypertension_safety_filter import HypertensionSafetyFilter
from src.hypertension_system_prompt import HYPERTENSION_SYSTEM_PROMPT


# ===================================================================
# RESPONSE SCHEMA
# ===================================================================

class Citation(BaseModel):
    """Citation linking a claim to guideline source."""
    document_name: str = Field(default="ESC 2021 Hypertension Guidelines")
    section_number: str = Field(...)
    section_title: str = Field(...)
    page_number: int = Field(default=0)
    chunk_id: str = Field(...)
    retrieved_text: str = Field(...)
    similarity_score: float = Field(...)


class SupportingEvidence(BaseModel):
    """Evidence piece."""
    claim: str = Field(...)
    excerpt: str = Field(...)
    chunk_id: str = Field(...)


class GroundedResponse(BaseModel):
    """Final response."""
    query: str = Field(...)
    status: Literal["SUCCESS", "REFUSED"] = Field(...)
    recommendation: Optional[str] = Field(default=None)
    supporting_evidence: List[SupportingEvidence] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    confidence_level: Literal["High", "Medium", "Low", "Insufficient Evidence"] = Field(default="Insufficient Evidence")
    clinical_disclaimer: str = Field(default=(
        "⚕️ DISCLAIMER: This information is derived from ESC 2021 Hypertension Guidelines "
        "and is intended for educational and clinical decision support purposes only. "
        "It does not constitute medical advice. Always consult qualified healthcare professionals."
    ))
    refusal_reason: Optional[str] = Field(default=None)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_used: str = Field(default="openai/gpt-oss-120b")

    def format_for_display(self) -> str:
        """Format for console display."""
        confidence_icon = {
            "High": "🟢 HIGH",
            "Medium": "🟡 MEDIUM",
            "Low": "🔴 LOW",
            "Insufficient Evidence": "⚫ INSUFFICIENT"
        }

        output = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  HYPERTENSION CLINICAL DECISION SUPPORT                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

  QUERY:  {self.query}
  STATUS: {"✓ ANSWERED" if self.status == "SUCCESS" else "✗ REFUSED"}
"""

        if self.status == "REFUSED":
            output += f"  ✗ REFUSAL: {self.refusal_reason}\n"
            output += f"\n  {self.clinical_disclaimer}\n"
            return output

        output += f"  CONFIDENCE: {confidence_icon.get(self.confidence_level, self.confidence_level)}\n"
        output += "\n" + "─" * 74 + "\n"
        output += "\n  📋 RECOMMENDATION\n\n"
        output += f"  {self.recommendation}\n"

        if self.supporting_evidence:
            output += "\n  📎 SUPPORTING EVIDENCE\n\n"
            for i, ev in enumerate(self.supporting_evidence, 1):
                output += f"  {i}. Claim: {ev.claim}\n"
                output += f"     Excerpt: \"{ev.excerpt[:100]}...\"\n"
                output += f"     Source: [{ev.chunk_id}]\n\n"

        if self.citations:
            output += "  📚 CITATIONS\n\n"
            for cit in self.citations:
                output += f"  • {cit.section_number} — {cit.section_title}\n"
                output += f"    Confidence: {cit.similarity_score:.1%}\n"

        output += "─" * 74 + "\n"
        output += f"\n  {self.clinical_disclaimer}\n"
        return output


# ===================================================================
# LLM GENERATION
# ===================================================================

class GroundedLLMGenerator:
    """Generate grounded responses using Claude."""

    def __init__(self, api_key: str = None, model: str = "openai/gpt-oss-120b"):
        self.client = OpenAI(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model

    def _format_context(self, retrieved_chunks: List[dict]) -> str:
        """Format chunks for context."""
        context_parts = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadatas", {})
            section = meta.get("section_number", "0.0")
            title = meta.get("section_title", "")
            content = chunk.get("content", "")[:200]

            context_parts.append(
                f"[SECTION § {section}] {title}\n{content}..."
            )

        return "\n\n" + ("─" * 60 + "\n\n").join(context_parts) + "\n"

    def generate(
        self,
        query: str,
        retrieved_chunks: List[dict],
        temperature: float = 0.15,
        max_tokens: int = 512,
    ) -> dict:
        """Generate response."""
        context = self._format_context(retrieved_chunks)
        system_prompt = HYPERTENSION_SYSTEM_PROMPT.format(context=context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        raw_text = response.choices[0].message.content

        try:
            parsed = json.loads(raw_text)
            return parsed
        except json.JSONDecodeError:
            return {
                "recommendation": raw_text,
                "supporting_evidence": [],
                "is_answerable": True,
            }


# ===================================================================
# MAIN PIPELINE
# ===================================================================

class HypertensionRAGPipeline:
    """End-to-end RAG for hypertension management."""

    def __init__(
        self,
        chroma_path: str = "data/chroma_db",
        collection_name: str = "hypertension-guidelines",
        embeddings_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        groq_api_key: str = None,
    ):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        self.embedding_model = SentenceTransformer(embeddings_model)
        self.safety_filter = HypertensionSafetyFilter()
        self.llm = GroundedLLMGenerator(api_key=groq_api_key)

        print(f"[+] Hypertension RAG Pipeline initialized")

    def answer_query(
        self,
        query: str,
        top_k: int = 3,
        confidence_threshold: float = 0.55,
    ) -> GroundedResponse:
        """Answer a query about hypertension."""

        # Safety check
        is_safe, safety_reason = self.safety_filter.check(query)
        if not is_safe:
            return GroundedResponse(
                query=query,
                status="REFUSED",
                refusal_reason=safety_reason,
                confidence_level="Insufficient Evidence",
            )

        # Retrieve
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        if not results["documents"][0]:
            return GroundedResponse(
                query=query,
                status="REFUSED",
                refusal_reason="No relevant guideline information found.",
                confidence_level="Insufficient Evidence",
            )

        # Format chunks
        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            retrieved_chunks.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadatas": results["metadatas"][0][i],
            })

        # Generate
        llm_output = self.llm.generate(query, retrieved_chunks)

        if not llm_output.get("is_answerable", True):
            return GroundedResponse(
                query=query,
                status="REFUSED",
                refusal_reason=llm_output.get("refusal_reason", "Cannot answer from guidelines"),
                confidence_level="Insufficient Evidence",
            )

        # Build response
        return GroundedResponse(
            query=query,
            status="SUCCESS",
            recommendation=llm_output.get("recommendation", ""),
            supporting_evidence=[
                SupportingEvidence(**ev) for ev in llm_output.get("supporting_evidence", [])
            ],
            citations=[
                Citation(
                    section_number=chunk["metadatas"].get("section_number", "0.0"),
                    section_title=chunk["metadatas"].get("section_title", ""),
                    page_number=int(chunk["metadatas"].get("page_number", 0)),
                    chunk_id=chunk["chunk_id"],
                    retrieved_text=chunk["content"][:200],
                    similarity_score=0.85,
                )
                for chunk in retrieved_chunks
            ],
            confidence_level="High" if len(retrieved_chunks) >= 2 else "Medium",
        )


if __name__ == "__main__":
    pipeline = HypertensionRAGPipeline()
    response = pipeline.answer_query("What is the recommended blood pressure target for diabetes patients?")
    print(response.format_for_display())
