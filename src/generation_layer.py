"""
OPTIMIZED VERSION OF generation_layer.py
=========================================

Changes made:
1. ✅ Fixed model name: llama-3.1-8b-instant → llama-3.3-70b-versatile
2. ✅ Compressed system prompt (~67% smaller)
3. ✅ Reduced context window from 5 to 3 chunks
4. ✅ Lowered max_tokens from 1024 to 512
5. ✅ Truncated chunk content to 200 chars per chunk
6. ✅ Optimized _format_context() for token efficiency

Total estimated savings: ~70% tokens per request
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


# ===================================================================
# SCHEMA DEFINITIONS (Pydantic) - NO CHANGES
# ===================================================================

class Citation(BaseModel):
    """Structured citation linking a claim to its exact source chunk."""
    document_name: str = Field(..., description="PDF filename, e.g., 'NICE_CG124.pdf'")
    section_number: str = Field(..., description="Section ID, e.g., '1.5'")
    section_title: str = Field(..., description="Section heading")
    page_number: int = Field(..., description="Page number (1-indexed)")
    chunk_id: str = Field(..., description="Unique chunk identifier from the vector store")
    evidence_grade: str = Field(default="N/A", description="USPSTF grade or NICE marker")
    target_population: str = Field(default="Adults", description="Guideline population scope")
    retrieved_text: str = Field(..., description="Direct excerpt from the guideline")
    similarity_score: float = Field(..., description="Retrieval confidence (0.0-1.0)")


class SupportingEvidence(BaseModel):
    """A single piece of evidence linking an excerpt to a claim."""
    claim: str = Field(..., description="The specific clinical claim being supported")
    excerpt: str = Field(..., description="Verbatim quote from the guideline")
    chunk_id: str = Field(..., description="Chunk ID that contains this excerpt")


class GroundedResponse(BaseModel):
    """Complete response following Day 3 specification."""
    query: str = Field(..., description="Original user query")
    status: Literal["SUCCESS", "REFUSED"] = Field(..., description="Did we answer or refuse?")

    # --- Success fields ---
    recommendation: Optional[str] = Field(
        default=None,
        description="Short, direct summary grounded solely in evidence"
    )
    supporting_evidence: List[SupportingEvidence] = Field(
        default_factory=list,
        description="Bullet points linking excerpts directly to claims"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Complete citation mapping for all retrieved chunks"
    )
    confidence_level: Literal["High", "Medium", "Low", "Insufficient Evidence"] = Field(
        default="Insufficient Evidence",
        description="Calibrated confidence status"
    )
    clinical_disclaimer: str = Field(
        default=(
            "⚕️ DISCLAIMER: This information is derived from clinical guidelines and is "
            "intended for educational and clinical decision support purposes only. It does "
            "not constitute medical advice. Always consult qualified healthcare professionals "
            "for patient-specific treatment decisions."
        ),
        description="Standard clinical disclaimer"
    )

    # --- Refusal fields ---
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Detailed reason if query was refused"
    )
    context_found: Optional[str] = Field(
        default=None,
        description="What context WAS found (for informative refusal messages)"
    )
    context_lacking: Optional[str] = Field(
        default=None,
        description="What context is MISSING (for informative refusal messages)"
    )

    # --- Metadata ---
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_used: str = Field(default="llama-3.3-70b-versatile")
    retrieval_scores: List[float] = Field(default_factory=list, description="Similarity scores from retrieval")

    def format_for_display(self) -> str:
        """Formats the response for console/Streamlit display."""
        confidence_icon = {
            "High": "🟢 HIGH",
            "Medium": "🟡 MEDIUM",
            "Low": "🔴 LOW",
            "Insufficient Evidence": "⚫ INSUFFICIENT"
        }

        output = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  CLINICAL DECISION SUPPORT RESPONSE                                    ║
╚══════════════════════════════════════════════════════════════════════════╝

  QUERY:  {self.query}
  STATUS: {"✓ ANSWERED" if self.status == "SUCCESS" else "✗ REFUSED"}
"""

        if self.status == "REFUSED":
            output += f"""
  ┌─ REFUSAL REASON ──────────────────────────────────────────────────────┐
  │ {self.refusal_reason}
  └───────────────────────────────────────────────────────────────────────┘
"""
            if self.context_found:
                output += f"  Context found:   {self.context_found}\n"
            if self.context_lacking:
                output += f"  Context lacking: {self.context_lacking}\n"
            output += f"\n  {self.clinical_disclaimer}\n"
            return output

        # --- SUCCESS case ---
        output += f"  CONFIDENCE: {confidence_icon.get(self.confidence_level, self.confidence_level)}\n"
        output += "\n" + "─" * 74 + "\n"
        output += "\n  📋 RECOMMENDATION\n\n"
        output += f"  {self.recommendation}\n"

        if self.supporting_evidence:
            output += "\n  📎 SUPPORTING EVIDENCE\n\n"
            for i, ev in enumerate(self.supporting_evidence, 1):
                output += f"  {i}. Claim: {ev.claim}\n"
                output += f"     Excerpt: \"{ev.excerpt}\"\n"
                output += f"     Source: [{ev.chunk_id}]\n\n"

        if self.citations:
            output += "  📚 CITATIONS\n\n"
            for cit in self.citations:
                output += f"  • {cit.document_name} § {cit.section_number}"
                output += f" — {cit.section_title} (p. {cit.page_number})\n"
                output += f"    Chunk: {cit.chunk_id} | "
                output += f"Grade: {cit.evidence_grade} | "
                output += f"Confidence: {cit.similarity_score:.1%}\n"
                output += f"    Population: {cit.target_population}\n\n"

        output += "─" * 74 + "\n"
        output += f"\n  {self.clinical_disclaimer}\n"
        output += "\n" + "═" * 74 + "\n"
        return output

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)


# ===================================================================
# SAFETY & REFUSAL LOGIC - NO CHANGES
# ===================================================================

class SafetyFilter:
    """Pre-retrieval safety filter: detects out-of-scope queries."""

    OUT_OF_SCOPE_KEYWORDS = {
        "covid", "coronavirus", "diabetes", "heart disease", "cardiac arrest",
        "myocardial infarction", "stroke", "neurological", "parkinson",
        "alzheimer", "cancer", "tumor", "oncology", "lung cancer",
        "breast cancer", "prostate", "mental health", "anxiety disorder",
        "depression", "ocd", "ptsd", "schizophrenia", "bipolar",
        "pregnancy", "obstetric", "gynecology", "asthma", "copd",
        "respiratory failure", "pneumonia", "gastric", "liver disease",
        "kidney disease", "renal failure", "dialysis", "transplant",
        "autoimmune", "rheumatoid arthritis", "gout", "lupus",
        "dermatology", "psoriasis", "eczema", "insomnia",
    }

    IN_SCOPE_KEYWORDS = {
        "hip fracture", "osteoporosis", "bone density", "fracture risk",
        "screening", "dxa", "dexa", "t-score", "frax", "qfracture",
        "bisphosphonate", "alendronate", "risedronate", "ibandronate",
        "zoledronic", "hormone replacement", "hrt", "calcium", "vitamin d",
        "elderly", "older adults", "women", "men", "postmenopausal",
        "menopause", "analgesia", "pain management", "surgery", "surgical",
        "orthopedic", "orthopaedic", "geriatric", "rehabilitation",
        "mobilisation", "mobilization", "physiotherapy", "fall",
        "fragility fracture", "vertebral", "femoral", "hip",
        "subtrochanteric", "intracapsular", "extracapsular",
        "arthroplasty", "hemiarthroplasty", "intramedullary",
        "bone", "fracture",
    }

    @staticmethod
    def check(query: str) -> tuple[bool, Optional[str]]:
        """Returns (is_safe, refusal_reason_if_unsafe)."""
        query_lower = query.lower()

        for keyword in SafetyFilter.OUT_OF_SCOPE_KEYWORDS:
            if keyword in query_lower:
                return False, (
                    f"Your query mentions '{keyword}', which is outside the scope "
                    f"of our clinical guidelines (hip fracture management and "
                    f"osteoporosis screening/treatment). Please consult appropriate "
                    f"clinical resources for this topic."
                )

        has_in_scope = any(kw in query_lower for kw in SafetyFilter.IN_SCOPE_KEYWORDS)
        if not has_in_scope:
            return False, (
                "Your query does not appear to relate to our available guidelines "
                "(hip fracture management or osteoporosis screening/treatment). "
                "Please rephrase your question to focus on these clinical areas."
            )

        return True, None


class ConfidenceFilter:
    """Post-retrieval confidence filter: checks retrieval quality."""

    DEFAULT_THRESHOLD = 0.55
    DEFAULT_MIN_CHUNKS = 2

    @staticmethod
    def check(
        similarities: List[float],
        threshold: float = DEFAULT_THRESHOLD,
        min_chunks: int = DEFAULT_MIN_CHUNKS,
    ) -> tuple[bool, Optional[str], Optional[float], str]:
        """Validates retrieval confidence."""
        if not similarities:
            return False, "No relevant information found in the guideline corpus.", None, "Insufficient Evidence"

        max_sim = max(similarities)

        if max_sim < threshold:
            return False, (
                f"Retrieval confidence too low (best match: {max_sim:.1%}, "
                f"threshold: {threshold:.1%}). The available guidelines do not "
                f"contain sufficiently relevant information to answer this query."
            ), max_sim, "Insufficient Evidence"

        if len(similarities) < min_chunks:
            return False, (
                f"Only {len(similarities)} relevant section(s) found; we require "
                f"at least {min_chunks} independent sources for a confident "
                f"recommendation."
            ), max_sim, "Low"

        if max_sim >= 0.75:
            level = "High"
        elif max_sim >= 0.60:
            level = "Medium"
        else:
            level = "Low"

        return True, None, max_sim, level


# ===================================================================
# CITATION VALIDATION - NO CHANGES
# ===================================================================

class CitationValidator:
    """Post-generation validator that ensures citations map to retrieved chunks."""

    @staticmethod
    def extract_cited_ids(text: str) -> List[str]:
        """Extract all [chunk_id] references from generated text."""
        return re.findall(r'\[([A-Za-z0-9_\-\.]+(?:_p\d+_c\d+|_parent_\d+)[A-Za-z0-9_\-\.]*)\]', text)

    @staticmethod
    def validate(
        generated_text: str,
        valid_chunk_ids: set[str],
    ) -> tuple[str, List[str], List[str]]:
        """Validates citations in generated text."""
        cited_ids = CitationValidator.extract_cited_ids(generated_text)
        valid_found = [cid for cid in cited_ids if cid in valid_chunk_ids]
        invalid_found = [cid for cid in cited_ids if cid not in valid_chunk_ids]

        return generated_text, valid_found, invalid_found

    @staticmethod
    def verify_claims(response_text: str, retrieved_chunks: List[dict]) -> dict:
        """Verify each claim against retrieved evidence."""
        chunks_text = " ".join([c.get("content", "") for c in retrieved_chunks])
        
        claims = response_text.split(". ")
        verified = 0
        
        for claim in claims:
            if any(word in chunks_text for word in claim.split()[:5]):
                verified += 1
        
        return {
            "total_claims": len(claims),
            "verified_claims": verified,
            "faithfulness": verified / len(claims) if claims else 0
        }


# ===================================================================
# LLM GENERATION WITH OPTIMIZATIONS
# ===================================================================

class GroundedLLMGenerator:
    """
    OPTIMIZED VERSION: Compressed prompt, reduced tokens, better efficiency.
    Uses Groq LLaMA 3.3 70B via OpenAI-compatible client.
    """

    # ✨ OPTIMIZED SYSTEM PROMPT (67% smaller than original)
    SYSTEM_PROMPT = """You are a clinical decision support assistant. Output ONLY valid JSON.

REQUIRED JSON FORMAT:
{{
    "recommendation": "Evidence-grounded clinical recommendation from excerpts only",
    "supporting_evidence": [
        {{"claim": "...", "excerpt": "...", "chunk_id": "..."}}
    ],
    "is_answerable": true,
    "refusal_reason": null
}}

CRITICAL RULES:
1. Use ONLY information from provided excerpts - NO external knowledge
2. Reference chunk IDs in square brackets: [chunk_id]
3. Return is_answerable=false with refusal_reason if excerpts insufficient
4. Target population MUST be mentioned
5. Ground every claim in evidence

Guideline excerpts:
{context}

Output JSON only, no markdown formatting."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",  # ✅ FIXED: WAS llama-3.1-8b-instant
        base_url: str = "https://api.groq.com/openai/v1",
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    def _format_context(self, retrieved_chunks: List[dict]) -> str:
        """
        FORMAT CHUNKS WITH TRUNCATION to reduce tokens.
        Truncates content to 200 chars per chunk (save ~300-500 tokens).
        """
        context_parts = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadatas", {})
            doc = meta.get("document_name", "Unknown")
            section = meta.get("section_number", "0.0")
            title = meta.get("section_title", "")
            grade = meta.get("evidence_grade", "N/A")
            population = meta.get("target_population", "Adults")
            page = meta.get("page_number", "?")
            chunk_id = chunk.get("chunk_id", "unknown")
            content = chunk.get("content", "")
            
            # ✨ TRUNCATE CONTENT to 200 chars max
            if len(content) > 200:
                content = content[:200].rsplit(" ", 1)[0] + "..."

            context_parts.append(
                f"[{chunk_id}] {doc} § {section} (p.{page}) | Grade: {grade}\n"
                f"Population: {population}\n"
                f"{content}"
            )

        return "\n\n" + ("\n" + "─" * 50 + "\n").join(context_parts) + "\n"

    def generate(
        self,
        query: str,
        retrieved_chunks: List[dict],
        temperature: float = 0.15,
        max_tokens: int = 512,  # ✅ REDUCED from 1024 (50% savings)
    ) -> dict:
        """
        Generate grounded response with optimized parameters.
        """
        context = self._format_context(retrieved_chunks)
        system_prompt = self.SYSTEM_PROMPT.format(context=context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )

        raw_text = response.choices[0].message.content.strip()

        # Parse JSON from the LLM response
        try:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(raw_text)
            return parsed
        except json.JSONDecodeError:
            return {
                "recommendation": raw_text,
                "supporting_evidence": [],
                "is_answerable": True,
                "_parse_fallback": True,
            }


# ===================================================================
# MAIN RAG PIPELINE - OPTIMIZED
# ===================================================================

class RAGPipeline:
    """
    Optimized end-to-end RAG pipeline.
    Changes:
    - Reduced top_k from 5 to 3 chunks (save ~1000 tokens)
    - Compressed system prompt (save ~400 tokens)
    - Reduced max_tokens from 1024 to 512 (save ~200 tokens)
    - Truncated chunk content (save ~300-500 tokens)
    Total: ~70% token reduction per request
    """

    def __init__(
        self,
        chroma_path: str = "data/chroma_db",
        collection_name: str = "clinical-guidelines",
        embeddings_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        groq_api_key: str = None,
        llm_model: str = "llama-3.3-70b-versatile",  # ✅ FIXED
        groq_base_url: str = "https://api.groq.com/openai/v1",
    ):
        # Load ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(collection_name)

        # Load embedding model
        self.embedding_model = SentenceTransformer(embeddings_model)

        # Initialize LLM
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.llm = GroundedLLMGenerator(
            api_key=api_key,
            model=llm_model,
            base_url=groq_base_url,
        )

        # Filters
        self.safety_filter = SafetyFilter()
        self.confidence_filter = ConfidenceFilter()
        self.citation_validator = CitationValidator()

        print(f"[+] RAG Pipeline initialized (model: {llm_model}, optimized version)")

    def answer_query(
        self,
        query: str,
        top_k: int = 3,  # ✅ REDUCED from 5 to 3 chunks
        confidence_threshold: float = 0.55,
        min_chunks: int = 2,
    ) -> GroundedResponse:
        """
        Answer a clinical query end-to-end with optimizations.
        """

        # ─── Step 1: SAFETY CHECK ───
        is_safe, safety_reason = self.safety_filter.check(query)
        if not is_safe:
            return GroundedResponse(
                query=query,
                status="REFUSED",
                refusal_reason=safety_reason,
                confidence_level="Insufficient Evidence",
                model_used=self.llm.model,
            )

        # ─── Step 2: RETRIEVAL ───
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        similarities = [1.0 - dist for dist in results["distances"][0]]

        # ─── Step 3: CONFIDENCE CHECK ───
        passes, conf_reason, max_score, conf_level = self.confidence_filter.check(
            similarities,
            threshold=confidence_threshold,
            min_chunks=min_chunks,
        )

        if not passes:
            context_found = None
            if results["documents"][0]:
                top_doc = results["metadatas"][0][0].get("document_name", "Unknown")
                top_section = results["metadatas"][0][0].get("section_title", "Unknown")
                context_found = (
                    f"Best match: {top_doc} — {top_section} "
                    f"(similarity: {max_score:.1%})" if max_score else "No matches"
                )

            return GroundedResponse(
                query=query,
                status="REFUSED",
                refusal_reason=conf_reason,
                confidence_level="Insufficient Evidence",
                context_found=context_found,
                context_lacking=(
                    "Sufficiently relevant clinical guideline sections "
                    f"(threshold: {confidence_threshold:.0%})"
                ),
                retrieval_scores=similarities,
                model_used=self.llm.model,
            )

        # ─── Step 4: FORMAT RETRIEVED CHUNKS (OPTIMIZED) ───
        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            retrieved_chunks.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadatas": results["metadatas"][0][i],
                "similarity": similarities[i],
            })

        valid_chunk_ids = {chunk["chunk_id"] for chunk in retrieved_chunks}

        # ─── Step 5: LLM GENERATION ───
        llm_output = self.llm.generate(query, retrieved_chunks)

        # ─── Step 6: HANDLE LLM REFUSAL ───
        if not llm_output.get("is_answerable", True):
            return GroundedResponse(
                query=query,
                status="REFUSED",
                refusal_reason=llm_output.get(
                    "refusal_reason",
                    "The LLM determined the guidelines do not contain "
                    "sufficient information to answer this query."
                ),
                confidence_level="Insufficient Evidence",
                retrieval_scores=similarities,
                model_used=self.llm.model,
            )

        # ─── Step 7: CITATION VALIDATION ───
        recommendation_text = llm_output.get("recommendation", "")
        _, valid_cites, invalid_cites = self.citation_validator.validate(
            recommendation_text, valid_chunk_ids
        )

        if invalid_cites:
            print(f"[!] Warning: {len(invalid_cites)} invalid citation(s) detected: {invalid_cites}")

        # ─── Step 8: BUILD SUPPORTING EVIDENCE ───
        supporting_evidence = []
        for ev in llm_output.get("supporting_evidence", []):
            ev_chunk_id = ev.get("chunk_id", "")
            if ev_chunk_id in valid_chunk_ids or not ev_chunk_id:
                supporting_evidence.append(SupportingEvidence(
                    claim=ev.get("claim", ""),
                    excerpt=ev.get("excerpt", ""),
                    chunk_id=ev_chunk_id if ev_chunk_id in valid_chunk_ids else retrieved_chunks[0]["chunk_id"],
                ))

        # ─── Step 9: BUILD CITATIONS ───
        citations = []
        for chunk in retrieved_chunks:
            meta = chunk["metadatas"]
            citations.append(Citation(
                document_name=meta.get("document_name", "Unknown"),
                section_number=meta.get("section_number", "0.0"),
                section_title=meta.get("section_title", ""),
                page_number=int(meta.get("page_number", 1)),
                chunk_id=chunk["chunk_id"],
                evidence_grade=meta.get("evidence_grade", "N/A"),
                target_population=meta.get("target_population", "Adults"),
                retrieved_text=chunk["content"][:400],
                similarity_score=chunk["similarity"],
            ))

        # ─── Step 10: ASSEMBLE FINAL RESPONSE ───
        return GroundedResponse(
            query=query,
            status="SUCCESS",
            recommendation=recommendation_text,
            supporting_evidence=supporting_evidence,
            citations=citations,
            confidence_level=conf_level,
            retrieval_scores=similarities,
            model_used=self.llm.model,
        )


# ===================================================================
# DEMO / STANDALONE USAGE
# ===================================================================

if __name__ == "__main__":
    import sys

    api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("GROQ_API_KEY")

    pipeline = RAGPipeline(
        chroma_path="data/chroma_db",
        groq_api_key=api_key,
    )

    test_queries = [
        "What is the recommended analgesia for hip fracture patients upon admission?",
        "Should all elderly women get osteoporosis screening?",
        "What treatment do you recommend for COVID-19?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        response = pipeline.answer_query(query, top_k=3)  # ✅ Using 3 chunks instead of 5
        print(response.format_for_display())
        print("\n--- JSON ---")
        print(response.to_json())
