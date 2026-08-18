"""
Day 3: Generation Layer with Strict Grounding & Citations
==========================================================
Implements the LLM generation, citation formatting, and refusal logic.

Usage:
    from generation import RAGPipeline
    pipeline = RAGPipeline(api_key="sk-...", model="gpt-4-turbo")
    response = pipeline.answer_query(query, top_k=3)
    print(response.format_for_display())
"""

import json
import os
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
import openai

# ===================================================================
# SCHEMA DEFINITIONS (Pydantic)
# ===================================================================

class Citation(BaseModel):
    """Structured citation linking recommendation to source evidence."""
    document_name: str = Field(..., description="PDF filename, e.g., 'NICE_CG124.pdf'")
    section_number: str = Field(..., description="Section ID, e.g., '1.5'")
    section_title: str = Field(..., description="Section heading")
    page_number: int = Field(..., description="Page number (1-indexed)")
    evidence_grade: str = Field(default="N/A", description="Grade A/B/C/D or NICE marker")
    target_population: str = Field(default="Adults", description="Guideline population scope")
    retrieved_text: str = Field(..., description="Direct excerpt from the guideline")
    similarity_score: float = Field(..., description="Retrieval confidence (0.0-1.0)")


class RecommendationWithCitation(BaseModel):
    """A single recommendation grounded in evidence."""
    recommendation_text: str = Field(..., description="The clinical guidance")
    primary_citation: Citation = Field(..., description="Most relevant source")
    supporting_citations: List[Citation] = Field(default_factory=list, description="Additional evidence")


class GroundedResponse(BaseModel):
    """Complete response with status, safety info, and citations."""
    query: str = Field(..., description="Original user query")
    status: Literal["SUCCESS", "REFUSED"] = Field(..., description="Did we answer or refuse?")
    
    # Success case
    recommendations: List[RecommendationWithCitation] = Field(
        default_factory=list, 
        description="Grounded recommendations if status=SUCCESS"
    )
    overall_confidence: Optional[float] = Field(
        default=None, 
        description="Average retrieval confidence"
    )
    
    # Refusal case
    refusal_reason: Optional[str] = Field(
        default=None, 
        description="Why we refused, if status=REFUSED"
    )
    
    # Metadata
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    model_used: str = Field(default="gpt-4-turbo")
    
    def format_for_display(self) -> str:
        """Formats response for HTML/Streamlit display."""
        output = f"""
╔════════════════════════════════════════════════════════════════════╗
║ CLINICAL DECISION SUPPORT RESPONSE                                ║
╚════════════════════════════════════════════════════════════════════╝

QUERY: {self.query}

QUERY STATUS: {"✓ ANSWERED" if self.status == "SUCCESS" else "✗ REFUSED"}
"""
        
        if self.status == "REFUSED":
            output += f"\nREFUSAL REASON:\n{self.refusal_reason}\n"
            return output
        
        # SUCCESS case
        output += f"OVERALL CONFIDENCE: {'🟢 HIGH (>0.70)' if self.overall_confidence > 0.7 else '🟡 MODERATE (0.55-0.70)' if self.overall_confidence > 0.55 else '🔴 LOW'} ({self.overall_confidence:.1%})\n"
        output += "\n" + "─" * 70 + "\n"
        
        for i, rec in enumerate(self.recommendations, 1):
            output += f"\n[RECOMMENDATION #{i}]\n\n{rec.recommendation_text}\n"
            output += f"\n  Primary Source: {rec.primary_citation.document_name} "
            output += f"§ {rec.primary_citation.section_number} "
            output += f"(p. {rec.primary_citation.page_number}) "
            output += f"[{rec.primary_citation.evidence_grade}]\n"
            output += f"  Population:    {rec.primary_citation.target_population}\n"
            output += f"  Confidence:    {rec.primary_citation.similarity_score:.1%}\n"
            
            if rec.supporting_citations:
                output += f"\n  Supporting Evidence:\n"
                for sup in rec.supporting_citations:
                    output += f"    • {sup.document_name} § {sup.section_number} (p. {sup.page_number})\n"
        
        output += "\n" + "─" * 70 + "\n"
        return output


# ===================================================================
# SAFETY & REFUSAL LOGIC
# ===================================================================

class SafetyFilter:
    """Checks query safety before retrieval."""
    
    OUT_OF_SCOPE_KEYWORDS = {
        # Conditions NOT covered by hip fracture + osteoporosis guidelines
        "covid", "coronavirus", "diabetes", "heart", "cardiac", "myocardial",
        "stroke", "neurological", "parkinson", "alzheimer", "cancer", "tumor",
        "oncology", "lung", "breast", "prostate", "mental health", "anxiety",
        "depression", "ocd", "ptsd", "schizophrenia", "bipolar", "pregnancy",
        "obstetric", "gynecology", "asthma", "copd", "respiratory", "infection",
        "bacterial", "viral", "pneumonia", "covid", "gastric", "liver", "kidney",
        "renal", "dialysis", "transplant", "immunology", "autoimmune", "arthritis",
        "rheumatology", "gout", "lupus", "dermatology", "skin", "psoriasis", "eczema"
    }
    
    IN_SCOPE_KEYWORDS = {
        # Conditions we CAN answer about
        "hip fracture", "osteoporosis", "bone", "fracture", "screening",
        "dxa", "dexa", "t-score", "frax", "bisphosphonate", "alendronate",
        "risedronate", "ibandronate", "zoledronic", "hormone replacement",
        "hrt", "calcium", "vitamin d", "elderly", "older", "women", "men",
        "menopause", "postmenopausal", "analgesia", "pain management",
        "surgery", "surgical", "orthopedic", "geriatric"
    }
    
    @staticmethod
    def is_out_of_scope(query: str) -> tuple[bool, Optional[str]]:
        """Check if query is outside guideline scope."""
        query_lower = query.lower()
        
        # Explicit out-of-scope detection
        for keyword in SafetyFilter.OUT_OF_SCOPE_KEYWORDS:
            if keyword in query_lower:
                return True, (
                    f"Your query mentions '{keyword}', which is outside the scope of our "
                    f"medical guidelines (hip fracture management and osteoporosis screening). "
                    f"Please consult appropriate clinical resources for this topic."
                )
        
        # Heuristic: Require at least one in-scope keyword
        has_in_scope = any(kw in query_lower for kw in SafetyFilter.IN_SCOPE_KEYWORDS)
        if not has_in_scope:
            return True, (
                "Your query does not appear to relate to our available guidelines "
                "(hip fracture management or osteoporosis screening). "
                "Please rephrase your question to focus on these topics."
            )
        
        return False, None


class ConfidenceFilter:
    """Checks retrieval confidence before generation."""
    
    CONFIDENCE_THRESHOLD = 0.55  # Minimum acceptable similarity score
    MIN_CHUNKS = 2  # Minimum relevant chunks required
    
    @staticmethod
    def check_retrieval(
        similarities: List[float],
        min_threshold: float = CONFIDENCE_THRESHOLD,
        min_chunks: int = MIN_CHUNKS
    ) -> tuple[bool, Optional[str], Optional[float]]:
        """
        Validates retrieved chunks.
        Returns (passes_check, refusal_reason_if_fails, max_confidence_score)
        """
        if not similarities:
            return False, "No relevant information found in the guidelines.", None
        
        max_sim = max(similarities)
        
        if max_sim < min_threshold:
            return False, (
                f"The retrieved evidence has low confidence ({max_sim:.1%}). "
                f"We cannot confidently answer this query based on our guidelines. "
                f"Please consult clinical experts or primary sources."
            ), max_sim
        
        if len(similarities) < min_chunks:
            return False, (
                f"Only {len(similarities)} relevant section(s) found. "
                f"We need at least {min_chunks} independent sources for confident recommendations."
            ), max_sim
        
        return True, None, max_sim


# ===================================================================
# LLM GENERATION WITH STRICT GROUNDING
# ===================================================================

class GroundedLLMGenerator:
    """Generates recommendations strictly grounded in retrieved evidence."""
    
    SYSTEM_PROMPT = """You are an expert clinical decision support assistant trained on official medical guidelines.

CRITICAL CONSTRAINTS:
1. You MUST ONLY use information from the provided guideline excerpts.
2. You MUST NOT use external knowledge, personal experience, or information not explicitly in the context.
3. Every recommendation MUST be directly traceable to the provided evidence.
4. If a question cannot be answered from the guidelines, you MUST refuse explicitly.
5. Be concise, precise, and cite section numbers when referencing guidelines.
6. Always mention the target population the recommendation applies to.

Guidelines Provided:
{context}

Answer the user's query ONLY based on the above guidelines. If the guidelines don't address it, say so."""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        openai.api_key = api_key
        self.model = model
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[dict]
    ) -> str:
        """Generate grounded response from retrieved chunks."""
        
        # Format context from chunks
        context_parts = []
        for chunk in retrieved_chunks:
            doc = chunk.get("metadatas", {}).get("document_name", "Unknown")
            section = chunk.get("metadatas", {}).get("section_number", "0.0")
            title = chunk.get("metadatas", {}).get("section_title", "")
            content = chunk.get("content", "")
            
            context_parts.append(
                f"[{doc} § {section}: {title}]\n{content}"
            )
        
        full_context = "\n\n---\n\n".join(context_parts)
        
        system_prompt = self.SYSTEM_PROMPT.format(context=full_context)
        
        # Call LLM with low temperature for consistency
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.2,  # Low temp = reproducible answers
            max_tokens=600,
            top_p=0.9
        )
        
        return response["choices"][0]["message"]["content"]


# ===================================================================
# MAIN RAG PIPELINE
# ===================================================================

class RAGPipeline:
    """End-to-end RAG pipeline: safety → retrieval → generation → citation."""
    
    def __init__(
        self,
        chroma_path: str = "data/chroma_db",
        embeddings_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        openai_api_key: str = None,
        llm_model: str = "gpt-4-turbo"
    ):
        # Load Chroma
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection("clinical-guidelines")
        
        # Load embedding model
        self.embeddings_model = SentenceTransformer(embeddings_model)
        
        # Setup LLM
        self.llm = GroundedLLMGenerator(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            model=llm_model
        )
        
        # Safety filters
        self.safety_filter = SafetyFilter()
        self.confidence_filter = ConfidenceFilter()
    
    def answer_query(
        self,
        query: str,
        top_k: int = 3,
        confidence_threshold: float = 0.55
    ) -> GroundedResponse:
        """
        Answers a clinical query end-to-end.
        
        Args:
            query: User's clinical question
            top_k: Number of chunks to retrieve
            confidence_threshold: Minimum similarity score (0.0-1.0)
        
        Returns:
            GroundedResponse with recommendations or refusal
        """
        
        # Step 1: SAFETY CHECK
        is_oob, oob_reason = self.safety_filter.is_out_of_scope(query)
        if is_oob:
            return GroundedResponse(
                query=query,
                status="REFUSED",
                refusal_reason=oob_reason,
                model_used=self.llm.model
            )
        
        # Step 2: RETRIEVAL
        query_embedding = self.embeddings_model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Extract similarities (convert distance to similarity)
        similarities = [
            1.0 - dist  # Cosine distance → cosine similarity
            for dist in results["distances"][0]
        ]
        
        # Step 3: CONFIDENCE CHECK
        passes_conf, conf_reason, max_conf = self.confidence_filter.check_retrieval(
            similarities,
            min_threshold=confidence_threshold
        )
        if not passes_conf:
            return GroundedResponse(
                query=query,
                status="REFUSED",
                refusal_reason=conf_reason,
                model_used=self.llm.model
            )
        
        # Step 4: FORMAT RETRIEVED CHUNKS
        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            retrieved_chunks.append({
                "content": results["documents"][0][i],
                "metadatas": results["metadatas"][0][i],
                "similarity": similarities[i]
            })
        
        # Step 5: GENERATE GROUNDED RESPONSE
        generated_text = self.llm.generate(query, retrieved_chunks)
        
        # Step 6: FORMAT CITATIONS
        primary_citation = Citation(
            document_name=retrieved_chunks[0]["metadatas"].get("document_name", "Unknown"),
            section_number=retrieved_chunks[0]["metadatas"].get("section_number", "0.0"),
            section_title=retrieved_chunks[0]["metadatas"].get("section_title", ""),
            page_number=int(retrieved_chunks[0]["metadatas"].get("page_number", 1)),
            evidence_grade=retrieved_chunks[0]["metadatas"].get("evidence_grade", "N/A"),
            target_population=retrieved_chunks[0]["metadatas"].get("target_population", "Adults"),
            retrieved_text=retrieved_chunks[0]["content"][:300],  # Truncate for display
            similarity_score=retrieved_chunks[0]["similarity"]
        )
        
        supporting_citations = [
            Citation(
                document_name=chunk["metadatas"].get("document_name", "Unknown"),
                section_number=chunk["metadatas"].get("section_number", "0.0"),
                section_title=chunk["metadatas"].get("section_title", ""),
                page_number=int(chunk["metadatas"].get("page_number", 1)),
                evidence_grade=chunk["metadatas"].get("evidence_grade", "N/A"),
                target_population=chunk["metadatas"].get("target_population", "Adults"),
                retrieved_text=chunk["content"][:300],
                similarity_score=chunk["similarity"]
            )
            for chunk in retrieved_chunks[1:]
        ]
        
        # Step 7: BUILD RESPONSE
        recommendation = RecommendationWithCitation(
            recommendation_text=generated_text,
            primary_citation=primary_citation,
            supporting_citations=supporting_citations
        )
        
        return GroundedResponse(
            query=query,
            status="SUCCESS",
            recommendations=[recommendation],
            overall_confidence=max_conf,
            model_used=self.llm.model
        )


# ===================================================================
# DEMO USAGE
# ===================================================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline(
        chroma_path="data/chroma_db",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Test queries
    test_queries = [
        # In-scope
        "What is the recommended analgesia for hip fracture patients upon admission?",
        # In-scope, ambiguous
        "Should all elderly women get osteoporosis screening?",
        # Out-of-scope
        "What treatment do you recommend for COVID-19?"
    ]
    
    for query in test_queries:
        print("\n" + "=" * 80)
        response = pipeline.answer_query(query)
        print(response.format_for_display())
        print("\nJSON Response:")
        print(json.dumps(response.model_dump(indent=2), default=str))
