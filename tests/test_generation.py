"""
Day 3: Generation Layer End-to-End Test
=========================================
Tests the complete RAG pipeline: safety → retrieval → generation → citation.

Runs 5 benchmark queries across all categories:
  - In-scope direct (should answer with citations)
  - In-scope ambiguous (should answer cautiously)
  - Out-of-scope (should refuse)

Saves results to data/generation_test_results.json

Usage:
    python -m tests.test_generation
    # or with explicit API key:
    python -m tests.test_generation gsk_your_key_here
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation_layer import RAGPipeline, GroundedResponse


# ─── Test Queries ───────────────────────────────────────────────────

TEST_QUERIES = [
    {
        "id": "gen_q001",
        "query": "What is the recommended analgesia for hip fracture patients upon admission?",
        "category": "in_scope_direct",
        "expected_status": "SUCCESS",
        "expected_min_confidence": "Medium",
        "notes": "Should reference NICE CG124 § 1.3 pain management",
    },
    {
        "id": "gen_q002",
        "query": "Should all elderly women get osteoporosis screening?",
        "category": "in_scope_ambiguous",
        "expected_status": "SUCCESS",
        "expected_min_confidence": "Medium",
        "notes": "Should reference NICE osteoporosis guidelines on risk assessment",
    },
    {
        "id": "gen_q003",
        "query": "What imaging is recommended if a hip fracture is suspected but initial X-rays are negative?",
        "category": "in_scope_direct",
        "expected_status": "SUCCESS",
        "expected_min_confidence": "Medium",
        "notes": "Should reference MRI as first-line for occult fracture",
    },
    {
        "id": "gen_q004",
        "query": "What treatment do you recommend for COVID-19?",
        "category": "out_of_scope",
        "expected_status": "REFUSED",
        "expected_min_confidence": "Insufficient Evidence",
        "notes": "Must refuse — COVID is not in our guideline scope",
    },
    {
        "id": "gen_q005",
        "query": "What type of surgical procedure is recommended for subtrochanteric fractures?",
        "category": "in_scope_direct",
        "expected_status": "SUCCESS",
        "expected_min_confidence": "Medium",
        "notes": "Should reference intramedullary nail from NICE CG124 § 1.6",
    },
]


def run_test(pipeline: RAGPipeline, test_case: dict) -> dict:
    """Run a single test case and return results."""
    print(f"\n{'─' * 74}")
    print(f"  TEST: {test_case['id']} ({test_case['category']})")
    print(f"  QUERY: {test_case['query']}")
    print(f"{'─' * 74}")

    start = time.time()
    response = pipeline.answer_query(
        query=test_case["query"],
        top_k=5,
        confidence_threshold=0.55,
    )
    latency = time.time() - start

    # Display response
    print(response.format_for_display())

    # Validate expectations
    status_match = response.status == test_case["expected_status"]
    print(f"  ✓ Status match: {status_match} (got: {response.status}, expected: {test_case['expected_status']})")

    if response.status == "SUCCESS":
        has_citations = len(response.citations) > 0
        has_evidence = len(response.supporting_evidence) > 0
        has_recommendation = response.recommendation is not None and len(response.recommendation) > 0
        print(f"  ✓ Has recommendation: {has_recommendation}")
        print(f"  ✓ Has citations: {has_citations} ({len(response.citations)} citations)")
        print(f"  ✓ Has supporting evidence: {has_evidence} ({len(response.supporting_evidence)} items)")
        print(f"  ✓ Confidence level: {response.confidence_level}")
    else:
        print(f"  ✓ Refusal reason: {response.refusal_reason[:100]}...")

    print(f"  ⏱ Latency: {latency:.2f}s")

    return {
        "test_id": test_case["id"],
        "query": test_case["query"],
        "category": test_case["category"],
        "expected_status": test_case["expected_status"],
        "actual_status": response.status,
        "status_match": status_match,
        "confidence_level": response.confidence_level,
        "num_citations": len(response.citations),
        "num_evidence": len(response.supporting_evidence),
        "recommendation_length": len(response.recommendation) if response.recommendation else 0,
        "refusal_reason": response.refusal_reason,
        "latency_seconds": round(latency, 3),
        "retrieval_scores": response.retrieval_scores,
        "full_response_json": json.loads(response.to_json()),
    }


def main():
    print("=" * 74)
    print("  DAY 3: GENERATION LAYER END-TO-END TEST")
    print("=" * 74)

    # Get API key
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[!] Error: No API key. Set GROQ_API_KEY env var or pass as argument.")
        sys.exit(1)

    # Initialize pipeline
    print("\n[+] Initializing RAG Pipeline...")
    pipeline = RAGPipeline(
        chroma_path="data/chroma_db",
        groq_api_key=api_key,
    )

    # Run all tests
    results = []
    for test_case in TEST_QUERIES:
        result = run_test(pipeline, test_case)
        results.append(result)

    # Summary
    print("\n" + "=" * 74)
    print("  TEST SUMMARY")
    print("=" * 74)

    total = len(results)
    passed = sum(1 for r in results if r["status_match"])
    avg_latency = sum(r["latency_seconds"] for r in results) / total

    print(f"\n  Passed: {passed}/{total}")
    print(f"  Avg Latency: {avg_latency:.2f}s")

    for r in results:
        icon = "✅" if r["status_match"] else "❌"
        print(f"  {icon} {r['test_id']} ({r['category']}): "
              f"status={r['actual_status']}, "
              f"confidence={r['confidence_level']}, "
              f"citations={r['num_citations']}, "
              f"latency={r['latency_seconds']:.2f}s")

    # Save results
    output_path = "data/generation_test_results.json"
    os.makedirs(Path(output_path).parent, exist_ok=True)

    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": pipeline.llm.model,
        "total_tests": total,
        "passed": passed,
        "avg_latency_seconds": round(avg_latency, 3),
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[+] Results saved to: {output_path}")
    print("=" * 74)


if __name__ == "__main__":
    main()
