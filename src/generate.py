"""
generate.py
-----------
Step 5 of the RAG pipeline (retrieval -> generation).

Takes a user question, retrieves top_k chunks via retrieve.py, assembles
a grounded prompt, and sends it to Claude via the Anthropic API to
produce a final, cited answer with evidence and citations.

Run:
    python src/generate.py "What is the surgical timing for pilon fractures?"
"""

import os
import sys

import anthropic

from retrieve import Retriever, build_prompt, DEFAULT_TOP_K

# API configuration
ANTHROPIC_API_KEY = "حطو ال API بتاع كلود هنا"
MODEL = "claude-sonnet-4-6"  # swap for whichever Claude model your API access includes


def answer_question(query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Retrieve top-K chunks, generate a cited answer, and return
    the full result with evidence and citations."""
    retriever = Retriever()
    hits = retriever.retrieve(query, top_k=top_k)
    prompt = build_prompt(query, hits)

    # Use provided key, fall back to env var
    api_key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # Fallback: return evidence without LLM generation
        print("[!] No ANTHROPIC_API_KEY found. Showing retrieved evidence only.\n")
        return {
            "query": query,
            "answer": None,
            "hits": hits,
            "prompt": prompt,
        }

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    answer_text = response.content[0].text

    return {
        "query": query,
        "answer": answer_text,
        "hits": hits,
        "prompt": prompt,
    }


def display_result(result: dict):
    """Pretty-print the RAG result with evidence and citations."""
    print("=" * 80)
    print(f"QUERY: {result['query']}")
    print("=" * 80)

    # Show retrieved evidence
    print(f"\n--- Retrieved Evidence (Top {len(result['hits'])} chunks) ---\n")
    for h in result["hits"]:
        m = h["metadata"]
        print(f"  [Source {h['rank']}] similarity={h['similarity']:.4f}")
        print(f"    Chunk ID:   {h['chunk_id']}")
        print(f"    Guideline:  {m['document_code']} - {m['document_name']}")
        print(f"    Section:    §{m['section_number']} {m['section_title']}")
        print(f"    Page:       {m['page_number']}")
        print(f"    Population: {m['target_population']}")
        print(f"    Evidence:   {m['evidence_grade']}")
        print(f"    Content:    {h['content'][:150].strip()}...")
        print()

    # Show generated answer
    if result["answer"]:
        print("--- Generated Answer (with citations) ---\n")
        print(result["answer"])
    else:
        print("--- No LLM answer generated (API key missing) ---")
        print("Full prompt was assembled — set ANTHROPIC_API_KEY to generate.")

    print("\n" + "=" * 80)


def main():
    query = " ".join(sys.argv[1:]) or "What is the surgical timing for hip fracture surgery?"
    result = answer_question(query)
    display_result(result)


if __name__ == "__main__":
    main()
