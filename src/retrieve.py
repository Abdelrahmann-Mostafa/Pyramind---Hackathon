"""
retrieve.py
-----------
Step 4 of the RAG pipeline.

Loads the persistent ChromaDB collection and the S-PubMedBERT model,
and exposes a `retrieve()` function for semantic search plus a
`build_prompt()` helper that assembles a grounded prompt for an LLM.

TOP_K = 17 — returns the 17 most similar chunks with evidence and
citations for maximum context coverage.

Run directly for a quick CLI test:
    python src/retrieve.py "What is the surgical timing for open fractures?"
"""

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "clinical_guidelines"
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

DEFAULT_TOP_K = 17          # hackathon requirement: retrieve top 17
MIN_TOP_K, MAX_TOP_K = 1, 50


class Retriever:
    def __init__(self):
        print("[+] Loading retrieval model ...")
        self.model = SentenceTransformer(MODEL_NAME)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self.collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found at {CHROMA_DIR}. "
                "Run src/embed_and_index.py first."
            ) from e
        print(f"[+] Collection loaded: {self.collection.count()} chunks indexed")

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K,
                 metadata_filter: dict | None = None) -> list:
        top_k = max(MIN_TOP_K, min(MAX_TOP_K, top_k))
        query_vec = self.model.encode([query], normalize_embeddings=True).tolist()

        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=top_k,
            where=metadata_filter,  # e.g. {"document_code": "NG37"}
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for rank in range(len(results["ids"][0])):
            hits.append({
                "rank": rank + 1,
                "chunk_id": results["ids"][0][rank],
                "content": results["documents"][0][rank],
                "metadata": results["metadatas"][0][rank],
                "similarity": 1.0 - results["distances"][0][rank],
            })
        return hits


def build_prompt(query: str, hits: list) -> str:
    """Assembles a grounded, citation-friendly prompt for the generation step."""
    context_blocks = []
    for h in hits:
        m = h["metadata"]
        context_blocks.append(
            f"[Source {h['rank']} | {m['document_code']} §{m['section_number']} "
            f"{m['section_title']} | p.{m['page_number']} | similarity={h['similarity']:.3f}]\n"
            f"{h['content']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a clinical decision support assistant. Answer the question using
ONLY the guideline excerpts below. Cite the source number(s) you used in
square brackets, e.g. [Source 1]. If the excerpts do not contain enough
information to answer, say so explicitly rather than guessing.

QUESTION:
{query}

GUIDELINE EXCERPTS:
{context}

ANSWER:"""
    return prompt


def main():
    query = " ".join(sys.argv[1:]) or "What is the surgical timing for open fractures?"
    retriever = Retriever()
    hits = retriever.retrieve(query, top_k=DEFAULT_TOP_K)

    print(f"\nQuery: {query}\n" + "=" * 80)
    for h in hits:
        m = h["metadata"]
        print(f"\n[Rank {h['rank']}] similarity={h['similarity']:.4f} | {h['chunk_id']}")
        print(f"  {m['document_code']} §{m['section_number']} {m['section_title']} (p.{m['page_number']})")
        print(f"  {h['content'][:200].strip()}...")

    print("\n" + "=" * 80)
    print("Assembled LLM prompt preview:\n")
    print(build_prompt(query, hits))


if __name__ == "__main__":
    main()
