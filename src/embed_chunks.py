"""
Embedding Layer (src/embed_chunks.py)
--------------------------------------
Loads section-aware chunks produced by ingestion.py (data/processed_chunks.json)
and generates dense vector embeddings using a domain-specific biomedical
sentence embedding model.

Model chosen: pritamdeka/S-PubMedBert-MS-MARCO
------------------------------------------------
- Pretrained from scratch on PubMed abstracts/full text (PubMedBERT), so it
  already understands clinical vocabulary relevant to these guidelines
  (DXA, FRAX, bisphosphonate, postmenopausal, etc.) rather than treating
  them as rare/out-of-vocabulary tokens the way a general-purpose model would.
- Fine-tuned on MS MARCO-style query/passage pairs adapted to biomedical
  text, so it is optimized specifically for retrieval (matching a clinical
  question to the passage that answers it) rather than generic sentence
  similarity — a direct match for this RAG use case.
- Runs locally via sentence-transformers: no API key, no per-token cost,
  deterministic output, and no data leaves this machine.
- Fixed, versioned HF model ID, so the embedding space stays stable across
  ingestion runs (mixing embedding models across chunks silently breaks
  retrieval quality).

Output: data/chunk_embeddings.json — each chunk's original metadata plus
its embedding vector, keyed by chunk_id.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"


def load_chunks(input_file: str = "data/processed_chunks.json") -> List[Dict[str, Any]]:
    """Loads section-aware chunks produced by the ingestion pipeline."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {input_file}. Run ingestion.py first."
        )
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Generates dense embeddings for each chunk's content field and attaches
    the resulting vector to the chunk's existing metadata.
    """
    print(f"[+] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [chunk["content"] for chunk in chunks]
    print(f"[+] Embedding {len(texts)} chunks (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    )

    embedded_chunks = []
    for chunk, vector in zip(chunks, embeddings):
        embedded_chunks.append({
            **chunk,
            "embedding": vector.tolist(),
            "embedding_model": model_name,
            "embedding_dim": len(vector),
        })

    return embedded_chunks


def run_embedding_pipeline(
    input_file: str = "data/processed_chunks.json",
    output_file: str = "data/chunk_embeddings.json",
    model_name: str = MODEL_NAME,
) -> List[Dict[str, Any]]:
    """Executes the end-to-end embedding pipeline over processed chunks."""
    os.makedirs(Path(output_file).parent, exist_ok=True)

    print("=" * 80)
    print("CHUNK EMBEDDING PIPELINE")
    print("=" * 80)

    chunks = load_chunks(input_file)
    print(f"[+] Loaded {len(chunks)} chunks from {input_file}")

    embedded_chunks = embed_chunks(chunks, model_name=model_name)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE: Embedded {len(embedded_chunks)} chunks with '{model_name}'.")
    print(f"Output saved to: {output_file}")
    print("=" * 80)

    return embedded_chunks


if __name__ == "__main__":
    run_embedding_pipeline()
