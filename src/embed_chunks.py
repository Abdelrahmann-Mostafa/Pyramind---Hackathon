"""
Day 1: Embedding Generation
============================
Loads child chunks from processed_chunks.json and generates dense embeddings
using S-PubMedBert. Only children are embedded for vector search.
Outputs chunk_embeddings.json with embeddings attached.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"


def load_child_chunks(input_file: str = "data/processed_chunks.json") -> List[Dict[str, Any]]:
    """Load only child chunks (is_parent=False) from the processed chunks file."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {input_file}. Run ingestion.py first.")
    with open(input_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    return [c for c in all_chunks if not c.get("is_parent", False)]


def embed_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Generate normalized embeddings for each chunk's embedding_text field.
    Returns a copy of chunks with embedding, embedding_model, embedding_dim added.
    """
    print(f"[+] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [chunk["embedding_text"] for chunk in chunks]
    print(f"[+] Embedding {len(texts)} child chunks (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    )

    embedded_chunks = []
    for chunk, vector in zip(chunks, embeddings):
        new_chunk = chunk.copy()
        new_chunk["embedding"] = vector.tolist()
        new_chunk["embedding_model"] = model_name
        new_chunk["embedding_dim"] = len(vector)
        embedded_chunks.append(new_chunk)

    return embedded_chunks


def run_embedding_pipeline(
    input_file: str = "data/processed_chunks.json",
    output_file: str = "data/chunk_embeddings.json",
    model_name: str = MODEL_NAME,
) -> List[Dict[str, Any]]:
    """Execute the embedding pipeline over child chunks."""
    os.makedirs(Path(output_file).parent, exist_ok=True)

    print("=" * 80)
    print("CHUNK EMBEDDING PIPELINE")
    print("=" * 80)

    chunks = load_child_chunks(input_file)
    print(f"[+] Loaded {len(chunks)} child chunks from {input_file}")

    embedded_chunks = embed_chunks(chunks, model_name=model_name)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE: Embedded {len(embedded_chunks)} child chunks with '{model_name}'.")
    print(f"Output saved to: {output_file}")
    print("=" * 80)

    return embedded_chunks


if __name__ == "__main__":
    run_embedding_pipeline()