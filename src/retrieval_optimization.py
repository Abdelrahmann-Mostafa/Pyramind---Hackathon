"""
Day 2: Retrieval Optimization & Evaluation
===========================================
Implements ChromaDB indexing, configurable retrieval with compression,
evaluation against benchmark queries, and hyperparameter tuning.

Usage:
    from src.retrieval_optimization import create_chroma_index, retrieve_with_config, run_evaluation
"""

import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
CHROMA_PATH = "data/chroma_db"
EMBEDDINGS_FILE = "data/chunk_embeddings.json"
BENCHMARK_FILE = "data/benchmark_queries.json"


def create_chroma_index(
    embeddings_file: str = EMBEDDINGS_FILE,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = "clinical-guidelines",
) -> chromadb.Collection:
    """
    Create or overwrite a ChromaDB collection with child chunk embeddings.
    Returns the collection object.
    """
    if not Path(embeddings_file).exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}. Run embed_chunks.py first.")

    with open(embeddings_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client = chromadb.PersistentClient(path=chroma_path)
    # Delete existing collection if any
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [c["chunk_id"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["content"] for c in chunks]
    # Metadata: exclude embedding and content
    metadatas = []
    for c in chunks:
        meta = {k: v for k, v in c.items() if k not in ["embedding", "content", "embedding_model", "embedding_dim"]}
        # Ensure all values are strings, numbers, or bools (ChromaDB accepts)
        for k, v in meta.items():
            if not isinstance(v, (str, int, float, bool)):
                meta[k] = str(v)
        metadatas.append(meta)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"[+] ChromaDB index created: {collection.count()} vectors in '{collection_name}'")
    return collection


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the embedding model."""
    return SentenceTransformer(model_name)


def retrieve_with_config(
    query: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    k: int = 3,
    compress: bool = False,
    top_n_sentences: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks for a query with optional contextual compression.

    Returns a list of results with metadata and similarity scores.
    Each result includes 'parent_chunk' if a parent_id exists (loaded from processed_chunks.json).
    """
    # Encode query
    query_embedding = embedding_model.encode([query], normalize_embeddings=True).tolist()

    # Search
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    # Convert distances to similarities (cosine distance -> similarity)
    similarities = [1.0 - dist for dist in results["distances"][0]]

    # Build result list
    retrieved = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        content = results["documents"][0][i]
        # Convert similarity to float
        sim = similarities[i]

        # Load parent chunk if parent_id exists
        parent_chunk = None
        parent_id = meta.get("parent_id")
        if parent_id:
            # We need to load parent chunks from processed_chunks.json
            # For efficiency, we could cache, but here we load on demand (simple)
            # We'll implement a helper that loads all parents once and caches.
            parent_chunk = _load_parent_chunk(parent_id)

        result = {
            "rank": i + 1,
            "chunk_id": results["ids"][0][i],
            "content": content,
            "similarity_score": sim,
            "parent_chunk": parent_chunk,
            "document_name": meta.get("document_name", "Unknown"),
            "section_number": meta.get("section_number", "0.0"),
            "section_title": meta.get("section_title", ""),
            "page_number": int(meta.get("page_number", 1)),
            "evidence_grade": meta.get("evidence_grade", "N/A"),
            "target_population": meta.get("target_population", "Adults"),
        }
        retrieved.append(result)

    # Optional compression
    if compress:
        retrieved = compress_context(query, retrieved, top_n_sentences)

    return retrieved


# Cache for parent chunks (load once)
_PARENT_CACHE = None

def _load_parent_chunk(parent_id: str) -> Optional[Dict[str, Any]]:
    """Load parent chunk by ID from processed_chunks.json, with caching."""
    global _PARENT_CACHE
    if _PARENT_CACHE is None:
        try:
            with open("data/processed_chunks.json", "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
            _PARENT_CACHE = {c["chunk_id"]: c for c in all_chunks if c.get("is_parent", False)}
        except FileNotFoundError:
            _PARENT_CACHE = {}
    return _PARENT_CACHE.get(parent_id)


def compress_context(query: str, chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Compress each chunk by keeping only the top-N sentences most relevant to the query.
    Uses the embedding model to score sentences.
    """
    # Load model if not already loaded globally
    global _COMPRESSION_MODEL
    if "_COMPRESSION_MODEL" not in globals():
        _COMPRESSION_MODEL = load_embedding_model()

    model = _COMPRESSION_MODEL
    query_vec = model.encode([query], normalize_embeddings=True)

    compressed = []
    for chunk in chunks:
        text = chunk["content"]
        # Split into sentences (simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 1:
            compressed.append(chunk)
            continue

        # Encode all sentences
        sent_vectors = model.encode(sentences, normalize_embeddings=True)
        # Compute cosine similarities with query
        sims = np.dot(sent_vectors, query_vec.T).flatten()
        # Get top indices
        top_indices = np.argsort(sims)[-top_n:][::-1]
        top_sentences = [sentences[i] for i in sorted(top_indices)]
        compressed_text = ". ".join(top_sentences)
        # Update content but keep other fields
        new_chunk = chunk.copy()
        new_chunk["content"] = compressed_text
        compressed.append(new_chunk)

    return compressed


# Global cache for compression model
_COMPRESSION_MODEL = None
import re  # for sentence splitting


def evaluate_retrieval(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    benchmark_file: str = BENCHMARK_FILE,
    k: int = 3,
    compress: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate retrieval performance against benchmark queries.
    Returns metrics: precision@k, recall@k, MRR, avg_latency.
    """
    with open(benchmark_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    total_queries = len(queries)
    precision_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0
    latencies = []

    for q in queries:
        qid = q["query_id"]
        query_text = q["query"]
        ground_truth = set(q["ground_truth_chunk_ids"])
        if not ground_truth:
            continue  # skip queries without ground truth

        start = time.time()
        results = retrieve_with_config(query_text, collection, embedding_model, k=k, compress=compress)
        latency = time.time() - start
        latencies.append(latency)

        retrieved_ids = [r["chunk_id"] for r in results]
        relevant_retrieved = set(retrieved_ids) & ground_truth

        precision = len(relevant_retrieved) / k
        recall = len(relevant_retrieved) / len(ground_truth) if ground_truth else 0.0
        precision_sum += precision
        recall_sum += recall

        # MRR: reciprocal rank of first relevant
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in ground_truth:
                mrr = 1.0 / (i + 1)
                break
        mrr_sum += mrr

    avg_precision = precision_sum / total_queries
    avg_recall = recall_sum / total_queries
    avg_mrr = mrr_sum / total_queries
    avg_latency = np.mean(latencies) if latencies else 0.0

    return {
        "k": k,
        "compression": compress,
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "mrr": avg_mrr,
        "avg_latency_seconds": avg_latency,
        "total_queries": total_queries,
    }


def test_hyperparameters(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    benchmark_file: str = BENCHMARK_FILE,
) -> List[Dict[str, Any]]:
    """
    Test combinations of k and compression, return results list.
    """
    k_values = [1, 3, 5, 10]
    compress_values = [False, True]
    results = []

    for k in k_values:
        for compress in compress_values:
            print(f"Testing k={k}, compress={compress}...")
            metrics = evaluate_retrieval(collection, embedding_model, benchmark_file, k=k, compress=compress)
            results.append({
                "config": {"k": k, "compression": compress},
                "results": metrics
            })

    return results


def run_evaluation_pipeline(
    embeddings_file: str = EMBEDDINGS_FILE,
    chroma_path: str = CHROMA_PATH,
    benchmark_file: str = BENCHMARK_FILE,
    output_log: str = "data/retrieval_metrics_log.json",
) -> Dict[str, Any]:
    """
    Run the full Day 2 evaluation: index ChromaDB, test hyperparameters, log results.
    """
    print("=" * 80)
    print("RETRIEVAL OPTIMIZATION & EVALUATION")
    print("=" * 80)

    # Create index
    collection = create_chroma_index(embeddings_file, chroma_path)
    embedding_model = load_embedding_model()

    # Load benchmark queries
    with open(benchmark_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"[+] Loaded {len(queries)} benchmark queries.")

    # Test hyperparameters
    config_results = test_hyperparameters(collection, embedding_model, benchmark_file)

    # Find optimal config (max precision, then recall)
    best = max(config_results, key=lambda x: (x["results"]["precision_at_k"], x["results"]["recall_at_k"]))
    optimal_config = best["config"]
    optimal_metrics = best["results"]

    # Prepare full log
    log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_chunks_indexed": collection.count(),
        "embedding_model": MODEL_NAME,
        "embedding_dim": 768,
        "database": "ChromaDB",
        "test_suite_size": len(queries),
        "configurations_tested": config_results,
        "optimal_config": {
            "config": optimal_config,
            "reasoning": f"Highest precision ({optimal_metrics['precision_at_k']:.1%}) and recall ({optimal_metrics['recall_at_k']:.1%})",
            "metrics": optimal_metrics,
        }
    }

    # Save log
    os.makedirs(Path(output_log).parent, exist_ok=True)
    with open(output_log, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE. Results saved to: {output_log}")
    print(f"Optimal config: k={optimal_config['k']}, compression={optimal_config.get('compression', False)}")
    print(f"  Precision@{optimal_config['k']}: {optimal_metrics['precision_at_k']:.1%}")
    print(f"  Recall@{optimal_config['k']}: {optimal_metrics['recall_at_k']:.1%}")
    print(f"  MRR: {optimal_metrics['mrr']:.2f}")
    print("=" * 80)

    return log


if __name__ == "__main__":
    run_evaluation_pipeline()