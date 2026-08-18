"""
embed_and_index.py
------------------
Step 3 of the RAG pipeline.

Reads data/processed_chunks.json (from chunking.py), encodes each
chunk's `embedding_text` field using the PubMedBERT model fine-tuned
for MS-MARCO passage retrieval, and stores everything in a persistent
ChromaDB collection.

Model:  pritamdeka/S-PubMedBert-MS-MARCO
Vector DB:  ChromaDB (persistent, at data/chroma_db/)

Output: data/chroma_db/  (persistent ChromaDB directory)

Run:
    python src/embed_and_index.py
"""

import json
import shutil
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "processed_chunks.json"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "clinical_guidelines"
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
BATCH_SIZE = 64  # batch size for embedding + insertion


def main():
    # ---- Load chunks ------------------------------------------------
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found — run src/chunking.py first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[+] Loaded {len(chunks)} chunks from {INPUT_FILE}")

    # ---- Load embedding model ---------------------------------------
    print(f"[+] Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"[+] Model loaded  (embedding dim = {model.get_sentence_embedding_dimension()})")

    # ---- Prepare ChromaDB -------------------------------------------
    # Wipe any previous index so we get a clean rebuild every time
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"[+] Removed old ChromaDB at {CHROMA_DIR}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )
    print(f"[+] Created ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}")

    # ---- Embed & insert in batches ----------------------------------
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = chunks[start:end]

        # Text to embed — uses the enriched embedding_text field
        texts = [c["embedding_text"] for c in batch]

        # Encode
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # Prepare ChromaDB fields
        ids = [c["chunk_id"] for c in batch]
        documents = [c["content"] for c in batch]
        metadatas = [
            {
                "document_name": c["document_name"],
                "document_code": c["document_code"],
                "section_number": c["section_number"],
                "section_title": c["section_title"],
                "page_number": c["page_number"],
                "target_population": c["target_population"],
                "evidence_grade": c["evidence_grade"],
                "char_count": c["char_count"],
            }
            for c in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"    embedded & indexed chunks {start+1}-{end} / {total}")

    # ---- Summary ----------------------------------------------------
    print(f"\n[+] Done! Indexed {collection.count()} chunks into '{COLLECTION_NAME}'")
    print(f"[+] ChromaDB persisted at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
