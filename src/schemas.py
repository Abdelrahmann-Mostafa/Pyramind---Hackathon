import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "raw_pages.json"
OUTPUT_FILE = DATA_DIR / "processed_chunks.json"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 40  # drop tiny trailing fragments (e.g. lone page footers)


def split_text(text: str, chunk_size: int, overlap: int) -> list:
    """Fixed-size sliding-window splitter with overlap.
    Tries to break on whitespace near the boundary to avoid cutting mid-word."""
    text = " ".join(text.split())  # normalize whitespace
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    step = chunk_size - overlap
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # nudge the cut point to the nearest preceding space, if one exists
        # within a small window, so we don't split a word in half
        if end < text_len:
            nudge = text.rfind(" ", start + int(chunk_size * 0.8), end)
            if nudge != -1:
                end = nudge

        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)

        if end >= text_len:
            break
        start += step

    return chunks


def build_chunks(pages: list) -> list:
    all_chunks = []

    for page in pages:
        page_chunks = split_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(page_chunks):
            chunk_id = f"{page['document_code']}_p{page['page_number']:02d}_c{i:02d}"

            evidence_grade = ", ".join(page["recommendation_years"]) or "N/A"

            embedding_text = (
                f"{page['document_name']} ({page['document_code']}) | "
                f"Section {page['section_number']}: {page['section_title']}\n"
                f"{chunk_text}"
            )

            all_chunks.append({
                "chunk_id": chunk_id,
                "document_name": page["document_name"],
                "document_code": page["document_code"],
                "section_number": page["section_number"],
                "section_title": page["section_title"],
                "page_number": page["page_number"],
                "target_population": page["target_population"],
                "evidence_grade": evidence_grade,
                "char_count": len(chunk_text),
                "content": chunk_text,
                "embedding_text": embedding_text,
            })

    return all_chunks


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found — run src/ingestion.py first.")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    chunks = build_chunks(pages)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    sizes = [c["char_count"] for c in chunks]
    print(f"[+] Built {len(chunks)} chunks from {len(pages)} pages")
    print(f"[+] Chunk size stats -> min: {min(sizes)}, max: {max(sizes)}, avg: {sum(sizes)//len(sizes)}")
    print(f"[+] Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
