"""
Ingestion and Section-Aware Chunking Layer (src/ingestion.py)
------------------------------------------------------------
Extracts text from clinical guideline PDFs (NICE CG124 & USPSTF Osteoporosis Screening),
performs deterministic section-aware chunking, captures hierarchical metadata
(document name, section number/title, page number, evidence grade, target population),
and writes structured chunks to data/processed_chunks.json.
"""

import os
import re
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any

from schemas import ProcessedChunk


# ---------------------------------------------------------------------------
# Regex patterns for structural and clinical metadata extraction
# ---------------------------------------------------------------------------

SECTION_HEADER_PATTERN = re.compile(
    r'^(?P<num>\d+\.\d+(\.\d+)?)\s+(?P<title>[A-Za-z0-9\s,\-\(\)\/\:]{3,80})',
    re.MULTILINE
)

# Detects implicit sub-section boundaries on headerless pages
# (e.g. NICE CG124 pages 9-14 that lack numbered headers)
IMPLICIT_BREAK_PATTERNS = [
    re.compile(r'^Why the committee made the (?:2023 )?recommendation', re.I),
    re.compile(r'^(?:\d+)\s+[A-Z][a-z].*(?:effectiveness|fracture|replacement)', re.I),
    re.compile(r'^Update information', re.I),
    re.compile(r'^Context\s*$', re.I),
    re.compile(r'^Finding more information', re.I),
]

USPSTF_GRADE_PATTERN = re.compile(
    r'\b(Grade\s+[A-D]|I\s+statement|Grade\s+I)\b',
    re.IGNORECASE
)

TARGET_POP_PATTERNS = [
    (re.compile(r'women\s+(aged\s+)?65\s+(years\s+)?(and|or)\s+older', re.I), "Women ≥ 65 years"),
    (re.compile(r'postmenopausal\s+women\s+younger\s+than\s+65', re.I), "Postmenopausal women < 65 years with elevated risk"),
    (re.compile(r'\bmen\b', re.I), "Men (Asymptomatic adults)"),
    (re.compile(r'adults\s+(aged\s+)?18\s+and\s+over|hip\s+fracture', re.I), "Adults with acute hip fracture"),
]

# Minimum character count for a chunk to be worth indexing
MIN_CHUNK_CHARS = 100


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def detect_target_population(text: str, default_pop: str = "Adults") -> str:
    """Detects clinical target population from chunk text."""
    for pattern, pop_label in TARGET_POP_PATTERNS:
        if pattern.search(text):
            return pop_label
    return default_pop


def is_front_matter(page_num: int, page_text: str) -> bool:
    """
    Detects front-matter / table-of-contents pages generically.
    Checks for 'Contents' keyword in the first few pages of any document.
    """
    if page_num <= 3:
        lower = page_text.lower()
        if "contents" in lower and ("overview" in lower or "recommendations" in lower):
            return True
    return False


def is_implicit_break(line: str) -> bool:
    """Checks if a line signals an implicit sub-section boundary."""
    for pattern in IMPLICIT_BREAK_PATTERNS:
        if pattern.match(line):
            return True
    return False


def extract_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts raw text and metadata page by page using PyMuPDF.
    Returns a list of dicts with page_number, raw text, and filename.
    """
    doc = fitz.open(file_path)
    filename = Path(file_path).name
    pages_data = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text()
        pages_data.append({
            "filename": filename,
            "page_number": page_idx + 1,
            "text": text.strip()
        })

    return pages_data


def split_text_with_overlap(
    text: str, 
    max_chars: int = 600, 
    overlap_chars: int = 100
) -> List[str]:
    """
    Splits long section text into overlapping windows without severing paragraphs or sentences.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        # Fallback to newline or sentence split
        paragraphs = [p.strip() for p in re.split(r'(?<=[.\n])\s+', text) if p.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)
        if current_length + para_len > max_chars and current_chunk:
            combined = "\n\n".join(current_chunk)
            chunks.append(combined)
            # Sliding overlap: keep the tail paragraph(s)
            overlap_acc = 0
            overlap_chunk = []
            for p in reversed(current_chunk):
                if overlap_acc + len(p) <= overlap_chars or not overlap_chunk:
                    overlap_chunk.insert(0, p)
                    overlap_acc += len(p)
                else:
                    break
            current_chunk = overlap_chunk
            current_length = sum(len(p) for p in current_chunk) + (len(current_chunk) * 2)

        current_chunk.append(para)
        current_length += para_len + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def make_embedding_text(section_num: str, section_title: str, content: str) -> str:
    """
    Creates a search-optimized text representation by prefixing with section context.
    This improves embedding similarity for section-specific queries.
    """
    return f"[Section {section_num}: {section_title}] {content}"


def make_chunk_id(doc_stem: str, page_num: int, counter: int) -> str:
    """Generates a deterministic, collision-resistant chunk ID using the full document stem."""
    return f"{doc_stem}_p{page_num:02d}_c{counter:03d}"


# ---------------------------------------------------------------------------
# Core chunking engine
# ---------------------------------------------------------------------------

def _flush_block(
    block_lines: List[str],
    section_num: str,
    section_title: str,
    page_num: int,
    filename: str,
    doc_stem: str,
    chunk_counter: int,
    max_chunk_chars: int,
    overlap_chars: int,
    output_list: List[Dict[str, Any]],
) -> int:
    """
    Flushes accumulated block lines into one or more validated chunks.
    Returns the updated chunk_counter.
    """
    block_text = "\n".join(block_lines).strip()
    if len(block_text) < MIN_CHUNK_CHARS:
        return chunk_counter

    # Extract clinical metadata
    grade_match = USPSTF_GRADE_PATTERN.search(block_text)
    evidence_grade = grade_match.group(0).title() if grade_match else "N/A"
    pop = detect_target_population(block_text)

    sub_texts = split_text_with_overlap(block_text, max_chunk_chars, overlap_chars)
    for sub_t in sub_texts:
        if len(sub_t.strip()) < MIN_CHUNK_CHARS:
            continue

        chunk_counter += 1
        chunk_id = make_chunk_id(doc_stem, page_num, chunk_counter)
        embedding_text = make_embedding_text(section_num, section_title, sub_t)

        # Validate through Pydantic schema
        chunk = ProcessedChunk(
            chunk_id=chunk_id,
            content=sub_t,
            embedding_text=embedding_text,
            section_number=section_num,
            section_title=section_title,
            page_number=page_num,
            document_name=filename,
            evidence_grade=evidence_grade,
            target_population=pop,
            char_count=len(sub_t),
        )
        output_list.append(chunk.model_dump())

    return chunk_counter


def section_aware_chunker(
    pages_data: List[Dict[str, Any]], 
    max_chunk_chars: int = 600,
    overlap_chars: int = 100
) -> List[Dict[str, Any]]:
    """
    Parses pages and produces section-aware chunks bound to exact section hierarchies,
    page numbers, document names, and evidence grades.
    
    Handles both explicit section headers (numbered like '1.3 Analgesia') and
    implicit sub-section breaks (e.g. 'Why the committee made the recommendation').
    """
    processed_chunks = []
    current_section_num = "0.0"
    current_section_title = "Overview & Scope"
    chunk_counter = 0

    for page in pages_data:
        filename = page["filename"]
        doc_stem = Path(filename).stem
        page_num = page["page_number"]
        page_text = page["text"]

        if not page_text:
            continue

        # Generic front-matter / contents detection
        if is_front_matter(page_num, page_text):
            continue

        # Split page into structural blocks or paragraphs
        lines = page_text.split("\n")
        current_block = []

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            # Check for explicit section header match (e.g., '1.3 Analgesia' or '2.1 Risk Assessment')
            sec_match = SECTION_HEADER_PATTERN.match(trimmed)
            if sec_match:
                # Flush previous block
                chunk_counter = _flush_block(
                    current_block, current_section_num, current_section_title,
                    page_num, filename, doc_stem, chunk_counter,
                    max_chunk_chars, overlap_chars, processed_chunks,
                )
                current_block = []
                current_section_num = sec_match.group("num").strip()
                current_section_title = sec_match.group("title").strip()
                continue

            # Check for implicit sub-section break (headerless pages)
            if is_implicit_break(trimmed) and current_block:
                chunk_counter = _flush_block(
                    current_block, current_section_num, current_section_title,
                    page_num, filename, doc_stem, chunk_counter,
                    max_chunk_chars, overlap_chars, processed_chunks,
                )
                current_block = []
                # Keep the break line as the start of the next block
                # (it provides context for the upcoming content)

            current_block.append(trimmed)

        # Flush remaining block for this page
        chunk_counter = _flush_block(
            current_block, current_section_num, current_section_title,
            page_num, filename, doc_stem, chunk_counter,
            max_chunk_chars, overlap_chars, processed_chunks,
        )
        current_block = []

    return processed_chunks


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_ingestion_pipeline(
    pdf_dir: str = "data/guidelines", 
    output_file: str = "data/processed_chunks.json"
) -> List[Dict[str, Any]]:
    """
    Executes the end-to-end ingestion pipeline over all guideline PDFs.
    """
    pdf_dir_path = Path(pdf_dir)
    os.makedirs(pdf_dir_path, exist_ok=True)
    os.makedirs(Path(output_file).parent, exist_ok=True)

    pdf_files = sorted(list(pdf_dir_path.glob("*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF guideline files found in directory: {pdf_dir}")

    all_chunks = []
    print("=" * 80)
    print("CLINICAL GUIDELINE INGESTION & SECTION-AWARE CHUNKING PIPELINE")
    print("=" * 80)

    for pdf_path in pdf_files:
        print(f"\n[+] Ingesting: {pdf_path.name}")
        pages = extract_pdf_pages(str(pdf_path))
        print(f"    - Extracted {len(pages)} physical pages.")
        
        chunks = section_aware_chunker(pages)
        print(f"    - Generated {len(chunks)} section-aware chunks.")
        all_chunks.extend(chunks)

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE: Successfully processed and indexed {len(all_chunks)} chunks.")
    print(f"Output saved to: {output_file}")
    print("=" * 80)

    return all_chunks


if __name__ == "__main__":
    chunks = run_ingestion_pipeline()
    print("\n--- SAMPLE CHUNKS PREVIEW ---")
    for sample in chunks[:3]:
        print(json.dumps(sample, indent=2))
        print("-" * 50)



