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
from typing import List, Dict, Any, Optional


SECTION_HEADER_PATTERN = re.compile(
    r'^(?P<num>\d+\.\d+(\.\d+)?)\s+(?P<title>[A-Za-z0-9\s,\-\(\)\/\:]{3,80})',
    re.MULTILINE
)

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


def detect_target_population(text: str, default_pop: str = "Adults") -> str:
    """Detects clinical target population from chunk text."""
    for pattern, pop_label in TARGET_POP_PATTERNS:
        if pattern.search(text):
            return pop_label
    return default_pop


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
    max_chars: int = 1800, 
    overlap_chars: int = 250
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


def section_aware_chunker(
    pages_data: List[Dict[str, Any]], 
    max_chunk_chars: int = 1800,
    overlap_chars: int = 250
) -> List[Dict[str, Any]]:
    """
    Parses pages and produces section-aware chunks bound to exact section hierarchies,
    page numbers, document names, and evidence grades.
    """
    processed_chunks = []
    current_section_num = "0.0"
    current_section_title = "Overview & Scope"
    chunk_counter = 0

    for page in pages_data:
        filename = page["filename"]
        page_num = page["page_number"]
        page_text = page["text"]

        if not page_text:
            continue

        # Check if page is front-matter / contents
        if page_num <= 3 and "Contents" in page_text and filename == "NICE_CG124.pdf":
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
                # Flush previous block if exists
                if current_block:
                    block_text = "\n".join(current_block).strip()
                    if len(block_text) > 40:
                        # Extract evidence grade
                        grade_match = USPSTF_GRADE_PATTERN.search(block_text)
                        evidence_grade = grade_match.group(0).title() if grade_match else "N/A"
                        pop = detect_target_population(block_text)

                        sub_texts = split_text_with_overlap(block_text, max_chunk_chars, overlap_chars)
                        for sub_t in sub_texts:
                            chunk_counter += 1
                            processed_chunks.append({
                                "chunk_id": f"{filename[:8]}_p{page_num:02d}_c{chunk_counter:03d}",
                                "content": sub_t,
                                "section_number": current_section_num,
                                "section_title": current_section_title,
                                "page_number": page_num,
                                "document_name": filename,
                                "evidence_grade": evidence_grade,
                                "target_population": pop,
                                "char_count": len(sub_t)
                            })
                    current_block = []

                current_section_num = sec_match.group("num").strip()
                current_section_title = sec_match.group("title").strip()
                continue

            current_block.append(trimmed)

        # Flush remaining block for this page
        if current_block:
            block_text = "\n".join(current_block).strip()
            if len(block_text) > 80:
                grade_match = USPSTF_GRADE_PATTERN.search(block_text)
                evidence_grade = grade_match.group(0).title() if grade_match else "N/A"
                pop = detect_target_population(block_text)

                sub_texts = split_text_with_overlap(block_text, max_chunk_chars, overlap_chars)
                for sub_t in sub_texts:
                    if len(sub_t.strip()) > 80:
                        chunk_counter += 1
                        processed_chunks.append({
                            "chunk_id": f"{filename[:8]}_p{page_num:02d}_c{chunk_counter:03d}",
                            "content": sub_t,
                            "section_number": current_section_num,
                            "section_title": current_section_title,
                            "page_number": page_num,
                            "document_name": filename,
                            "evidence_grade": evidence_grade,
                            "target_population": pop,
                            "char_count": len(sub_t)
                        })

    return processed_chunks


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
