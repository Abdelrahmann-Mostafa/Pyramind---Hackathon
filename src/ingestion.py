"""
Day 1: Document Ingestion & Hierarchical Chunking
==================================================
Implements research-informed parent-child chunking with section detection,
metadata extraction, and validation.

Usage:
    from src.ingestion import run_ingestion_pipeline
    chunks = run_ingestion_pipeline()
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# We'll use the same tokenizer for approximate token counting
_tokenizer = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO").tokenizer

from src.schemas import ProcessedChunk

# ===================================================================
# CONSTANTS & PATTERNS
# ===================================================================

PARENT_SIZE_TOKENS = 1024
CHILD_SIZE_TOKENS = 256
CHILD_OVERLAP_TOKENS = 50
MIN_CHUNK_CHARS = 100  # minimal content length to keep

SECTION_HEADER_PATTERN = re.compile(
    r'^(?P<num>\d+\.\d+(\.\d+)?)\s+(?P<title>[A-Za-z0-9\s,\-\(\)\/\:]{3,80})',
    re.MULTILINE
)

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

SECTION_TYPE_PATTERNS = {
    "Diagnosis": re.compile(r'\b(diagnos|imaging|x-ray|mri|ct|suspected)\b', re.I),
    "Treatment": re.compile(r'\b(treat|therapy|surgery|operation|anaesthesia|analgesia|pain|medication)\b', re.I),
    "Screening": re.compile(r'\b(screen|prevent|detect|early|risk|assessment)\b', re.I),
    "Adverse Events": re.compile(r'\b(adverse|complication|side effect|comorbidity|mortality)\b', re.I),
    "General": re.compile(r'.*'),
}

FOOTER_PATTERNS = [
    re.compile(r'hip fracture: management \(cg124\)', re.I),
    re.compile(r'©\s*nice\b.*', re.I),
    re.compile(r'page\s+\d+\s+of\s+\d+', re.I),
    re.compile(r'subject to notice of rights.*', re.I),
    re.compile(r'all rights reserved\.?', re.I),
    re.compile(r'^www\.nice\.org\.uk.*', re.I),
]

FRONT_MATTER_PATTERNS = [
    re.compile(r'^clinical guideline$', re.I),
    re.compile(r'^published:\s*.*$', re.I),
    re.compile(r'^last updated:\s*.*$', re.I),
    re.compile(r'^your responsibility$', re.I),
    re.compile(r'^using this guideline$', re.I),
    re.compile(r'^recommendations$', re.I),
    re.compile(r'^who is it for\??$', re.I),
    re.compile(r'^overview$', re.I),
    re.compile(r'^this guideline is the basis of.*$', re.I),
]


# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

def is_footer_or_boilerplate(line: str) -> bool:
    """True for legal boilerplate, footers, or front-matter noise."""
    if not line or not line.strip():
        return True
    clean = line.strip()
    if any(p.search(clean) for p in FOOTER_PATTERNS):
        return True
    if clean.lower().startswith('©') or 'all rights reserved' in clean.lower():
        return True
    if clean.lower().startswith('www.'):
        return True
    return False


def clean_page_lines(lines: List[str]) -> List[str]:
    """Remove boilerplate and front matter, keep clinical content."""
    cleaned = []
    for line in lines:
        trimmed = line.strip()
        if not trimmed:
            continue
        if is_footer_or_boilerplate(trimmed):
            continue
        if trimmed.lower().startswith('hip fracture: management') and len(trimmed) < 60:
            continue
        if any(p.match(trimmed) for p in FRONT_MATTER_PATTERNS):
            continue
        cleaned.append(trimmed)
    return cleaned


def is_front_matter(page_num: int, page_text: str) -> bool:
    """Detect table of contents or front-matter pages."""
    if page_num <= 3:
        lower = page_text.lower()
        if "contents" in lower and ("overview" in lower or "recommendations" in lower):
            return True
    return False


def detect_target_population(text: str) -> str:
    """Extract target population from text."""
    for pattern, label in TARGET_POP_PATTERNS:
        if pattern.search(text):
            return label
    return "Adults"


def detect_section_type(text: str) -> str:
    """Classify section type based on keywords."""
    for sect_type, pattern in SECTION_TYPE_PATTERNS.items():
        if pattern.search(text):
            return sect_type
    return "General"


def looks_like_recommendation_block(text: str) -> bool:
    """Heuristic: true if text is a numbered recommendation or actionable statement."""
    if not text or not text.strip():
        return False
    if re.match(r'^\s*\d+(?:\.\d+)+\s+(?:Do|Offer|Consider|Advise|Use|Ensure|Provide|Discuss|Refer|Schedule)', text, re.I):
        return True
    if re.match(r'^\s*\d+\.\d+(?:\.\d+)?\s+[A-Za-z0-9]', text):
        return True
    if re.match(r'^\s*\d+\s+[A-Z][A-Za-z]', text):
        return True
    return False


def extract_evidence_grade(text: str) -> str:
    """Find USPSTF grade or NICE marker."""
    match = USPSTF_GRADE_PATTERN.search(text)
    return match.group(0).title() if match else "N/A"


def count_tokens(text: str) -> int:
    """Approximate token count using the same tokenizer as the embedding model."""
    return len(_tokenizer.encode(text, add_special_tokens=False))


def split_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
    """
    Split text into overlapping segments by token count, respecting sentence boundaries.
    Uses the model's tokenizer to determine token count.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    # Simple sentence splitting (period, newline, etc.)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    segments = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            # flush current segment
            seg_text = " ".join(current)
            segments.append(seg_text)
            # keep overlap
            overlap_text = " ".join(current)
            overlap_tokens_count = min(overlap_tokens, count_tokens(overlap_text))
            # keep as many tokens as possible for overlap
            overlap_words = overlap_text.split()
            keep = []
            acc = 0
            for w in reversed(overlap_words):
                w_tok = count_tokens(w)
                if acc + w_tok <= overlap_tokens:
                    keep.insert(0, w)
                    acc += w_tok
                else:
                    break
            current = keep if keep else []
            current_tokens = acc
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        segments.append(" ".join(current))
    return segments


# ===================================================================
# PDF EXTRACTION
# ===================================================================

def extract_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    """Extract raw text and metadata page by page."""
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


# ===================================================================
# SECTION DETECTION AND CHUNKING
# ===================================================================

def section_detector(text: str) -> List[Tuple[str, str, int]]:
    """
    Find all section headers and their start positions.
    Returns list of (section_number, section_title, start_char_index).
    """
    matches = []
    for match in SECTION_HEADER_PATTERN.finditer(text):
        num = match.group("num").strip()
        title = match.group("title").strip()
        start = match.start()
        matches.append((num, title, start))
    return matches


def create_parent_chunks(
    pages_data: List[Dict[str, Any]],
    parent_size_tokens: int = PARENT_SIZE_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Create parent chunks of approximately `parent_size_tokens` tokens,
    preserving section boundaries as much as possible.
    Returns a list of parent chunks with metadata.
    """
    parents = []
    current_parent = {
        "content": "",
        "sections": [],
        "page_numbers": set(),
        "document_name": None,
        "evidence_grades": [],
        "target_populations": [],
    }

    def flush_parent() -> Optional[Dict[str, Any]]:
        nonlocal current_parent
        if not current_parent["content"]:
            return None
        # Aggregate metadata
        content = current_parent["content"].strip()
        if not content or len(content) < MIN_CHUNK_CHARS:
            return None
        # Determine primary section
        if current_parent["sections"]:
            # Use the first section number and title
            sec_num, sec_title = current_parent["sections"][0]
        else:
            sec_num, sec_title = "0.0", "General"
        page = min(current_parent["page_numbers"]) if current_parent["page_numbers"] else 1
        ev_grade = max(set(current_parent["evidence_grades"]), key=lambda g: current_parent["evidence_grades"].count(g)) if current_parent["evidence_grades"] else "N/A"
        pop = max(set(current_parent["target_populations"]), key=lambda p: current_parent["target_populations"].count(p)) if current_parent["target_populations"] else "Adults"
        doc = current_parent["document_name"] or "Unknown"
        sect_type = detect_section_type(content)
        return {
            "content": content,
            "section_number": sec_num,
            "section_title": sec_title,
            "page_number": page,
            "document_name": doc,
            "evidence_grade": ev_grade,
            "target_population": pop,
            "section_type": sect_type,
            "char_count": len(content),
        }

    for page in pages_data:
        filename = page["filename"]
        page_num = page["page_number"]
        text = page["text"]

        if is_front_matter(page_num, text):
            continue

        lines = clean_page_lines(text.split("\n"))
        page_text = "\n".join(lines)

        if not page_text:
            continue

        # Detect sections on this page
        sections = section_detector(page_text)

        if sections:
            # Process each section in order
            for idx, (sec_num, sec_title, start_pos) in enumerate(sections):
                # Determine the end of this section (next section start or end of text)
                end_pos = sections[idx+1][2] if idx+1 < len(sections) else len(page_text)
                section_text = page_text[start_pos:end_pos].strip()
                if not section_text:
                    continue
                # Add to current parent or start new parent if section changes
                # If the current parent already has a different section, flush it.
                if current_parent["sections"] and current_parent["sections"][-1][0] != sec_num:
                    parent_data = flush_parent()
                    if parent_data:
                        parents.append(parent_data)
                    current_parent = {
                        "content": "",
                        "sections": [],
                        "page_numbers": set(),
                        "document_name": filename,
                        "evidence_grades": [],
                        "target_populations": [],
                    }
                current_parent["content"] += "\n\n" + section_text
                current_parent["sections"].append((sec_num, sec_title))
                current_parent["page_numbers"].add(page_num)
                current_parent["document_name"] = filename
                current_parent["evidence_grades"].append(extract_evidence_grade(section_text))
                current_parent["target_populations"].append(detect_target_population(section_text))
                # Check if current parent exceeds size; if so, flush and start new
                if count_tokens(current_parent["content"]) > parent_size_tokens:
                    parent_data = flush_parent()
                    if parent_data:
                        parents.append(parent_data)
                    # Start new parent with the remaining text (overlap)
                    # We'll keep last section as part of new parent
                    remaining_text = current_parent["content"]
                    # Keep only last part that fits within parent size? Actually we'll just reset.
                    # But better: keep the last section to maintain continuity.
                    # Simple: keep the last section's content as new parent start.
                    last_sec = current_parent["sections"][-1]
                    # We'll re-add that section to new parent
                    current_parent = {
                        "content": section_text,  # start with the section that caused overflow
                        "sections": [(sec_num, sec_title)],
                        "page_numbers": {page_num},
                        "document_name": filename,
                        "evidence_grades": [extract_evidence_grade(section_text)],
                        "target_populations": [detect_target_population(section_text)],
                    }
        else:
            # No sections: treat as part of current parent (if any) or create new parent with default section
            if not current_parent["sections"]:
                current_parent["sections"] = [("0.0", "General")]
                current_parent["document_name"] = filename
            current_parent["content"] += "\n\n" + page_text
            current_parent["page_numbers"].add(page_num)
            # update metadata from page_text
            ev = extract_evidence_grade(page_text)
            if ev != "N/A":
                current_parent["evidence_grades"].append(ev)
            pop = detect_target_population(page_text)
            if pop != "Adults":
                current_parent["target_populations"].append(pop)

        # Check if parent exceeds size after adding page
        if count_tokens(current_parent["content"]) > parent_size_tokens:
            parent_data = flush_parent()
            if parent_data:
                parents.append(parent_data)
            # Keep the last section as new parent start
            if current_parent["sections"]:
                last_sec = current_parent["sections"][-1]
                # We'll keep the content of the last section? But we need to extract from the page text.
                # Simpler: reset and start new parent with the current page text.
                current_parent = {
                    "content": page_text,
                    "sections": current_parent["sections"][-1:],
                    "page_numbers": {page_num},
                    "document_name": filename,
                    "evidence_grades": [extract_evidence_grade(page_text)],
                    "target_populations": [detect_target_population(page_text)],
                }

    # Flush final parent
    parent_data = flush_parent()
    if parent_data:
        parents.append(parent_data)

    # Assign unique IDs to parents
    for idx, parent in enumerate(parents):
        # Use document stem and page number for ID
        doc_stem = Path(parent["document_name"]).stem
        parent["chunk_id"] = f"{doc_stem}_parent_{idx+1:03d}"
        parent["is_parent"] = True
        parent["parent_id"] = None
        parent["embedding_text"] = f"[Section {parent['section_number']}: {parent['section_title']}] {parent['content']}"

    return parents


def create_child_chunks(
    parent_chunks: List[Dict[str, Any]],
    child_size_tokens: int = CHILD_SIZE_TOKENS,
    overlap_tokens: int = CHILD_OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Split each parent chunk into child chunks of `child_size_tokens` with overlap.
    Each child inherits parent metadata and stores parent_id.
    """
    children = []
    child_counter = 1
    for parent in parent_chunks:
        parent_id = parent["chunk_id"]
        content = parent["content"]
        # Split content into child segments
        segments = split_text_by_tokens(content, child_size_tokens, overlap_tokens)
        for seg in segments:
            if len(seg) < MIN_CHUNK_CHARS and not looks_like_recommendation_block(seg):
                continue
            child = {
                "content": seg,
                "embedding_text": f"[Section {parent['section_number']}: {parent['section_title']}] {seg}",
                "parent_id": parent_id,
                "section_number": parent["section_number"],
                "section_title": parent["section_title"],
                "section_type": parent.get("section_type", "General"),
                "page_number": parent["page_number"],
                "document_name": parent["document_name"],
                "evidence_grade": parent["evidence_grade"],
                "target_population": parent["target_population"],
                "char_count": len(seg),
                "is_parent": False,
            }
            # Assign chunk_id
            doc_stem = Path(parent["document_name"]).stem
            child["chunk_id"] = f"{doc_stem}_p{parent['page_number']:02d}_c{child_counter:03d}"
            child_counter += 1
            children.append(child)
    return children


def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove exact duplicates and whitespace-normalized duplicates."""
    seen = set()
    deduped = []
    for chunk in chunks:
        text = re.sub(r'\s+', ' ', chunk["content"]).strip().lower()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        deduped.append(chunk)
    return deduped


def validate_chunks(chunks: List[Dict[str, Any]]) -> bool:
    """Basic validation: each chunk has required fields and unique IDs."""
    ids = set()
    for chunk in chunks:
        required = ["chunk_id", "content", "section_number", "page_number", "document_name"]
        for field in required:
            if field not in chunk:
                print(f"Validation error: missing '{field}' in chunk {chunk.get('chunk_id', 'unknown')}")
                return False
        if chunk["chunk_id"] in ids:
            print(f"Duplicate chunk_id: {chunk['chunk_id']}")
            return False
        ids.add(chunk["chunk_id"])
        if len(chunk["content"]) < MIN_CHUNK_CHARS and not looks_like_recommendation_block(chunk["content"]):
            print(f"Warning: chunk {chunk['chunk_id']} is short ({len(chunk['content'])} chars)")
    return True


def run_ingestion_pipeline(
    pdf_dir: str = "data/guidelines",
    output_file: str = "data/processed_chunks.json",
) -> List[Dict[str, Any]]:
    """
    Execute full ingestion: extract PDFs, create parent chunks, child chunks,
    deduplicate, validate, and save.
    """
    pdf_dir_path = Path(pdf_dir)
    output_path = Path(output_file)
    if not pdf_dir_path.is_absolute():
        pdf_dir_path = Path(__file__).resolve().parent.parent / pdf_dir
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent.parent / output_file

    os.makedirs(pdf_dir_path, exist_ok=True)
    os.makedirs(output_path.parent, exist_ok=True)

    pdf_files = sorted(list(pdf_dir_path.glob("*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir_path}")

    print("=" * 80)
    print("CLINICAL GUIDELINE INGESTION (Hierarchical Chunking)")
    print("=" * 80)

    all_parents = []
    for pdf_path in pdf_files:
        print(f"\n[+] Processing: {pdf_path.name}")
        pages = extract_pdf_pages(str(pdf_path))
        print(f"    - {len(pages)} pages extracted.")
        parents = create_parent_chunks(pages)
        print(f"    - {len(parents)} parent chunks created.")
        all_parents.extend(parents)

    # Create children from all parents
    all_children = create_child_chunks(all_parents)
    print(f"\n[+] Created {len(all_children)} child chunks from {len(all_parents)} parents.")

    # Deduplicate children (parents likely unique)
    all_children = deduplicate_chunks(all_children)

    # Combine parents and children for output
    all_chunks = all_parents + all_children

    # Validate
    if not validate_chunks(all_chunks):
        raise ValueError("Chunk validation failed.")

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE: {len(all_chunks)} total chunks")
    print(f"  - Parents: {len(all_parents)}")
    print(f"  - Children: {len(all_children)}")
    print(f"Output saved to: {output_path}")
    print("=" * 80)

    return all_chunks


if __name__ == "__main__":
    run_ingestion_pipeline()