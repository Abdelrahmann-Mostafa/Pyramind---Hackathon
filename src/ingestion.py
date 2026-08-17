"""
ingestion.py
------------
Step 1 of the RAG pipeline.

Reads the source clinical guideline PDFs, extracts text page-by-page
(with page numbers preserved), cleans the extracted text (removes
repeated legal boilerplate, divider lines, markdown hashes, bullet
glyphs, dot-leaders, hyphenation breaks, and control characters),
detects section headers / numbering (e.g. "1.4 Ongoing orthopaedic
management", "1.4.5"), and tags basic clinical metadata (evidence/
update year, target population).

Output: data/raw_pages.json
    A flat list of page-level records, each with the cleaned text plus
    document/page/section metadata. This is the input to chunking.py.

Run:
    python src/ingestion.py
"""

import json
import re
import unicodedata
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "raw_pages.json"

# Register every source PDF here. Add more guidelines by appending to this list.
SOURCE_DOCUMENTS = [
    {
        "path": DATA_DIR / "NG38_fractures_noncomplex.pdf",
        "document_name": "Fractures (non-complex): assessment and management",
        "document_code": "NG38",
        "publisher": "NICE",
        "published_date": "2016-02-17",
    },
    {
        "path": DATA_DIR / "NG37_fractures_complex.pdf",
        "document_name": "Fractures (complex): assessment and management",
        "document_code": "NG37",
        "publisher": "NICE",
        "published_date": "2016-02-17",
        "last_updated": "2022-11-23",
    },
]

# ---------------------------------------------------------------------
# Section / metadata detection
# ---------------------------------------------------------------------

SECTION_HEADER_RE = re.compile(r"^\s*(\d\.\d)\s+([A-Z][A-Za-z0-9 ,\-()/]+)\s*$", re.MULTILINE)
RECOMMENDATION_RE = re.compile(r"^\s*(\d\.\d\.\d+)\b")

POPULATION_PATTERNS = [
    (re.compile(r"\bchildren\s*\(under 16s?\)", re.I), "Children (under 16)"),
    (re.compile(r"\badults?\s*\(16 or over\)", re.I), "Adults (16+)"),
    (re.compile(r"\bskeletally (immature|mature)\b", re.I), "Skeletal maturity dependent"),
]

YEAR_TAG_RE = re.compile(r"\[(19|20)\d{2}(?:,\s*amended\s*(19|20)\d{2})?\]")

# ---------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------

BOILERPLATE_PATTERNS = [
    re.compile(r"©\s*NICE\s*\d{4}\.?\s*All rights reserved\.?", re.I),
    re.compile(r"Subject to Notice of rights.*?notice-of-rights\)\.?", re.I | re.S),
    re.compile(r"www\.nice\.org\.uk\S*", re.I),
    re.compile(r"Page\s*\d+\s*of\s*\d*", re.I),
    re.compile(r"ISBN:\s*[\d\-]+", re.I),
]

DIVIDER_LINE_RE = re.compile(r"^[\s\-_=~*.•]{3,}$", re.MULTILINE)
HASH_RE = re.compile(r"#{1,6}\s*")
BULLET_CHARS_RE = re.compile(r"[•●▪◦‣·]")
DOT_LEADER_RE = re.compile(r"\.{3,}")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Strips markdown artifacts, divider lines, repeated legal boilerplate,
    bullet glyphs, dot-leaders, and normalizes whitespace/unicode — leaving
    only the actual clinical prose."""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\xa0", " ")

    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)

    text = DIVIDER_LINE_RE.sub(" ", text)
    text = HASH_RE.sub("", text)
    text = BULLET_CHARS_RE.sub("", text)
    text = DOT_LEADER_RE.sub(" ", text)
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)  # de-hyphenate line-wrapped words
    text = CONTROL_CHARS_RE.sub("", text)
    text = " ".join(text.split())

    return text.strip()


def is_toc_page(raw_text: str) -> bool:
    """Detects table-of-contents pages using dot-leader density in the
    RAW (uncleaned) text — must run BEFORE clean_text(), since cleaning
    strips dot-leaders and would make this check always return False."""
    dot_leader_ratio = raw_text.count("...") / max(len(raw_text.splitlines()), 1)
    return dot_leader_ratio > 0.5


def detect_population(text: str) -> str:
    for pattern, label in POPULATION_PATTERNS:
        if pattern.search(text):
            return label
    return "Adults and children (general)"


def detect_current_section(text: str, running_section: dict) -> dict:
    match = SECTION_HEADER_RE.search(text)
    if match:
        running_section = {
            "section_number": match.group(1),
            "section_title": match.group(2).strip(),
        }
    return running_section


def extract_pages(doc_config: dict) -> list:
    pages_out = []
    running_section = {"section_number": "", "section_title": "Front matter"}

    with pdfplumber.open(doc_config["path"]) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            if not raw_text.strip():
                continue

            if is_toc_page(raw_text):
                continue

            running_section = detect_current_section(raw_text, running_section)

            years = sorted(set(m.group(0).strip("[]") for m in YEAR_TAG_RE.finditer(raw_text)))
            population = detect_population(raw_text)

            cleaned_text = clean_text(raw_text)
            if not cleaned_text:
                continue

            record = {
                "document_name": doc_config["document_name"],
                "document_code": doc_config["document_code"],
                "publisher": doc_config["publisher"],
                "published_date": doc_config["published_date"],
                "last_updated": doc_config.get("last_updated", doc_config["published_date"]),
                "page_number": page_index,
                "section_number": running_section["section_number"],
                "section_title": running_section["section_title"],
                "target_population": population,
                "recommendation_years": years,
                "text": cleaned_text,
            }
            pages_out.append(record)

    return pages_out


def main():
    all_pages = []
    for doc_config in SOURCE_DOCUMENTS:
        if not doc_config["path"].exists():
            raise FileNotFoundError(f"Missing source PDF: {doc_config['path']}")
        pages = extract_pages(doc_config)
        print(f"[+] {doc_config['document_code']}: extracted {len(pages)} pages")
        all_pages.extend(pages)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, indent=2, ensure_ascii=False)

    print(f"[+] Wrote {len(all_pages)} page records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
