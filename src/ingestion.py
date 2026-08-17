
import json
import re
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

# Matches top-level section headers like "1.4 Ongoing orthopaedic management"
SECTION_HEADER_RE = re.compile(r"^\s*(\d\.\d)\s+([A-Z][A-Za-z0-9 ,\-()/]+)\s*$", re.MULTILINE)

# Matches individual recommendation numbers like "1.4.5" at the start of a line
RECOMMENDATION_RE = re.compile(r"^\s*(\d\.\d\.\d+)\b")

# Simple population/age detector used later for metadata filtering
POPULATION_PATTERNS = [
    (re.compile(r"\bchildren\s*\(under 16s?\)", re.I), "Children (under 16)"),
    (re.compile(r"\badults?\s*\(16 or over\)", re.I), "Adults (16+)"),
    (re.compile(r"\bskeletally (immature|mature)\b", re.I), "Skeletal maturity dependent"),
]

# Recommendation-level year tags NICE embeds, e.g. "[2016]" or "[2022]"
YEAR_TAG_RE = re.compile(r"\[(19|20)\d{2}(?:,\s*amended\s*(19|20)\d{2})?\]")


def detect_population(text: str) -> str:
    for pattern, label in POPULATION_PATTERNS:
        if pattern.search(text):
            return label
    return "Adults and children (general)"


def detect_current_section(text: str, running_section: dict) -> dict:
    """Update the running (section_number, section_title) state based on
    any section header found on this page. Carries forward the last known
    section across pages that don't start a new one."""
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

            # Skip table-of-contents pages: mostly dot-leader lines, no real
            # clinical content, and they'd otherwise pollute the index with
            # noise chunks that can spuriously match on section-title words.
            dot_leader_ratio = raw_text.count("...") / max(len(raw_text.splitlines()), 1)
            if dot_leader_ratio > 0.5:
                continue

            running_section = detect_current_section(raw_text, running_section)

            record = {
                "document_name": doc_config["document_name"],
                "document_code": doc_config["document_code"],
                "publisher": doc_config["publisher"],
                "published_date": doc_config["published_date"],
                "last_updated": doc_config.get("last_updated", doc_config["published_date"]),
                "page_number": page_index,
                "section_number": running_section["section_number"],
                "section_title": running_section["section_title"],
                "target_population": detect_population(raw_text),
                "recommendation_years": sorted(set(
                    m.group(0).strip("[]") for m in YEAR_TAG_RE.finditer(raw_text)
                )),
                "text": raw_text.strip(),
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
