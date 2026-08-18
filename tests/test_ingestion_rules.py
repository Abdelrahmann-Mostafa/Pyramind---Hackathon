import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion import clean_page_lines, _flush_block


class IngestionRulesTests(unittest.TestCase):
    def test_clean_page_lines_removes_footer_and_front_matter(self):
        lines = [
            "Hip fracture: management",
            "Clinical guideline",
            "Published: 22 June 2011",
            "1.2.1",
            "Perform surgery on the day of, or the day after, admission.",
            "Hip fracture: management (CG124)",
            "© NICE 2026.",
            "Page 6 of 28",
        ]

        cleaned = clean_page_lines(lines)

        self.assertNotIn("Clinical guideline", cleaned)
        self.assertNotIn("© NICE 2026.", cleaned)
        self.assertNotIn("Page 6 of 28", cleaned)
        self.assertIn("1.2.1", cleaned)
        self.assertIn("Perform surgery on the day of, or the day after, admission.", cleaned)

    def test_flush_rejects_micro_chunks_and_footer_only_blocks(self):
        output = []

        count = _flush_block(
            ["• uncontrolled diabetes"],
            "1.2",
            "Timing of surgery",
            7,
            "NICE_CG124.pdf",
            "NICE_CG124",
            0,
            600,
            100,
            output,
        )

        self.assertEqual(count, 0)
        self.assertEqual(output, [])

    def test_clean_page_lines_keeps_real_recommendation_lines(self):
        lines = [
            "1.3.4",
            "Offer paracetamol every 6 hours preoperatively unless contraindicated.",
            "[2011]",
        ]

        cleaned = clean_page_lines(lines)

        self.assertEqual(cleaned, lines)


if __name__ == "__main__":
    unittest.main()
