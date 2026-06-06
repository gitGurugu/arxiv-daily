from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import fetch_arxiv


SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2606.00001</id>
    <updated>2026-06-06T12:00:00Z</updated>
    <published>2026-06-06T12:00:00Z</published>
    <title>Agentic Large Language Models for Visual Reasoning</title>
    <summary>
      We study tool use and retrieval augmented generation for multimodal
      large language model reasoning.
    </summary>
    <author><name>Alice Researcher</name></author>
    <author><name>Bob Scientist</name></author>
    <category term="cs.AI" />
    <category term="cs.CV" />
    <link href="https://arxiv.org/abs/2606.00001" rel="alternate" />
    <link title="pdf" href="https://arxiv.org/pdf/2606.00001" rel="related" type="application/pdf" />
  </entry>
  <entry>
    <id>https://arxiv.org/abs/2606.00002</id>
    <updated>2026-06-06T10:00:00Z</updated>
    <published>2026-06-06T10:00:00Z</published>
    <title>Unrelated Numerical Solver</title>
    <summary>This paper studies finite element methods.</summary>
    <author><name>Carol Engineer</name></author>
    <category term="math.NA" />
    <link href="https://arxiv.org/abs/2606.00002" rel="alternate" />
  </entry>
</feed>
"""


class FetchArxivTest(unittest.TestCase):
    def test_config_loads_and_merges_defaults(self) -> None:
        config = fetch_arxiv.load_config(ROOT / "config.yaml")

        self.assertEqual(config["report_timezone"], "Asia/Shanghai")
        self.assertIn("cs.AI", config["arxiv"]["categories"])
        self.assertEqual(config["arxiv"]["endpoint"], "https://export.arxiv.org/api/query")

    def test_atom_entries_are_filtered_and_rendered(self) -> None:
        config = fetch_arxiv.load_config(ROOT / "config.yaml")
        root = ET.fromstring(SAMPLE_FEED)
        entries = list(root.findall("atom:entry", fetch_arxiv.NS))
        tz = fetch_arxiv.get_report_timezone("Asia/Shanghai")

        papers = fetch_arxiv.filter_and_sort_papers(entries, config, date(2026, 6, 6), tz)
        markdown = fetch_arxiv.render_markdown(papers, date(2026, 6, 6), config)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].arxiv_id, "2606.00001")
        self.assertIn("Large Language Models", papers[0].matched_topics)
        self.assertIn("Agents and RAG", papers[0].matched_topics)
        self.assertIn("Agentic Large Language Models", markdown)
        self.assertNotIn("Unrelated Numerical Solver", markdown)

    def test_readme_marker_region_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            readme = Path(tmpdir) / "README.md"
            readme.write_text(
                "# Demo\n\n"
                f"{fetch_arxiv.README_START}\nold content\n{fetch_arxiv.README_END}\n",
                encoding="utf-8",
            )

            fetch_arxiv.update_readme(readme, "new content")
            content = readme.read_text(encoding="utf-8")

            self.assertIn("new content", content)
            self.assertNotIn("old content", content)


if __name__ == "__main__":
    unittest.main()
