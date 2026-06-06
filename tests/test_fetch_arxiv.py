from __future__ import annotations

import tempfile
import urllib.error
import unittest
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from unittest import mock

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

    def test_readme_replacement_preserves_backslashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            readme = Path(tmpdir) / "README.md"
            readme.write_text(
                "# Demo\n\n"
                f"{fetch_arxiv.README_START}\nold content\n{fetch_arxiv.README_END}\n",
                encoding="utf-8",
            )

            digest = r"Abstract: This paper studies \epsilon constraints and \e escapes."
            fetch_arxiv.update_readme(readme, digest)
            content = readme.read_text(encoding="utf-8")

            self.assertIn(r"\epsilon constraints", content)
            self.assertIn(r"\e escapes", content)
            self.assertNotIn("old content", content)

    def test_build_rss_feed_url_joins_categories(self) -> None:
        url = fetch_arxiv.build_rss_feed_url(
            ["cs.AI", "cs.LG", "cs.CV"],
            "https://rss.arxiv.org/atom/",
        )

        self.assertEqual(url, "https://rss.arxiv.org/atom/cs.AI+cs.LG+cs.CV")

    def test_fetch_arxiv_entries_from_rss(self) -> None:
        config = fetch_arxiv.load_config(ROOT / "config.yaml")
        config["arxiv"]["categories"] = ["cs.AI", "cs.CV"]

        with mock.patch.object(fetch_arxiv, "http_get_text", return_value=SAMPLE_FEED) as get_text:
            entries = fetch_arxiv.fetch_arxiv_entries_from_rss(config)

        self.assertEqual(len(entries), 2)
        get_text.assert_called_once()
        self.assertEqual(
            get_text.call_args.args[0],
            "https://rss.arxiv.org/atom/cs.AI+cs.CV",
        )
        self.assertIsNone(get_text.call_args.args[1])

    def test_rss_source_falls_back_to_search_api(self) -> None:
        config = fetch_arxiv.load_config(ROOT / "config.yaml")
        config["arxiv"]["source"] = "rss"
        root = ET.fromstring(SAMPLE_FEED)
        entries = list(root.findall("atom:entry", fetch_arxiv.NS))

        with mock.patch.object(
            fetch_arxiv,
            "fetch_arxiv_entries_from_rss",
            side_effect=TimeoutError("rss timeout"),
        ), mock.patch.object(
            fetch_arxiv,
            "fetch_arxiv_entries_from_search",
            return_value=entries,
        ) as search:
            fetched = fetch_arxiv.fetch_arxiv_entries(config)

        self.assertEqual(len(fetched), 2)
        search.assert_called_once_with(config)

    def test_split_category_fetch_continues_after_partial_failure(self) -> None:
        config = fetch_arxiv.load_config(ROOT / "config.yaml")
        config["arxiv"]["source"] = "search"
        config["arxiv"]["categories"] = ["cs.AI", "cs.CV"]
        config["arxiv"]["request_delay_seconds"] = 0
        root = ET.fromstring(SAMPLE_FEED)
        entries = list(root.findall("atom:entry", fetch_arxiv.NS))

        def fake_fetch(categories, _arxiv_config, _max_results):
            if categories == ["cs.AI"]:
                return entries
            raise TimeoutError("slow category")

        with mock.patch.object(
            fetch_arxiv,
            "fetch_arxiv_entries_for_categories",
            side_effect=fake_fetch,
        ):
            fetched = fetch_arxiv.fetch_arxiv_entries(config)

        self.assertEqual(len(fetched), 2)

    def test_split_category_fetch_fails_when_all_categories_fail(self) -> None:
        config = fetch_arxiv.load_config(ROOT / "config.yaml")
        config["arxiv"]["source"] = "search"
        config["arxiv"]["categories"] = ["cs.AI", "cs.CV"]
        config["arxiv"]["request_delay_seconds"] = 0

        with mock.patch.object(
            fetch_arxiv,
            "fetch_arxiv_entries_for_categories",
            side_effect=TimeoutError("arxiv timeout"),
        ):
            with self.assertRaisesRegex(RuntimeError, "All arXiv category requests failed"):
                fetch_arxiv.fetch_arxiv_entries(config)

    def test_split_category_results_are_deduplicated_later_by_arxiv_id(self) -> None:
        config = fetch_arxiv.load_config(ROOT / "config.yaml")
        config["arxiv"]["source"] = "search"
        config["arxiv"]["categories"] = ["cs.AI", "cs.CV"]
        config["arxiv"]["request_delay_seconds"] = 0
        root = ET.fromstring(SAMPLE_FEED)
        entries = list(root.findall("atom:entry", fetch_arxiv.NS))

        with mock.patch.object(
            fetch_arxiv,
            "fetch_arxiv_entries_for_categories",
            return_value=entries,
        ):
            fetched = fetch_arxiv.fetch_arxiv_entries(config)

        tz = fetch_arxiv.get_report_timezone("Asia/Shanghai")
        papers = fetch_arxiv.filter_and_sort_papers(fetched, config, date(2026, 6, 6), tz)

        self.assertEqual(len(fetched), 4)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].arxiv_id, "2606.00001")

    def test_http_503_uses_retry_after_header(self) -> None:
        error = urllib.error.HTTPError(
            url="https://export.arxiv.org/api/query",
            code=503,
            msg="Service Unavailable",
            hdrs={"Retry-After": "7"},
            fp=None,
        )

        class FakeResponse:
            headers = mock.Mock()

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return SAMPLE_FEED.encode("utf-8")

        FakeResponse.headers.get_content_charset.return_value = "utf-8"

        with mock.patch.object(
            fetch_arxiv.urllib.request,
            "urlopen",
            side_effect=[error, FakeResponse()],
        ), mock.patch.object(fetch_arxiv.time, "sleep") as sleep:
            text = fetch_arxiv.http_get_text(
                "https://export.arxiv.org/api/query",
                {"search_query": "cat:cs.AI"},
                timeout_seconds=30,
                request_retries=2,
                retry_delay_seconds=30,
            )

        self.assertIn("Agentic Large Language Models", text)
        sleep.assert_called_once_with(7.0)

    def test_main_uses_latest_json_when_all_arxiv_requests_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readme = tmp / "README.md"
            latest_json = tmp / "data" / "latest.json"
            archive_dir = tmp / "data"
            readme.write_text(
                "# Demo\n\n"
                f"{fetch_arxiv.README_START}\nold\n{fetch_arxiv.README_END}\n",
                encoding="utf-8",
            )
            latest_json.parent.mkdir(parents=True)
            paper = fetch_arxiv.Paper(
                arxiv_id="2606.00001",
                title="Fallback Large Language Model Paper",
                authors=["Alice"],
                published="2026-06-05T00:00:00+08:00",
                updated="2026-06-05T00:00:00+08:00",
                categories=["cs.AI"],
                abstract="A fallback abstract about a large language model.",
                abs_url="https://arxiv.org/abs/2606.00001",
                pdf_url="https://arxiv.org/pdf/2606.00001",
                matched_topics=["Large Language Models"],
                score=3,
            )
            latest_json.write_text(
                fetch_arxiv.json.dumps(
                    {
                        "report_date": "2026-06-05",
                        "paper_count": 1,
                        "papers": [fetch_arxiv.asdict(paper)],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            config = fetch_arxiv.load_config(ROOT / "config.yaml")
            config["outputs"]["readme"] = str(readme)
            config["outputs"]["latest_json"] = str(latest_json)
            config["outputs"]["archive_dir"] = str(archive_dir)

            with mock.patch.object(fetch_arxiv, "load_config", return_value=config), mock.patch.object(
                fetch_arxiv,
                "fetch_arxiv_entries",
                side_effect=RuntimeError("All arXiv category requests failed"),
            ), mock.patch.object(
                fetch_arxiv.sys,
                "argv",
                ["fetch_arxiv.py", "--config", str(tmp / "config.yaml"), "--date", "2026-06-06"],
            ):
                exit_code = fetch_arxiv.main()

            content = readme.read_text(encoding="utf-8")
            archive_payload = fetch_arxiv.json.loads((archive_dir / "2026-06-06.json").read_text(encoding="utf-8"))
            latest_payload = fetch_arxiv.json.loads(latest_json.read_text(encoding="utf-8"))

            self.assertEqual(exit_code, 0)
            self.assertIn("showing the last successful digest from 2026-06-05", content)
            self.assertIn("Fallback Large Language Model Paper", content)
            self.assertTrue(archive_payload["fallback"])
            self.assertEqual(latest_payload["report_date"], "2026-06-05")

    def test_main_fails_when_arxiv_and_fallback_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = fetch_arxiv.load_config(ROOT / "config.yaml")
            config["outputs"]["readme"] = str(tmp / "README.md")
            config["outputs"]["latest_json"] = str(tmp / "data" / "latest.json")
            config["outputs"]["archive_dir"] = str(tmp / "data")

            with mock.patch.object(fetch_arxiv, "load_config", return_value=config), mock.patch.object(
                fetch_arxiv,
                "fetch_arxiv_entries",
                side_effect=RuntimeError("All arXiv category requests failed"),
            ), mock.patch.object(
                fetch_arxiv.sys,
                "argv",
                ["fetch_arxiv.py", "--config", str(tmp / "config.yaml"), "--date", "2026-06-06"],
            ):
                with self.assertRaisesRegex(RuntimeError, "All arXiv category requests failed"):
                    fetch_arxiv.main()


if __name__ == "__main__":
    unittest.main()
