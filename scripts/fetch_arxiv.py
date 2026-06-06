from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python 3.8 fallback is not expected.
    ZoneInfo = None  # type: ignore[assignment]


README_START = "<!-- ARXIV-DAILY:START -->"
README_END = "<!-- ARXIV-DAILY:END -->"

ATOM_NS = "http://www.w3.org/2005/Atom"
NS = {"atom": ATOM_NS}
USER_AGENT = "arxiv-daily/1.0 (https://github.com/gitGurugu/arxiv-daily; mailto:actions@github.com)"

DEFAULT_CONFIG: dict[str, Any] = {
    "report_timezone": "UTC",
    "arxiv": {
        "endpoint": "https://export.arxiv.org/api/query",
        "categories": ["cs.AI", "cs.LG", "cs.CV", "cs.CL"],
        "max_results": 100,
        "per_category_max_results": 25,
        "days_back": 3,
        "sort_by": "submittedDate",
        "sort_order": "descending",
        "timeout_seconds": 30,
        "request_retries": 5,
        "retry_delay_seconds": 30,
        "max_retry_delay_seconds": 60,
        "request_delay_seconds": 10,
        "split_categories": True,
    },
    "filters": {
        "require_topic_match": True,
        "min_score": 1,
        "max_papers_per_day": 30,
    },
    "outputs": {
        "readme": "README.md",
        "latest_json": "data/latest.json",
        "archive_dir": "data",
    },
    "llm": {
        "enabled": True,
        "max_papers": 8,
        "language": "Simplified Chinese",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "timeout_seconds": 45,
    },
    "topics": {},
}

FIXED_TIMEZONES = {
    "Asia/Shanghai": timezone(timedelta(hours=8), name="Asia/Shanghai"),
    "UTC": timezone.utc,
}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    updated: str
    categories: list[str]
    abstract: str
    abs_url: str
    pdf_url: str
    matched_topics: list[str]
    score: int
    summary_zh: str | None = None


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} must be JSON syntax. JSON is valid YAML and keeps this "
            "project dependency-free."
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping.")
    return deep_merge(DEFAULT_CONFIG, loaded)


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def get_report_timezone(name: str):
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except Exception:
            pass
    if name in FIXED_TIMEZONES:
        return FIXED_TIMEZONES[name]
    print(f"Unknown timezone {name!r}; falling back to UTC.", file=sys.stderr)
    return timezone.utc


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def extract_arxiv_id(abs_url: str) -> str:
    return abs_url.rstrip("/").split("/")[-1]


def build_query(categories: list[str]) -> str:
    if not categories:
        return "all:*"
    return " OR ".join(f"cat:{category}" for category in categories)


def topic_specs(config: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    topics = config.get("topics", {})
    if not isinstance(topics, dict):
        raise ValueError("topics must be a mapping.")

    specs: list[tuple[str, str, list[str]]] = []
    for topic_id, spec in topics.items():
        if isinstance(spec, dict):
            display_name = str(spec.get("display_name") or topic_id)
            keywords = spec.get("keywords", [])
        elif isinstance(spec, list):
            display_name = str(topic_id)
            keywords = spec
        else:
            continue
        specs.append((str(topic_id), display_name, [str(item) for item in keywords]))
    return specs


def keyword_matches(keyword: str, text: str) -> bool:
    keyword = clean_text(keyword).lower()
    if not keyword:
        return False
    if re.search(r"\s", keyword):
        return keyword in text
    pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def score_topics(
    title: str,
    abstract: str,
    specs: list[tuple[str, str, list[str]]],
) -> tuple[list[str], int]:
    title_text = title.lower()
    abstract_text = abstract.lower()
    matched_topics: list[str] = []
    total_score = 0

    for _topic_id, display_name, keywords in specs:
        topic_score = 0
        for keyword in keywords:
            if keyword_matches(keyword, title_text):
                topic_score += 3
            elif keyword_matches(keyword, abstract_text):
                topic_score += 1
        if topic_score > 0:
            matched_topics.append(display_name)
            total_score += topic_score

    return matched_topics, total_score


def retry_after_seconds(exc: BaseException) -> float | None:
    headers = getattr(exc, "headers", None)
    if not headers:
        return None
    retry_after = headers.get("Retry-After")
    if not retry_after:
        return None
    retry_after = str(retry_after).strip()
    if retry_after.isdigit():
        return float(retry_after)
    try:
        retry_at = datetime.strptime(retry_after, "%a, %d %b %Y %H:%M:%S %Z")
        retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except ValueError:
        return None


def retry_delay_for_attempt(
    attempt: int,
    retry_delay_seconds: float,
    exc: BaseException,
    max_retry_delay_seconds: float | None = None,
) -> float:
    retry_after = retry_after_seconds(exc)
    if retry_after is not None:
        delay = retry_after
    else:
        delay = max(0.0, float(retry_delay_seconds)) * attempt
    if max_retry_delay_seconds is not None:
        delay = min(delay, max(0.0, float(max_retry_delay_seconds)))
    return delay


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def http_get_text(
    url: str,
    params: dict[str, Any],
    timeout_seconds: int,
    request_retries: int = 3,
    retry_delay_seconds: float = 10,
    max_retry_delay_seconds: float | None = None,
) -> str:
    query = urllib.parse.urlencode(params)
    separator = "&" if urllib.parse.urlparse(url).query else "?"
    request = urllib.request.Request(
        f"{url}{separator}{query}",
        headers={"User-Agent": USER_AGENT},
        method="GET",
    )

    attempts = max(1, int(request_retries))
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            delay = retry_delay_for_attempt(
                attempt,
                retry_delay_seconds,
                exc,
                max_retry_delay_seconds=max_retry_delay_seconds,
            )
            print(
                f"arXiv request failed on attempt {attempt}/{attempts}: {exc}. "
                f"Retrying in {delay:g}s.",
                file=sys.stderr,
            )
            if delay:
                time.sleep(delay)

    assert last_error is not None
    raise last_error


def arxiv_query_params(
    categories: list[str],
    arxiv_config: dict[str, Any],
    max_results: int,
) -> dict[str, Any]:
    return {
        "search_query": build_query(categories),
        "start": 0,
        "max_results": max_results,
        "sortBy": arxiv_config.get("sort_by", "submittedDate"),
        "sortOrder": arxiv_config.get("sort_order", "descending"),
    }


def fetch_arxiv_entries_for_categories(
    categories: list[str],
    arxiv_config: dict[str, Any],
    max_results: int,
) -> list[ET.Element]:
    params = arxiv_query_params(categories, arxiv_config, max_results)
    xml_text = http_get_text(
        str(arxiv_config["endpoint"]),
        params,
        int(arxiv_config.get("timeout_seconds", 30)),
        int(arxiv_config.get("request_retries", 3)),
        float(arxiv_config.get("retry_delay_seconds", 10)),
        optional_float(arxiv_config.get("max_retry_delay_seconds")),
    )
    root = ET.fromstring(xml_text)
    return list(root.findall("atom:entry", NS))


def fetch_arxiv_entries(config: dict[str, Any]) -> list[ET.Element]:
    arxiv_config = config["arxiv"]
    categories = [str(category) for category in arxiv_config.get("categories", [])]
    max_results = int(arxiv_config.get("max_results", 100))
    split_categories = bool(arxiv_config.get("split_categories", True)) and bool(categories)

    if not split_categories:
        return fetch_arxiv_entries_for_categories(categories, arxiv_config, max_results)

    all_entries: list[ET.Element] = []
    failures: list[str] = []
    successful_requests = 0
    per_category_max_results = int(
        arxiv_config.get("per_category_max_results") or max_results
    )
    request_delay_seconds = max(0.0, float(arxiv_config.get("request_delay_seconds", 3)))

    for index, category in enumerate(categories):
        if index > 0 and request_delay_seconds:
            time.sleep(request_delay_seconds)
        try:
            entries = fetch_arxiv_entries_for_categories(
                [category],
                arxiv_config,
                per_category_max_results,
            )
            successful_requests += 1
            all_entries.extend(entries)
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, OSError, ET.ParseError) as exc:
            message = f"{category}: {exc}"
            failures.append(message)
            print(f"Warning: arXiv category request failed: {message}", file=sys.stderr)

    if failures:
        print(
            "Warning: continuing with partial arXiv results; failed categories: "
            + "; ".join(failures),
            file=sys.stderr,
        )
    if successful_requests == 0:
        raise RuntimeError("All arXiv category requests failed: " + "; ".join(failures))

    return all_entries


def child_text(entry: ET.Element, name: str) -> str:
    child = entry.find(f"atom:{name}", NS)
    return clean_text(child.text if child is not None else None)


def entry_to_paper(
    entry: ET.Element,
    specs: list[tuple[str, str, list[str]]],
    report_tz,
) -> Paper:
    title = child_text(entry, "title")
    abstract = child_text(entry, "summary")
    abs_url = child_text(entry, "id")
    arxiv_id = extract_arxiv_id(abs_url)
    published_raw = child_text(entry, "published")
    updated_raw = child_text(entry, "updated")
    published_dt = parse_datetime(published_raw)
    updated_dt = parse_datetime(updated_raw)
    published = published_dt.astimezone(report_tz).isoformat() if published_dt else published_raw
    updated = updated_dt.astimezone(report_tz).isoformat() if updated_dt else updated_raw

    authors = [
        child_text(author, "name")
        for author in entry.findall("atom:author", NS)
        if child_text(author, "name")
    ]
    categories = [
        str(category.attrib["term"])
        for category in entry.findall("atom:category", NS)
        if category.attrib.get("term")
    ]

    pdf_url = ""
    for link in entry.findall("atom:link", NS):
        title_attr = str(link.attrib.get("title") or "").lower()
        link_type = str(link.attrib.get("type") or "").lower()
        if title_attr == "pdf" or link_type == "application/pdf":
            pdf_url = str(link.attrib.get("href") or "")
            break
    if not pdf_url and "/abs/" in abs_url:
        pdf_url = abs_url.replace("/abs/", "/pdf/")

    matched_topics, score = score_topics(title, abstract, specs)
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        published=published,
        updated=updated,
        categories=categories,
        abstract=abstract,
        abs_url=abs_url,
        pdf_url=pdf_url,
        matched_topics=matched_topics,
        score=score,
    )


def filter_and_sort_papers(
    entries: list[ET.Element],
    config: dict[str, Any],
    report_day: date,
    report_tz,
) -> list[Paper]:
    specs = topic_specs(config)
    filters = config["filters"]
    days_back = int(config["arxiv"].get("days_back", 3))
    since_day = report_day - timedelta(days=days_back)
    require_topic_match = bool(filters.get("require_topic_match", True))
    min_score = int(filters.get("min_score", 1))
    max_papers = int(filters.get("max_papers_per_day", 30))

    papers_by_id: dict[str, Paper] = {}
    for entry in entries:
        paper = entry_to_paper(entry, specs, report_tz)
        published_dt = parse_datetime(child_text(entry, "published"))
        if published_dt:
            published_day = published_dt.astimezone(report_tz).date()
            if published_day < since_day or published_day > report_day:
                continue
        if require_topic_match and not paper.matched_topics:
            continue
        if paper.score < min_score:
            continue
        papers_by_id[paper.arxiv_id] = paper

    papers = list(papers_by_id.values())
    papers.sort(key=lambda item: (item.published, item.score, item.title), reverse=True)
    return papers[:max_papers]


def env_or_config(env_name: str, config_value: Any) -> str:
    value = os.getenv(env_name)
    if value and value.strip():
        return value.strip()
    return str(config_value or "").strip()


def http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_seconds: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset, errors="replace"))


def request_llm_summary(paper: Paper, llm_config: dict[str, Any]) -> str | None:
    api_key = env_or_config("OPENAI_API_KEY", "")
    if not api_key:
        return None

    model = env_or_config("OPENAI_MODEL", llm_config.get("model"))
    base_url = env_or_config("OPENAI_BASE_URL", llm_config.get("base_url"))
    if not model or not base_url:
        return None

    language = llm_config.get("language", "Simplified Chinese")
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You write concise {language} research paper summaries. "
                    "Return exactly two short sentences. Focus on the problem, "
                    "method, and main contribution. Avoid hype."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Title: {paper.title}\n\n"
                    f"Abstract: {paper.abstract}\n\n"
                    f"Topics: {', '.join(paper.matched_topics)}"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 220,
    }

    try:
        data = http_post_json(
            endpoint,
            payload,
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            int(llm_config.get("timeout_seconds", 45)),
        )
        return clean_text(data["choices"][0]["message"]["content"])
    except (KeyError, json.JSONDecodeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"Skipping LLM summary for {paper.arxiv_id}: {exc}", file=sys.stderr)
        return None


def add_optional_llm_summaries(papers: list[Paper], config: dict[str, Any]) -> None:
    llm_config = config.get("llm", {})
    if not bool(llm_config.get("enabled", False)):
        return
    if not env_or_config("OPENAI_API_KEY", ""):
        print("OPENAI_API_KEY is not set; skipping optional LLM summaries.", file=sys.stderr)
        return

    max_papers = int(llm_config.get("max_papers", 8))
    for paper in papers[:max_papers]:
        paper.summary_zh = request_llm_summary(paper, llm_config)


def short_authors(authors: list[str], limit: int = 5) -> str:
    if not authors:
        return "Unknown"
    if len(authors) <= limit:
        return ", ".join(authors)
    return ", ".join(authors[:limit]) + ", et al."


def short_text(value: str, limit: int = 520) -> str:
    value = clean_text(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def format_topic_overview(papers: list[Paper]) -> str:
    counts: dict[str, int] = {}
    for paper in papers:
        for topic in paper.matched_topics:
            counts[topic] = counts.get(topic, 0) + 1
    if not counts:
        return ""

    rows = ["| Topic | Papers |", "| --- | ---: |"]
    for topic, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rows.append(f"| {topic} | {count} |")
    return "\n".join(rows)


def render_markdown(
    papers: list[Paper],
    report_day: date,
    config: dict[str, Any],
    status_note: str | None = None,
) -> str:
    categories = ", ".join(config["arxiv"].get("categories", []))
    lines = [
        f"## Latest Papers ({report_day.isoformat()})",
        "",
        f"Tracked categories: `{categories}`",
        "",
        f"Found `{len(papers)}` matching papers.",
    ]
    if status_note:
        lines.extend(["", f"> {status_note}"])

    overview = format_topic_overview(papers)
    if overview:
        lines.extend(["", "### Topic Overview", "", overview])

    lines.extend(["", "### Papers", ""])
    if not papers:
        lines.append("No matching papers were found for this run.")
        return "\n".join(lines).rstrip()

    for index, paper in enumerate(papers, start=1):
        authors = short_authors(paper.authors)
        topics = ", ".join(paper.matched_topics) or "Unmatched"
        categories = ", ".join(paper.categories) or "Unknown"
        lines.extend(
            [
                f"{index}. **{paper.title}**",
                f"   - Authors: {authors}",
                f"   - arXiv: [{paper.arxiv_id}]({paper.abs_url}) | [PDF]({paper.pdf_url})",
                f"   - Published: `{paper.published[:10]}` | Categories: `{categories}`",
                f"   - Topics: {topics} | Score: `{paper.score}`",
                f"   - Abstract: {short_text(paper.abstract)}",
            ]
        )
        if paper.summary_zh:
            lines.append(f"   - Chinese summary: {paper.summary_zh}")
        lines.append("")

    return "\n".join(lines).rstrip()


def update_readme(path: Path, digest_markdown: str) -> None:
    block = f"{README_START}\n{digest_markdown.rstrip()}\n{README_END}"
    if path.exists():
        current = path.read_text(encoding="utf-8")
    else:
        current = "# arxiv-daily\n\n"

    if README_START in current and README_END in current:
        pattern = re.compile(
            rf"{re.escape(README_START)}.*?{re.escape(README_END)}",
            re.DOTALL,
        )
        updated = pattern.sub(block, current)
    else:
        separator = "\n" if current.endswith("\n") else "\n\n"
        updated = current + separator + block + "\n"

    if updated != current:
        path.write_text(updated, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def paper_from_payload(item: dict[str, Any]) -> Paper:
    allowed_keys = set(Paper.__dataclass_fields__)
    values = {key: value for key, value in item.items() if key in allowed_keys}
    return Paper(**values)


def load_latest_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Warning: failed to parse fallback payload {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
        print(f"Warning: fallback payload {path} has an invalid shape.", file=sys.stderr)
        return None
    return payload


def load_latest_papers(path: Path) -> tuple[list[Paper], str] | None:
    payload = load_latest_payload(path)
    if payload is None:
        return None
    try:
        papers = [paper_from_payload(item) for item in payload["papers"]]
    except TypeError as exc:
        print(f"Warning: fallback payload {path} contains invalid papers: {exc}", file=sys.stderr)
        return None
    return papers, str(payload.get("report_date") or "unknown date")


def build_payload(
    papers: list[Paper],
    report_day: date,
    config: dict[str, Any],
    fallback: bool = False,
    source_report_date: str | None = None,
) -> dict[str, Any]:
    return {
        "report_date": report_day.isoformat(),
        "categories": list(config["arxiv"].get("categories", [])),
        "days_back": int(config["arxiv"].get("days_back", 3)),
        "fallback": fallback,
        "source_report_date": source_report_date,
        "paper_count": len(papers),
        "papers": [asdict(paper) for paper in papers],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and render a daily arXiv digest.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the JSON-syntax YAML config file.",
    )
    parser.add_argument(
        "--date",
        help="Report date in YYYY-MM-DD format. Defaults to now in report_timezone.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rendered digest and JSON payload without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    report_tz = get_report_timezone(str(config.get("report_timezone", "UTC")))
    report_day = (
        date.fromisoformat(args.date)
        if args.date
        else datetime.now(report_tz).date()
    )

    outputs = config["outputs"]
    latest_json_path = Path(outputs["latest_json"])
    status_note = None
    fallback = False
    source_report_date = None

    try:
        entries = fetch_arxiv_entries(config)
        papers = filter_and_sort_papers(entries, config, report_day, report_tz)
        add_optional_llm_summaries(papers, config)
    except RuntimeError as exc:
        fallback_payload = load_latest_papers(latest_json_path)
        if fallback_payload is None:
            raise
        papers, source_report_date = fallback_payload
        fallback = True
        status_note = (
            "arXiv is temporarily unavailable for this run; "
            f"showing the last successful digest from {source_report_date}. "
            f"Original error: {exc}"
        )
        print(f"Warning: using fallback latest.json because arXiv failed: {exc}", file=sys.stderr)

    digest_markdown = render_markdown(papers, report_day, config, status_note=status_note)
    payload = build_payload(
        papers,
        report_day,
        config,
        fallback=fallback,
        source_report_date=source_report_date,
    )

    if args.dry_run:
        print(digest_markdown)
        print("\n--- JSON payload ---")
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    update_readme(Path(outputs["readme"]), digest_markdown)
    if not fallback:
        write_json(latest_json_path, payload)
    archive_path = Path(outputs["archive_dir"]) / f"{report_day.isoformat()}.json"
    write_json(archive_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
