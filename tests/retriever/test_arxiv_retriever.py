"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser
import pytest

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, RSS_EMPTY_RETRIES, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]


# ---------------------------------------------------------------------------
# Batch retry on retryable HTTP errors (429/500/503)
# ---------------------------------------------------------------------------


def _fake_results_for(entries):
    fake_results = []
    for entry in entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))
    return fake_results


def _new_entries(parsed):
    return [e for e in parsed.entries if e.get("arxiv_announce_type", "new") == "new"]


def test_retrieve_raw_papers_retries_on_503(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)

    calls = {"count": 0}

    class FlakyClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            calls["count"] += 1
            if calls["count"] == 1:
                raise arxiv_retriever.arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 503)
            return iter(_fake_results_for(_new_entries(mock_feedparser)))

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FlakyClient)

    retriever = ArxivRetriever(config)
    raw_papers = retriever._retrieve_raw_papers()
    assert calls["count"] == 2
    assert len(raw_papers) == len(_new_entries(mock_feedparser))


def test_retrieve_raw_papers_raises_when_503_persists(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)

    class Always503Client:
        def __init__(self, **kw):
            pass

        def results(self, search):
            raise arxiv_retriever.arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 503)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", Always503Client)

    retriever = ArxivRetriever(config)
    with pytest.raises(arxiv_retriever.arxiv.HTTPError):
        retriever._retrieve_raw_papers()


# ---------------------------------------------------------------------------
# RSS empty-feed retry
# ---------------------------------------------------------------------------


def test_rss_empty_feed_is_retried(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)

    parse_calls = {"count": 0}
    empty_feed = SimpleNamespace(feed=SimpleNamespace(title="ok"), entries=[])

    def _parse_then_fixture(url, *args, **kwargs):
        parse_calls["count"] += 1
        if parse_calls["count"] == 1:
            return empty_feed
        return mock_feedparser

    monkeypatch.setattr(arxiv_retriever.feedparser, "parse", _parse_then_fixture)

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            return iter(_fake_results_for(_new_entries(mock_feedparser)))

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    retriever = ArxivRetriever(config)
    raw_papers = retriever._retrieve_raw_papers()
    assert parse_calls["count"] == 2
    assert len(raw_papers) == len(_new_entries(mock_feedparser))


def test_rss_empty_feed_accepted_after_retries(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)

    empty_feed = SimpleNamespace(feed=SimpleNamespace(title="ok"), entries=[])
    parse_calls = {"count": 0}

    def _always_empty(url, *args, **kwargs):
        parse_calls["count"] += 1
        return empty_feed

    monkeypatch.setattr(arxiv_retriever.feedparser, "parse", _always_empty)

    retriever = ArxivRetriever(config)
    assert retriever._retrieve_raw_papers() == []
    assert parse_calls["count"] == RSS_EMPTY_RETRIES
