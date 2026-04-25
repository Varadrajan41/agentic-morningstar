"""
RSS feed ingestion.

Fetches entries from subscribed RSS/Atom feeds, scores them for relevance,
and embeds high-quality articles into the knowledge base — exactly the same
pipeline as web content ingestion, but driven by a list of saved feed URLs
rather than DuckDuckGo search results.

Feed subscriptions are persisted in rss_feeds.json so they survive restarts.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import feedparser

from src.ingestion.web_ingestion import score_and_embed_web_content
from src.tools.llm_tools import generate_feed_context
from src.utils.logger import get_logger

logger = get_logger(__name__)

FEEDS_FILE = Path("./rss_feeds.json")


# ---------------------------------------------------------------------------
# Feed persistence helpers
# ---------------------------------------------------------------------------

def load_saved_feeds() -> List[Dict[str, str]]:
    """
    Load the list of saved feed subscriptions from disk.

    Returns:
        List of dicts with keys: url, name, description, added_at
    """
    if not FEEDS_FILE.exists():
        return []
    try:
        return json.loads(FEEDS_FILE.read_text())
    except Exception:
        return []


def save_feeds(feeds: List[Dict[str, str]]) -> None:
    """Persist the full feed list to disk."""
    FEEDS_FILE.write_text(json.dumps(feeds, indent=2, ensure_ascii=False))


def add_feed(url: str, name: str = "", description: str = "") -> Dict[str, str]:
    """
    Add a new feed to the saved list (no-op if URL already present).

    Args:
        url: RSS/Atom feed URL
        name: Human-readable feed name (auto-detected if empty)
        description: Optional description

    Returns:
        The new feed entry dict
    """
    feeds = load_saved_feeds()
    if any(f["url"] == url for f in feeds):
        logger.info(f"Feed already subscribed: {url}")
        return next(f for f in feeds if f["url"] == url)

    entry = {
        "url": url,
        "name": name or url,
        "description": description,
        "added_at": datetime.now().strftime("%Y-%m-%d"),
    }
    feeds.append(entry)
    save_feeds(feeds)
    logger.info(f"✅ Subscribed to feed: {name or url}")
    return entry


def remove_feed(url: str) -> None:
    """Remove a feed subscription by URL."""
    feeds = [f for f in load_saved_feeds() if f["url"] != url]
    save_feeds(feeds)
    logger.info(f"🗑️ Removed feed: {url}")


# ---------------------------------------------------------------------------
# Feed metadata preview
# ---------------------------------------------------------------------------

def preview_feed(url: str, auto_detect_context: bool = True) -> Dict[str, str]:
    """
    Fetch feed metadata and auto-detect a scoring context using the LLM.

    Args:
        url:                  RSS/Atom feed URL
        auto_detect_context:  If True, call the LLM to generate a topic context
                              from the first 8 entry titles. Set to False in
                              unit tests or where LLM is unavailable.

    Returns:
        Dict with: title, subtitle, url, entry_count, auto_context, error (if any)
        `auto_context` is a short phrase describing the feed's topic — ready to
        use as the `query_context` in `ingest_rss_feed`, with no user input needed.
    """
    try:
        parsed = feedparser.parse(url)
        if parsed.bozo and not parsed.entries:
            return {"error": f"Could not parse feed: {parsed.bozo_exception}"}

        feed_title = parsed.feed.get("title", url)
        entry_titles = [e.get("title", "") for e in parsed.entries[:8] if e.get("title")]

        auto_context = feed_title  # safe fallback
        if auto_detect_context and entry_titles:
            try:
                auto_context = generate_feed_context(feed_title, entry_titles)
                logger.info(f"Auto-detected context for '{feed_title}': {auto_context}")
            except Exception as e:
                logger.warning(f"Could not auto-detect context for {url}: {e}")

        return {
            "title": feed_title,
            "subtitle": parsed.feed.get("subtitle", parsed.feed.get("description", "")),
            "url": url,
            "entry_count": len(parsed.entries),
            "auto_context": auto_context,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_rss_feed(
    feed_url: str,
    max_items: int = 10,
    query_context: str = "",
    min_score: int = 7,
    progress_callback: Optional[Callable[[int, int, str, bool], None]] = None,
) -> Dict[str, Any]:
    """
    Ingest entries from a single RSS/Atom feed into the knowledge base.

    Each entry is passed through `score_and_embed_web_content`, which:
    - Extracts the full article text (trafilatura)
    - Scores relevance via LLM (1-10)
    - Embeds and stores if score ≥ min_score

    RSS uses a slightly lower default threshold (7) than ad-hoc web search (8)
    because subscribed feeds are already topic-curated by the user.

    Args:
        feed_url:          RSS/Atom URL to parse
        max_items:         Maximum number of entries to process
        query_context:     Topic context for scoring (defaults to feed title)
        min_score:         Minimum LLM relevance score to embed (1-10)
        progress_callback: Optional fn(current, total, title, embedded) for UI

    Returns:
        Dict with: feed_title, total_fetched, embedded, rejected, skipped, errors
    """
    logger.info(f"📡 Fetching RSS feed: {feed_url}")

    try:
        parsed = feedparser.parse(feed_url)
    except Exception as e:
        logger.warning(f"RSS parse error for {feed_url}: {e}")
        return {
            "feed_title": feed_url,
            "total_fetched": 0,
            "embedded": 0,
            "rejected": 0,
            "skipped": 0,
            "errors": 1,
        }

    feed_title = parsed.feed.get("title", feed_url)
    context = query_context or feed_title
    entries = parsed.entries[:max_items]
    total = len(entries)

    logger.info(f"  → {total} entries from '{feed_title}'")

    embedded = rejected = skipped = errors = 0

    for i, entry in enumerate(entries, 1):
        title = entry.get("title", "Untitled")
        url = entry.get("link", "")
        snippet = entry.get("summary", entry.get("description", ""))

        if not url:
            errors += 1
            continue

        try:
            result = score_and_embed_web_content(
                title=title,
                url=url,
                snippet=snippet,
                query_context=context,
                min_score=min_score,
            )

            if result.get("skipped"):
                skipped += 1
                logger.debug(f"  ⏭️  Already ingested: {title[:60]}")
            elif result.get("embedded"):
                embedded += 1
                logger.info(f"  ✅ Embedded: {title[:60]} (score: {result['score']}/10)")
            else:
                rejected += 1
                logger.debug(f"  🗑️  Rejected: {title[:60]} (score: {result.get('score', 0)}/10)")

        except Exception as e:
            errors += 1
            logger.warning(f"  ⚠️  Error processing '{title[:40]}': {e}")
            result = {}

        if progress_callback:
            progress_callback(i, total, title, result.get("embedded", False))

    stats = {
        "feed_title": feed_title,
        "total_fetched": total,
        "embedded": embedded,
        "rejected": rejected,
        "skipped": skipped,
        "errors": errors,
    }
    logger.info(
        f"✅ Feed done: {embedded} embedded / {rejected} rejected / "
        f"{skipped} skipped / {errors} errors"
    )
    return stats


def ingest_all_saved_feeds(
    max_items_per_feed: int = 10,
    min_score: int = 7,
    progress_callback: Optional[Callable[[str, int, int, str, bool], None]] = None,
) -> Dict[str, Any]:
    """
    Ingest all saved feed subscriptions in sequence.

    Args:
        max_items_per_feed: Entries to process per feed
        min_score:          Minimum score threshold
        progress_callback:  Optional fn(feed_url, current, total, title, embedded)

    Returns:
        Aggregated stats: feeds_processed, total_embedded, total_rejected, total_skipped
    """
    feeds = load_saved_feeds()
    if not feeds:
        logger.info("No saved feeds to ingest.")
        return {"feeds_processed": 0, "total_embedded": 0, "total_rejected": 0, "total_skipped": 0}

    total_embedded = total_rejected = total_skipped = 0

    for feed in feeds:
        url = feed["url"]
        context = feed.get("description") or feed.get("name", "")

        def _per_entry_cb(current, total, title, embedded, _url=url):
            if progress_callback:
                progress_callback(_url, current, total, title, embedded)

        stats = ingest_rss_feed(
            feed_url=url,
            max_items=max_items_per_feed,
            query_context=context,
            min_score=min_score,
            progress_callback=_per_entry_cb,
        )
        total_embedded += stats["embedded"]
        total_rejected += stats["rejected"]
        total_skipped += stats["skipped"]

    return {
        "feeds_processed": len(feeds),
        "total_embedded": total_embedded,
        "total_rejected": total_rejected,
        "total_skipped": total_skipped,
    }
