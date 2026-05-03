"""
Streamlit UI for Agentic Morningstar - LangGraph Version.

Mirrors the original Project Morningstar UI but with agentic backend.
Features real-time agent reasoning display like Claude/ChatGPT.
"""
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env before any LangChain/LangGraph imports so LANGCHAIN_* vars are set
from dotenv import load_dotenv
load_dotenv()

import sqlite3
import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from src.graph.state import MorningstarState, QueryIntent
from src.main import build_graph
from src.ingestion.backup import export_knowledge_base, list_backups


@st.cache_resource
def get_compiled_app():
    """Build and compile the LangGraph app once, reused across all queries.

    Uses SqliteSaver so conversation memory persists across Streamlit restarts.
    check_same_thread=False is required because Streamlit is multi-threaded.
    """
    workflow = build_graph()
    conn = sqlite3.connect("./morningstar_memory.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)
from src.tools.chroma_tools import get_chroma_manager
from src.ingestion.web_ingestion import ingest_single_topic
from src.ingestion.arxiv_fetcher import daily_arxiv_ingest
from src.ingestion.rss_fetcher import (
    load_saved_feeds, save_feeds, add_feed, remove_feed,
    preview_feed, ingest_rss_feed, ingest_all_saved_feeds,
)
from src.tools.llm_tools import generate_arxiv_query, synthesize_answer_stream
from src.config import LLM_MODEL, ENABLE_SMART_WEB_INGESTION, ARXIV_MAX_RESULTS, ARXIV_MIN_SCORE

# ---------------------------------------------------------------------------
# Session management helpers
# Sessions are persisted in sessions.json so they survive Streamlit restarts.
# Each session maps to a unique LangGraph thread_id (SqliteSaver checkpoint).
# ---------------------------------------------------------------------------

MAX_HISTORY = 20  # max messages passed to the graph (prevents unbounded token growth)
SESSIONS_FILE = Path("./sessions.json")


def _load_sessions():
    if not SESSIONS_FILE.exists():
        return []
    try:
        return json.loads(SESSIONS_FILE.read_text())
    except Exception:
        return []


def _save_sessions(sessions: list) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2, ensure_ascii=False))


def _save_current_session() -> None:
    """Persist the active session (name + messages) to sessions.json."""
    thread_id = st.session_state.get("thread_id")
    if not thread_id:
        return
    sessions = _load_sessions()
    name = st.session_state.get("session_name", "Unnamed Session")
    messages = st.session_state.get("messages", [])
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for s in sessions:
        if s["id"] == thread_id:
            s["name"] = name
            s["messages"] = messages
            s["updated_at"] = now
            break
    else:
        sessions.append({
            "id": thread_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "messages": messages,
        })
    _save_sessions(sessions)


def _start_new_session(name: str = "New Session") -> None:
    """Save current session and start a blank one."""
    _save_current_session()
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.session_name = name


def _switch_session(session: dict) -> None:
    """Save current session then load a saved one."""
    _save_current_session()
    st.session_state.thread_id = session["id"]
    st.session_state.messages = session.get("messages", [])
    st.session_state.session_name = session["name"]


def _delete_session(session_id: str) -> None:
    sessions = [s for s in _load_sessions() if s["id"] != session_id]
    _save_sessions(sessions)


# Node icons for visual feedback
NODE_ICONS = {
    "router": "🎯",
    "decomposer": "✂️",
    "librarian": "📚",
    "analyst": "🔍",
    "web_scout": "🌐",
    "writer": "✍️",
    "__start__": "🚀",
    "__end__": "✅"
}

NODE_LABELS = {
    "router": "Understanding Query",
    "decomposer": "Decomposing Query",
    "librarian": "Searching Knowledge Base",
    "analyst": "Evaluating Results",
    "web_scout": "Searching Web",
    "writer": "Synthesizing Answer",
    "__start__": "Starting",
    "__end__": "Complete"
}

# Page Setup
st.set_page_config(page_title="Agentic Morningstar AI", page_icon="🌅", layout="wide")
st.title("🌅 Agentic Morningstar (LangGraph)")
st.caption("Autonomous Research Assistant with Agentic RAG - Real-time Reasoning")

# --- SIDEBAR ---
with st.sidebar:
    # ── Session Manager (must be first so thread_id/messages are set before anything reads them) ──
    st.subheader("🗂️ Sessions")

    # Bootstrap session state on first run
    if "session_name" not in st.session_state:
        st.session_state.session_name = "Session 1"
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Current session: editable name + save button
    col_name, col_save = st.columns([3, 1])
    with col_name:
        edited_name = st.text_input(
            "Session name",
            value=st.session_state.session_name,
            label_visibility="collapsed",
            key="session_name_input",
        )
        if edited_name != st.session_state.session_name:
            st.session_state.session_name = edited_name
    with col_save:
        if st.button("💾", help="Save this session"):
            _save_current_session()
            st.toast("Session saved!")

    if st.button("➕ New Session", use_container_width=True):
        _start_new_session()
        st.rerun()

    # List saved sessions (excluding the one currently active)
    _all_sessions = _load_sessions()
    _other_sessions = [s for s in _all_sessions if s["id"] != st.session_state.thread_id]
    if _other_sessions:
        st.caption("**Saved sessions:**")
        for _s in reversed(_other_sessions[-6:]):   # newest first, show up to 6
            _col_lbl, _col_load, _col_del = st.columns([4, 1, 1])
            with _col_lbl:
                st.caption(f"**{_s['name']}** · {_s.get('updated_at', _s.get('created_at', ''))}")
            with _col_load:
                if st.button("📂", key=f"load_{_s['id']}", help="Switch to this session"):
                    _switch_session(_s)
                    st.rerun()
            with _col_del:
                if st.button("🗑️", key=f"del_{_s['id']}", help="Delete this session"):
                    _delete_session(_s["id"])
                    st.rerun()

    st.markdown("---")

    st.header("⚙️ Configuration")

    # Collection selection
    st.subheader("Search Collections")
    use_daily = st.checkbox("Fast Cards (daily_research)", value=True)
    use_deep = st.checkbox("Deep Dive (deep_dive_research)", value=True)
    
    st.markdown("---")
    
    # Agent settings
    st.subheader("Agent Settings")
    show_reasoning = st.checkbox("Show Agent Reasoning", value=True)
    enable_web_fallback = st.checkbox("Enable Web Fallback", value=True)
    smart_ingest = st.checkbox(
        "🧠 Smart Web Learning", 
        value=ENABLE_SMART_WEB_INGESTION,
        help="Automatically add high-quality web results to knowledge base"
    )
    
    st.markdown("---")
    
    # Knowledge Ingestion
    st.subheader("📥 Knowledge Ingestion")
    
    # Manual topic learning
    learn_topic = st.text_input("Learn about topic:", placeholder="Enter topic to research...")
    if st.button("🎓 Learn Now", disabled=not learn_topic):
        with st.spinner(f"Learning about: {learn_topic}..."):
            try:
                stats = ingest_single_topic(learn_topic)
                if stats["embedded"] > 0:
                    st.success(f"✅ Learned {stats['embedded']} new sources about '{learn_topic}'!")
                else:
                    st.info(f"No high-quality sources found for '{learn_topic}'")
            except Exception as e:
                st.error(f"Error during learning: {e}")
    
    # ArXiv ingestion with natural language query generation
    st.markdown("**📚 ArXiv Paper Ingestion**")
    
    # Natural language topic input
    research_topic = st.text_input(
        "What topic are you researching?",
        placeholder="e.g., multi-agent RAG systems for healthcare, transformer models for medical diagnosis",
        help="Describe your research interest in plain English. The AI will convert it to an ArXiv query."
    )
    
    # Optional priority keywords
    priority_keywords = st.text_input(
        "Priority keywords (optional):",
        placeholder="e.g., LangGraph, retrieval, clinical",
        help="Comma-separated keywords that should be given preference in the search"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            arxiv_max_results = st.slider(
                "Max papers:",
                min_value=5,
                max_value=50,
                value=ARXIV_MAX_RESULTS,
                step=5
            )
        with col2:
            arxiv_min_score = st.slider(
                "Quality threshold:",
                min_value=1,
                max_value=10,
                value=ARXIV_MIN_SCORE,
                help="Only papers scoring above this will be embedded"
            )
        sort_by_relevance = st.checkbox(
            "Sort by relevance (not date)",
            value=False,
            help=(
                "OFF (default): Fetches the 20 newest papers — best for daily ingestion of recent research.\n\n"
                "ON: Fetches papers most relevant to your query — use this when searching for a specific paper "
                "like 'AWQ paper' or 'BERT paper' that may be older."
            )
        )
    
    # Generate and preview query
    # Cached query key: invalidated when topic or keywords change
    topic_key = f"{research_topic}|{priority_keywords}"
    if st.session_state.get("arxiv_query_key") != topic_key:
        st.session_state.pop("arxiv_generated_query", None)

    if research_topic.strip():
        if st.button("🔍 Generate ArXiv Query"):
            with st.spinner("Generating ArXiv query from your topic..."):
                try:
                    generated_query = generate_arxiv_query(
                        research_topic=research_topic,
                        priority_keywords=priority_keywords
                    )
                    st.session_state["arxiv_generated_query"] = generated_query
                    st.session_state["arxiv_query_key"] = topic_key
                except Exception as e:
                    st.error(f"Error generating query: {e}")

        # Editable query box — always visible once generated, user can simplify before fetching
        if "arxiv_generated_query" in st.session_state and st.session_state.get("arxiv_query_key") == topic_key:
            edited_query = st.text_area(
                "ArXiv Query (edit before fetching if needed):",
                value=st.session_state["arxiv_generated_query"],
                height=80,
                help="Tip: If you get 0 results, simplify to fewer AND terms. E.g.: \"AWQ\" OR \"activation-aware weight quantization\""
            )
            # Keep edited version in sync
            st.session_state["arxiv_generated_query"] = edited_query

        if st.button("📥 Fetch ArXiv Papers"):
            try:
                if "arxiv_generated_query" in st.session_state and st.session_state.get("arxiv_query_key") == topic_key:
                    arxiv_query = st.session_state["arxiv_generated_query"]
                else:
                    with st.spinner("Generating ArXiv query..."):
                        arxiv_query = generate_arxiv_query(
                            research_topic=research_topic,
                            priority_keywords=priority_keywords
                        )
                st.code(f"Query: {arxiv_query}", language="text")

                # Progress bar UI elements
                progress_bar = st.progress(0, text="Fetching papers from ArXiv...")
                status_text = st.empty()

                def on_paper_progress(current, total, title, embedded):
                    pct = current / total
                    icon = "⭐" if embedded else "🗑️"
                    progress_bar.progress(pct, text=f"Processing paper {current}/{total}...")
                    status_text.caption(f"{icon} {title[:70]}...")

                stats = daily_arxiv_ingest(
                    query=arxiv_query,
                    max_results=arxiv_max_results,
                    min_score=arxiv_min_score,
                    sort_by_relevance=sort_by_relevance,
                    query_context=research_topic,
                    progress_callback=on_paper_progress
                )

                progress_bar.progress(1.0, text="Done!")
                status_text.empty()

                if stats["embedded"] > 0:
                    st.success(f"✅ Ingested {stats['embedded']} high-quality papers from {stats['total_fetched']} fetched!")
                elif stats["total_fetched"] == 0:
                    st.warning("⚠️ ArXiv returned 0 papers for this query.")
                    st.caption("The generated query may be too specific or have invalid syntax. Try clicking 'Preview ArXiv Query' first to review it, or broaden your topic description.")
                    # Clear cached query so next fetch generates a fresh one
                    st.session_state.pop("arxiv_generated_query", None)
                else:
                    st.info(f"No papers met the quality threshold (≥{arxiv_min_score}/10) from {stats['total_fetched']} fetched")
                    st.caption("Try broadening your topic or lowering the quality threshold.")

            except Exception as e:
                st.error(f"Error during ArXiv ingestion: {e}")
    else:
        st.info("👆 Enter a research topic above to generate an ArXiv query")
    
    st.markdown("---")

    # RSS Feed subscriptions
    st.subheader("📡 RSS Feed Subscriptions")

    # Initialise session flag so the feed list re-renders after add/remove
    if "rss_feeds_version" not in st.session_state:
        st.session_state.rss_feeds_version = 0

    saved_feeds = load_saved_feeds()

    # --- Add new feed ---
    with st.expander("➕ Add New Feed", expanded=not saved_feeds):
        new_feed_url = st.text_input(
            "Feed URL:",
            placeholder="https://example.com/feed.xml",
            key="rss_new_url",
        )
        if new_feed_url.strip():
            if st.button("🔍 Preview Feed"):
                with st.spinner("Fetching feed & auto-detecting topic context…"):
                    info = preview_feed(new_feed_url.strip(), auto_detect_context=True)
                if "error" in info:
                    st.error(f"Could not read feed: {info['error']}")
                else:
                    st.session_state["rss_preview"] = info
                    st.success(f"**{info['title']}** — {info['entry_count']} entries")
                    if info.get("subtitle"):
                        st.caption(info["subtitle"])

            preview = st.session_state.get("rss_preview", {})

            feed_name = st.text_input(
                "Feed name (optional):",
                value=preview.get("title", ""),
                key="rss_new_name",
            )

            # Show auto-detected context as the default; user can override it
            auto_ctx = preview.get("auto_context", "")
            if auto_ctx and auto_ctx != preview.get("title", ""):
                st.caption(f"🤖 Auto-detected topic: **{auto_ctx}**")

            feed_desc = st.text_input(
                "Topic context for scoring:",
                value=auto_ctx,
                placeholder="e.g., machine learning, AI safety",
                key="rss_new_desc",
                help=(
                    "Used by the LLM to score each article's relevance. "
                    "Auto-filled from feed content — edit if needed."
                ),
            )
            if st.button("✅ Subscribe"):
                add_feed(
                    url=new_feed_url.strip(),
                    name=feed_name.strip() or new_feed_url.strip(),
                    description=feed_desc.strip(),
                )
                st.session_state["rss_preview"] = {}
                st.session_state.rss_feeds_version += 1
                st.success("Subscribed!")
                st.rerun()

    # --- Saved feeds list ---
    saved_feeds = load_saved_feeds()  # reload after possible add
    if saved_feeds:
        rss_max_items = st.slider(
            "Max items per feed:", min_value=5, max_value=50, value=10, step=5, key="rss_max"
        )
        rss_min_score = st.slider(
            "Quality threshold:", min_value=1, max_value=10, value=7, key="rss_score",
            help="Only articles scoring ≥ this are embedded (RSS default is 7, slightly lower than web search 8 because feeds are already topic-curated)"
        )

        if st.button("📥 Ingest All Feeds"):
            all_progress = st.progress(0, text="Starting...")
            all_status = st.empty()
            feeds_done = [0]
            n_feeds = len(saved_feeds)

            def on_all_progress(feed_url, current, total, title, embedded):
                pct = (feeds_done[0] + current / total) / n_feeds
                all_progress.progress(min(pct, 1.0), text=f"Feed {feeds_done[0]+1}/{n_feeds}: {title[:50]}...")

            agg = ingest_all_saved_feeds(
                max_items_per_feed=rss_max_items,
                min_score=rss_min_score,
                progress_callback=on_all_progress,
            )
            all_progress.progress(1.0, text="Done!")
            all_status.success(
                f"✅ All feeds done — **{agg['total_embedded']}** embedded / "
                f"{agg['total_rejected']} rejected / {agg['total_skipped']} already known"
            )

        st.markdown("**Subscribed feeds:**")
        for feed in saved_feeds:
            col_info, col_ingest, col_del = st.columns([4, 1, 1])
            with col_info:
                st.markdown(f"**{feed['name']}**")
                if feed.get("description"):
                    st.caption(feed["description"])
                st.caption(f"Added {feed.get('added_at', '?')} · {feed['url'][:50]}...")

            with col_ingest:
                if st.button("📥", key=f"rss_ingest_{feed['url']}", help="Ingest this feed"):
                    prog = st.progress(0, text="Fetching...")
                    stat = st.empty()

                    def _cb(cur, tot, title, emb, _p=prog, _s=stat):
                        _p.progress(cur / tot, text=f"{title[:40]}...")
                        _s.caption(f"{'✅' if emb else '🗑️'} {title[:50]}")

                    res = ingest_rss_feed(
                        feed_url=feed["url"],
                        max_items=rss_max_items,
                        query_context=feed.get("description") or feed["name"],
                        min_score=rss_min_score,
                        progress_callback=_cb,
                    )
                    prog.progress(1.0, text="Done!")
                    stat.success(
                        f"**{res['embedded']}** embedded / {res['rejected']} rejected / "
                        f"{res['skipped']} skipped from {res['feed_title']}"
                    )

            with col_del:
                if st.button("🗑️", key=f"rss_del_{feed['url']}", help="Remove this feed"):
                    remove_feed(feed["url"])
                    st.session_state.rss_feeds_version += 1
                    st.rerun()
    else:
        st.info("No feeds subscribed yet. Add a feed URL above.")

    st.markdown("---")

    # Database stats
    st.subheader("💽 Database Stats")
    try:
        chroma = get_chroma_manager()
        stats = chroma.get_collection_stats()
        for name, count in stats.items():
            st.metric(name, count)
    except Exception as e:
        st.warning(f"DB not initialized: {e}")
    
    st.markdown("---")
    st.markdown(f"**LLM:** {LLM_MODEL}")

# --- MAIN TABS ---
chat_tab, kb_tab = st.tabs(["💬 Chat", "🗄️ Knowledge Base"])

# --- KNOWLEDGE BASE BROWSER TAB ---
with kb_tab:
    st.subheader("🗄️ Knowledge Base Browser")
    st.caption("Browse, inspect, and delete documents stored in your local knowledge base.")

    try:
        chroma = get_chroma_manager()
        kb_stats = chroma.get_collection_stats()

        col1, col2 = st.columns(2)
        for i, (name, count) in enumerate(kb_stats.items()):
            (col1 if i % 2 == 0 else col2).metric(name, f"{count} docs")

        if st.button("🔄 Refresh KB Stats"):
            st.rerun()

        st.markdown("---")

        for coll_name, collection in chroma.collections.items():
            total = kb_stats.get(coll_name, 0)
            st.subheader(f"📂 {coll_name}  ({total} documents)")

            if total == 0:
                st.info("No documents in this collection yet.")
                continue

            data = collection.get(include=["metadatas"])
            ids = data["ids"]
            metas = data["metadatas"]

            # Search filter
            search_filter = st.text_input(
                f"Filter {coll_name}:", placeholder="Type to filter by title...",
                key=f"filter_{coll_name}"
            )

            shown = 0
            for doc_id, meta in zip(ids, metas):
                title = meta.get("title", "Untitled")
                if search_filter and search_filter.lower() not in title.lower():
                    continue
                shown += 1
                score = meta.get("score", "?")
                doc_type = meta.get("type", "unknown")
                date = meta.get("date_ingested", "unknown")
                url = meta.get("url", "")

                with st.expander(f"{'⭐' if isinstance(score, (int,float)) and score >= 7 else '📄'} {title[:80]}  |  Score: {score}/10  |  {doc_type}"):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.caption(f"**Date ingested:** {date}")
                        if url:
                            st.caption(f"**Source:** [{url[:60]}...]({url})" if len(url) > 60 else f"**Source:** [{url}]({url})")
                        if meta.get("ai_summary"):
                            st.markdown(f"**Summary:** {meta['ai_summary']}")
                    with col_b:
                        if st.button("🗑️ Delete", key=f"del_{doc_id}"):
                            collection.delete(ids=[doc_id])
                            st.success(f"Deleted: {title[:40]}...")
                            st.rerun()

            if search_filter and shown == 0:
                st.info(f"No documents matching '{search_filter}' in {coll_name}.")

            st.markdown("---")

        # --- Backup / Export section ---
        st.subheader("💾 Backup & Export")
        col_exp, col_dl = st.columns([2, 1])

        with col_exp:
            if st.button("📤 Export Knowledge Base to JSON"):
                with st.spinner("Exporting..."):
                    try:
                        stats = export_knowledge_base()
                        st.success(
                            f"✅ Exported **{stats['total_docs']} docs** → `{stats['path']}` "
                            f"({stats['size_kb']} KB)"
                        )
                    except Exception as ex:
                        st.error(f"Export failed: {ex}")

        with col_dl:
            backups = list_backups()
            if backups:
                st.caption(f"**{len(backups)} backup(s)** in `morningstar_backups/`")
                for b in backups[:3]:  # show latest 3
                    # Offer in-browser download of the backup JSON
                    try:
                        backup_bytes = open(b["path"], "rb").read()
                        st.download_button(
                            label=f"⬇️ {b['modified']} ({b['size_kb']} KB)",
                            data=backup_bytes,
                            file_name=b["name"],
                            mime="application/json",
                            key=f"dl_{b['name']}"
                        )
                    except OSError:
                        st.caption(f"  {b['name']} ({b['size_kb']} KB)")
            else:
                st.caption("No backups yet.")

    except Exception as e:
        st.error(f"Could not load knowledge base: {e}")

# --- CHAT INTERFACE TAB ---
with chat_tab:

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("reasoning") and show_reasoning:
                with st.expander("🧠 Agent Reasoning"):
                    for step in message["reasoning"]:
                        st.markdown(f"• {step}")

            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for src in message["sources"]:
                        st.markdown(f"• {src}")

    # Chat input
    if prompt := st.chat_input("Ask Agentic Morningstar..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            app = get_compiled_app()

            initial_state: MorningstarState = {
                "query": prompt,
                "rewritten_query": "",
                "query_intent": None,
                "sub_queries": [],
                "keywords": [],
                "primary_terms": [],
                "is_time_sensitive": False,
                "collections_tried": [],
                "retrieved_docs": [],
                "web_search_performed": False,
                "web_search_count": 0,
                "web_results": [],
                "confidence_score": 0.0,
                "needs_web_fallback": enable_web_fallback,
                "smart_ingest_enabled": smart_ingest,
                "synthesized_answer": "",
                "citations": [],
                "retry_count": 0,
                "should_retry": False,
                "messages": [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-MAX_HISTORY:]
                ],
                "agent_reasoning": []
            }

            if not use_daily and not use_deep:
                st.warning("Please select at least one collection")
            else:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                reasoning_container = st.container()
                answer_placeholder = st.empty()
                all_reasoning = []
                final_answer = ""
                final_sources = []
                final_confidence = 0.0
                used_web = False

                # Track retrieved docs / web_results so we can stream the synthesis
                final_retrieved_docs = []
                final_web_results = []
                final_time_sensitive = False

                try:
                    for event in app.stream(initial_state, config=config):
                        node_name = list(event.keys())[0]
                        state_update = event[node_name]

                        if node_name in ["__start__", "__end__"]:
                            continue

                        icon = NODE_ICONS.get(node_name, "⚙️")
                        label = NODE_LABELS.get(node_name, node_name.replace("_", " ").title())
                        new_reasoning = state_update.get("agent_reasoning", [])

                        if new_reasoning:
                            displayed_count = len(all_reasoning)
                            new_steps = new_reasoning[displayed_count:]
                            if new_steps:
                                all_reasoning.extend(new_steps)
                                with reasoning_container:
                                    with st.status(f"{icon} {label}...", expanded=True):
                                        for step in new_steps:
                                            st.markdown(f"• {step}")

                        citations_update = state_update.get("citations")
                        if citations_update is not None:
                            final_sources = citations_update
                        if state_update.get("confidence_score") is not None:
                            final_confidence = state_update["confidence_score"]
                        if state_update.get("web_search_performed"):
                            used_web = True
                        if state_update.get("is_time_sensitive") is not None:
                            final_time_sensitive = state_update["is_time_sensitive"]
                        # Use `is not None` so we correctly capture analyst setting
                        # retrieved_docs=[] (empty list is falsy but meaningful).
                        retrieved = state_update.get("retrieved_docs")
                        if retrieved is not None:
                            final_retrieved_docs = retrieved
                        web = state_update.get("web_results")
                        if web is not None:
                            final_web_results = web

                    # --- Token-level streaming synthesis ---
                    with reasoning_container:
                        with st.status("✍️ Writing answer...", expanded=True):
                            st.markdown("• Synthesizing response from retrieved context")

                    answer_placeholder.markdown("▌")
                    final_answer = ""
                    streamed_citations = []

                    web_for_stream = final_web_results if final_web_results else None
                    for chunk in synthesize_answer_stream(
                        query=prompt,
                        documents=final_retrieved_docs,
                        web_results=web_for_stream,
                        is_time_sensitive=final_time_sensitive
                    ):
                        if isinstance(chunk, dict):
                            # Last item — citations emitted by the generator
                            streamed_citations = chunk.get("citations", [])
                        else:
                            final_answer += chunk
                            answer_placeholder.markdown(final_answer + "▌")

                    answer_placeholder.markdown(final_answer)
                    # Prefer graph citations (richer metadata); fall back to streamed ones
                    if not final_sources:
                        final_sources = streamed_citations

                    if final_answer:
                        # Confidence + source indicator
                        conf_pct = int(final_confidence * 100)
                        source_label = "🌐 Web" if used_web else "📚 Local KB"
                        if final_confidence >= 0.7:
                            conf_color = "🟢"
                        elif final_confidence >= 0.4:
                            conf_color = "🟡"
                        else:
                            conf_color = "🔴"
                        st.caption(
                            f"{conf_color} Confidence: **{conf_pct}%** &nbsp;|&nbsp; Source: {source_label}"
                            + (" *(low confidence triggered web fallback)*" if used_web and final_confidence < 0.6 else "")
                        )
                        st.progress(final_confidence)

                        if show_reasoning and all_reasoning:
                            with st.expander("🧠 Complete Agent Reasoning Chain"):
                                for i, step in enumerate(all_reasoning, 1):
                                    st.markdown(f"**{i}.** {step}")
                        if final_sources:
                            with st.expander("📚 Sources"):
                                for src in final_sources:
                                    st.markdown(f"• {src}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_answer,
                            "reasoning": all_reasoning if show_reasoning else [],
                            "sources": final_sources
                        })
                    else:
                        answer_placeholder.warning("No answer generated. The agent may not have found relevant information.")

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {e}"
                    })

    # --- Export current session ---
    if st.session_state.get("messages"):
        st.markdown("---")
        export_data = json.dumps(
            {
                "session_id": st.session_state.thread_id,
                "session_name": st.session_state.get("session_name", "Session"),
                "exported_at": datetime.now().isoformat(),
                "messages": st.session_state.messages,
            },
            indent=2,
            ensure_ascii=False,
        )
        st.download_button(
            label="📤 Export Chat",
            data=export_data,
            file_name=f"morningstar_{st.session_state.get('session_name', 'chat').replace(' ', '_')}.json",
            mime="application/json",
            help="Download this conversation as JSON",
        )

# --- FOOTER ---
st.markdown("---")
st.caption("Built with LangGraph | Local LLM via Ollama | Agentic RAG | Real-time Reasoning")
