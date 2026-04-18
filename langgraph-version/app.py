"""
Streamlit UI for Agentic Morningstar - LangGraph Version.

Mirrors the original Project Morningstar UI but with agentic backend.
Features real-time agent reasoning display like Claude/ChatGPT.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import MorningstarState, QueryIntent
from src.main import build_graph
from src.tools.chroma_tools import get_chroma_manager
from src.config import LLM_MODEL

# Node icons for visual feedback
NODE_ICONS = {
    "router": "🎯",
    "librarian": "📚",
    "analyst": "🔍",
    "web_scout": "🌐",
    "writer": "✍️",
    "__start__": "🚀",
    "__end__": "✅"
}

NODE_LABELS = {
    "router": "Understanding Query",
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

# --- CHAT INTERFACE ---

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session-1"

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show reasoning if available
        if message.get("reasoning") and show_reasoning:
            with st.expander("🧠 Agent Reasoning"):
                for step in message["reasoning"]:
                    st.markdown(f"• {step}")
        
        # Show sources if available
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.markdown(f"• {src}")

# Chat input
if prompt := st.chat_input("Ask Agentic Morningstar..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response with real-time streaming
    with st.chat_message("assistant"):
        # Build graph
        workflow = build_graph()
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
        
        # Prepare state
        initial_state: MorningstarState = {
            "query": prompt,
            "rewritten_query": "",
            "query_intent": None,
            "collections_tried": [],
            "retrieved_docs": [],
            "web_search_performed": False,
            "web_results": [],
            "confidence_score": 0.0,
            "needs_web_fallback": enable_web_fallback,
            "synthesized_answer": "",
            "citations": [],
            "retry_count": 0,
            "should_retry": False,
            "messages": [{"role": "user", "content": m["content"]} for m in st.session_state.messages],
            "agent_reasoning": []
        }
        
        # Override collections based on user selection
        if not use_daily and not use_deep:
            st.warning("Please select at least one collection")
        else:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Container for real-time reasoning
            reasoning_container = st.container()
            
            # Placeholder for final answer
            answer_placeholder = st.empty()
            
            # Track all reasoning steps
            all_reasoning = []
            final_answer = ""
            final_sources = []
            
            try:
                # Stream the agent execution in real-time
                for event in app.stream(initial_state, config=config):
                    # Get node name and state update
                    node_name = list(event.keys())[0]
                    state_update = event[node_name]
                    
                    # Skip start/end nodes for display
                    if node_name in ["__start__", "__end__"]:
                        continue
                    
                    # Get icon and label
                    icon = NODE_ICONS.get(node_name, "⚙️")
                    label = NODE_LABELS.get(node_name, node_name.replace("_", " ").title())
                    
                    # Get new reasoning steps from this node
                    new_reasoning = state_update.get("agent_reasoning", [])
                    
                    # Find reasoning that hasn't been displayed yet
                    if new_reasoning:
                        # Find new steps (beyond what we've already seen)
                        displayed_count = len(all_reasoning)
                        new_steps = new_reasoning[displayed_count:]
                        
                        if new_steps:
                            # Update our tracking
                            all_reasoning.extend(new_steps)
                            
                            # Display real-time status
                            with reasoning_container:
                                with st.status(f"{icon} {label}...", expanded=True):
                                    for step in new_steps:
                                        st.markdown(f"• {step}")
                    
                    # Check for final answer
                    if state_update.get("synthesized_answer"):
                        final_answer = state_update["synthesized_answer"]
                    
                    if state_update.get("citations"):
                        final_sources = state_update["citations"]
                
                # Display final answer
                if final_answer:
                    answer_placeholder.markdown(final_answer)
                    
                    # Show reasoning summary in expander
                    if show_reasoning and all_reasoning:
                        with st.expander("🧠 Complete Agent Reasoning Chain"):
                            for i, step in enumerate(all_reasoning, 1):
                                st.markdown(f"**{i}.** {step}")
                    
                    # Show sources
                    if final_sources:
                        with st.expander("📚 Sources"):
                            for src in final_sources:
                                st.markdown(f"• {src}")
                    
                    # Add to history
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

# --- FOOTER ---
st.markdown("---")
st.caption("Built with LangGraph | Local LLM via Ollama | Agentic RAG | Real-time Reasoning")
