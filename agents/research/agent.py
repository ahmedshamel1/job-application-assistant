"""
LangGraph Research Agent

Graph structure (this is the "supervisor pattern"):

    START
      ↓
  supervisor  ← ← ← ← ← ←
      ↓                    ↑
   (routes)          search_node
      ↓                    ↑
      ├── "search" ────────┘
      ├── "read"  ─── read_node ──┐
      │                           │
      └── "done"  ─── quality_check_node ── END
                         (formats report)

The supervisor is an LLM call that reads what has been collected so far
and decides the next step. This is the core LangGraph supervisor pattern:
one node routes, others execute, all loop back.
"""

import json
import os
import re

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.types import TextContent
from openai import OpenAI
from typing_extensions import TypedDict

load_dotenv()

MCP_URL = "http://localhost:3000/sse"
MODEL = os.getenv("RESEARCH_MODEL", "openai/gpt-4o-mini")


# ── State ──────────────────────────────────────────────────────────────────
# LangGraph passes this dict between every node. Each node returns a partial
# dict; LangGraph merges it into the state. Nothing here is LangChain-specific —
# StateGraph works with plain TypedDicts.

class ResearchState(TypedDict):
    task: str               # "Research {company} for {role} role"
    search_results: list    # accumulated web_search outputs
    page_summaries: list    # accumulated fetch_page outputs (truncated)
    next: str               # "search" | "read" | "done"
    next_input: str         # query string (for search) or URL (for read)
    report: str             # filled in when next == "done"
    steps: int              # safety counter


# ── MCP helper ─────────────────────────────────────────────────────────────

async def _call_mcp(tool_name: str, args: dict) -> str:
    """Open a fresh MCP session, call one tool, return the text result."""
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            for c in result.content:
                if isinstance(c, TextContent):
                    return c.text
    return ""


# ── LLM helper ─────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


# ── Graph nodes ────────────────────────────────────────────────────────────

def supervisor_node(state: ResearchState) -> dict:
    """
    The supervisor: reads everything collected so far, decides next action.

    Returns JSON with one of:
      {"action": "search", "input": "<query>"}
      {"action": "read",   "input": "<url>"}
      {"action": "done",   "report": "<full report text>"}

    This is the heart of the supervisor pattern — one node that controls routing.
    """
    collected = ""
    if state["search_results"]:
        collected += "\n\nSEARCH RESULTS COLLECTED:\n" + "\n---\n".join(state["search_results"][-3:])
    if state["page_summaries"]:
        collected += "\n\nPAGES READ:\n" + "\n---\n".join(state["page_summaries"][-2:])

    prompt = f"""You are a research agent investigating a job opportunity.

TASK: {state["task"]}

INFORMATION COLLECTED SO FAR:{collected if collected else " (nothing yet)"}

STEPS TAKEN: {state["steps"]}

Decide the next action. You have two tools:
- web_search: searches the web, takes a query string
- fetch_page: fetches a URL, takes a full URL

If you have enough information (company overview, role details, salary range, recent news),
write the final report.

Respond with ONLY valid JSON in one of these formats:

To search:
{{"action": "search", "input": "<search query>"}}

To read a URL from search results:
{{"action": "read", "input": "<full URL>"}}

To finish (when you have all 4 sections):
{{"action": "done", "report": "<full markdown report with 4 sections: Company Overview, Role Details, Salary Range, Recent News>"}}
"""

    response = _call_llm(prompt)

    # Extract JSON — LLMs sometimes wrap it in ```json ... ```
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return {"next": "done", "report": response, "steps": state["steps"] + 1}

    data = json.loads(match.group())
    action = data.get("action", "done")

    if action == "search":
        return {"next": "search", "next_input": data["input"], "steps": state["steps"] + 1}
    if action == "read":
        return {"next": "read", "next_input": data["input"], "steps": state["steps"] + 1}

    # action == "done"
    return {"next": "done", "report": data.get("report", ""), "steps": state["steps"] + 1}


async def search_node(state: ResearchState) -> dict:
    """Calls web_search on the MCP server. Returns accumulated results."""
    result = await _call_mcp("web_search", {"query": state["next_input"]})
    return {"search_results": [*state["search_results"], f"Query: {state['next_input']}\n{result}"]}


async def read_node(state: ResearchState) -> dict:
    """Calls fetch_page on the MCP server. Stores a short summary."""
    result = await _call_mcp("fetch_page", {"url": state["next_input"]})
    summary = result[:800]  # keep summaries short so the supervisor prompt stays manageable
    return {"page_summaries": [*state["page_summaries"], f"URL: {state['next_input']}\n{summary}"]}


def quality_check_node(state: ResearchState) -> dict:
    """
    Final node — if the supervisor produced a report, pass it through.
    If not (hit step limit), ask the LLM to compile from what was collected.
    """
    if state.get("report"):
        return {"report": state["report"]}

    # Safety fallback: compile whatever was gathered
    collected = "\n\n".join(state["search_results"] + state["page_summaries"])
    prompt = f"""Based on this research data, write a structured report about: {state["task"]}

DATA:
{collected[:3000]}

Write a report with these sections:
## Company Overview
## Role Details
## Salary Range
## Recent News
"""
    return {"report": _call_llm(prompt)}


# ── Routing ────────────────────────────────────────────────────────────────

def _route(state: ResearchState) -> str:
    """Called after supervisor_node — picks which node runs next."""
    if state["steps"] >= 7:  # hard cap to avoid runaway loops
        return "quality_check"
    return state["next"]


# ── Graph assembly ─────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("search", search_node)
    g.add_node("read", read_node)
    g.add_node("quality_check", quality_check_node)

    g.set_entry_point("supervisor")

    # Supervisor routes to one of three destinations
    g.add_conditional_edges(
        "supervisor",
        _route,
        {"search": "search", "read": "read", "done": "quality_check"},
    )

    # Workers always return to supervisor for the next decision
    g.add_edge("search", "supervisor")
    g.add_edge("read", "supervisor")

    # quality_check is terminal
    g.add_edge("quality_check", END)

    return g.compile()


# ── Public API ─────────────────────────────────────────────────────────────

async def run_research(task: str) -> str:
    """
    Entry point called by the A2A executor.
    task: plain text, e.g. "Research Google for the role of Software Engineer in London"
    Returns: markdown research report
    """
    graph = build_graph()
    initial_state: ResearchState = {
        "task": task,
        "search_results": [],
        "page_summaries": [],
        "next": "search",
        "next_input": "",
        "report": "",
        "steps": 0,
    }
    final_state = await graph.ainvoke(initial_state)
    return final_state["report"]
