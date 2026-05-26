# CLAUDE.md — Job Application Assistant

## Project Purpose

Learning project combining LangChain, LangGraph, CrewAI, AutoGen, MCP, and A2A in one real system.
Built to understand when and why you use each framework/protocol — not just how.

## User Background

- Knows LangChain and LangGraph already
- Has used MCP before in .NET (give LLM a list of tools, LLM picks one)
- Learning A2A protocol from the sibling repo: `c:\a2a learning project\a2a-mcp-openrouter`
- Prefers explanations from first principles, pointing to specific files and lines
- Uses Obsidian for notes alongside code

## Always Keep These In Sync

When making any code or architecture change, always update:
1. `README.md` — user-facing overview
2. `CLAUDE.md` — this file (Claude Code context)
3. `TODO.md` — mark completed steps, add new ones if needed

## Architecture In One Sentence

An LLM-powered A2A client discovers three specialized agents (LangGraph research, CrewAI writing, AutoGen interview), plans which ones to call based on the user's request, then orchestrates them in sequence — producing a tailored job application package saved to a text file.

## Key Design Decisions

- **MCP server** runs separately on its own port — exposes `web_search` and `fetch_page`
- **Each agent** runs as its own A2A server — different framework, same interface
- **A2A client** is a smart orchestrator: it discovers agents via their agent cards, asks an LLM to plan which agents to call and in what order, then executes the plan
- **The LLM orchestrator** reads agent cards the same way MCP reads tool descriptions — same idea, one level up (agents instead of tools)
- **No agent knows about other agents** — only the client orchestrates
- **LLM provider**: OpenRouter (OpenAI-compatible SDK) — key in `.env`

## Ports

| Service | Port |
|---|---|
| MCP Server | 3000 |
| Research Agent (LangGraph) | 8001 |
| Writing Agent (CrewAI) | 8002 |
| Interview Agent (AutoGen) | 8003 |

## Key Files

| File | Role |
|---|---|
| `.env` | OPENROUTER_API_KEY |
| `mcp_server/__main__.py` | web_search + fetch_page MCP tools |
| `agents/research/agent.py` | LangGraph supervisor + nodes |
| `agents/research/__main__.py` | Wraps LangGraph as A2A agent on port 8001 |
| `agents/writing/agent.py` | CrewAI crew: CV Writer + Cover Letter Writer |
| `agents/writing/__main__.py` | Wraps CrewAI as A2A agent on port 8002 |
| `agents/interview/agent.py` | AutoGen: MockInterviewer + CareerCoach |
| `agents/interview/__main__.py` | Wraps AutoGen as A2A agent on port 8003 |
| `client/__main__.py` | A2A orchestrator, saves output/result.txt |

## Data Flow

```
User CLI (--cv file --job "description")
    ↓
A2A Client
    ↓ fetches /.well-known/agent-card.json from each agent
DISCOVER all running agents
    ↓ agent cards (name, description, skills)
PLAN — LLM reads cards + user request → decides which agents to call + input templates
    ↓ ordered plan with {{variable}} input templates
EXECUTE each step in order:
    ↓ fills {{job_description}} → Research Agent (LangGraph + MCP)
    ↓ research_report added to context
    ↓ fills {{cv}} + {{job_description}} + {{research_report}} → Writing Agent (CrewAI)
    ↓ tailored_cv + cover_letter added to context
    ↓ fills {{job_description}} + {{tailored_cv}} → Interview Agent (AutoGen)
    ↓ interview_transcript added to context
SAVE → output/result.txt
```

## LLM Integration

All agents use OpenRouter via OpenAI-compatible SDK:
```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
```

## What NOT to Change

- Port assignments — client hardcodes the list of known agent URLs to discover
- MCP server must be running before Research Agent starts
- The LLM decides the order — do not hardcode it back

## Running Everything

```bash
# Terminal 1
uv run mcp-server

# Terminal 2
uv run research-agent

# Terminal 3
uv run writing-agent

# Terminal 4
uv run interview-agent

# Terminal 5
uv run client --cv dummy_cv.txt --job "Software Engineer at Google..."
```

## Output

`output/result.txt` — contains all four sections:
1. Research summary
2. Tailored CV
3. Cover letter
4. Mock interview Q&A