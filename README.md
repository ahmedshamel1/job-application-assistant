# Job Application Assistant

A learning project that combines LangChain, LangGraph, CrewAI, AutoGen, MCP, and A2A protocols in one real system.

## What It Does

User provides their CV + a job description → system produces:
1. A tailored CV
2. A personalized cover letter
3. A mock interview Q&A transcript

## Architecture

```
User (CLI)
    ↓ CV + Job Description
[A2A Client — smart orchestrator]
    ↓ 1. DISCOVER — fetches agent cards from all running agents
    ↓ 2. PLAN     — LLM reads cards, decides which agents to call + in what order
    ↓ 3. EXECUTE  — calls agents in planned order, passing outputs forward
    │
    ├──→ [Agent 1: LangGraph]  → researches company + role → research report
    │         uses MCP tools: web_search, fetch_page
    ├──→ [Agent 2: CrewAI]     → tailored CV + cover letter
    │         CV Writer role + Cover Letter Writer role
    └──→ [Agent 3: AutoGen]    → mock interview transcript
              MockInterviewer + CareerCoach conversation
```

The client works like MCP but one level up — instead of an LLM choosing which **tool** to call, an LLM chooses which **agent** to call based on their agent cards.

## Why Each Framework

| Agent | Framework | Reason |
|---|---|---|
| Research | LangGraph | Needs a loop: search → read → decide → repeat |
| Writing | CrewAI | Two distinct roles with clear handoff |
| Interview | AutoGen | Naturally a back-and-forth conversation |
| Tools | MCP | Standard protocol: web_search + fetch_page |
| Orchestration | A2A | Connects agents built in different frameworks |

## Project Structure

```
job-application-assistant/
├── .env                         ← OPENROUTER_API_KEY
├── pyproject.toml
├── CLAUDE.md                    ← Claude Code context
├── TODO.md                      ← Implementation checklist
├── output/                      ← Final text file saved here
├── mcp_server/
│   └── __main__.py              ← web_search + fetch_page tools
├── agents/
│   ├── research/                ← LangGraph agent (port 8001)
│   │   ├── __main__.py
│   │   └── agent.py
│   ├── writing/                 ← CrewAI agent (port 8002)
│   │   ├── __main__.py
│   │   └── agent.py
│   └── interview/               ← AutoGen agent (port 8003)
│       ├── __main__.py
│       └── agent.py
└── client/
    └── __main__.py              ← A2A orchestrator, saves output to file
```

## Running the Project

```bash
# Terminal 1 — MCP tool server
uv run mcp-server

# Terminal 2 — Research agent (LangGraph)
uv run research-agent

# Terminal 3 — Writing agent (CrewAI)
uv run writing-agent

# Terminal 4 — Interview agent (AutoGen)
uv run interview-agent

# Terminal 5 — Run the client
uv run client --cv dummy_cv.txt --job "job description here"
```

## Output

Results are saved to `output/result.txt` containing:
- Company research summary
- Tailored CV
- Cover letter
- Mock interview Q&A

## LLM Provider

Uses OpenRouter with OpenAI-compatible SDK. Set your key in `.env`:
```
OPENROUTER_API_KEY=your_key_here
```

## What You Learn From Each Piece

- **MCP**: How to expose tools in a standard way any framework can consume
- **LangGraph**: Supervisor pattern, stateful loops, conditional branching
- **CrewAI**: Role-based agents, task handoff between crew members
- **AutoGen**: Agent-to-agent conversation loops
- **LangChain**: The base layer that LangGraph builds on (you see it implicitly)
- **A2A**: How agents built in different frameworks discover and call each other