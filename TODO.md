# TODO — Job Application Assistant

## Step 1 — Project Setup ✅
- [x] Run `uv init` in the project folder
- [x] Create `.env` file with `OPENROUTER_API_KEY`
- [x] Add all dependencies to `pyproject.toml`:
  - `langchain`
  - `langgraph`
  - `crewai`
  - `pyautogen`
  - `mcp`
  - `a2a-sdk`
  - `openai` (OpenRouter uses OpenAI-compatible SDK)
  - `jinja2`
  - `click`
  - `uvicorn`
- [x] Create `output/` folder
- [x] Create `dummy_cv.txt` with a sample CV for testing

## Step 2 — MCP Tool Server ✅
- [x] Create `mcp_server/__main__.py`
- [x] Implement `web_search(query)` tool — Tavily API, returns top 5 results with titles/URLs/snippets
- [x] Implement `fetch_page(url)` tool — httpx fetch + HTML strip, capped at 5000 chars
- [x] Test: run the MCP server and call tools manually via sse_client
- [x] Understand: this is what you built in .NET, now in Python with the standard protocol

## Step 3 — LangGraph Research Agent (port 8001) ✅
- [x] Create `agents/research/agent.py`
  - [x] Define graph nodes: supervisor_node, search_node, read_node, quality_check_node
  - [x] Define supervisor that routes between nodes via conditional edges
  - [x] Connect to MCP server to use web_search + fetch_page
  - [x] Output: structured research report (company info, role info, salary range)
- [x] Create `agents/research/__main__.py`
  - [x] Wrap LangGraph agent as an A2A agent (a2a-sdk v1.0.3)
  - [x] Serve on port 8001
- [x] Test: send a company name, get back a research report
- [x] Understand: LangGraph supervisor pattern + stateful loops

## Step 4 — CrewAI Writing Agent (port 8002) ✅
- [x] Create `agents/writing/agent.py`
  - [x] Define CV Writer agent with role + goal + backstory
  - [x] Define Cover Letter Writer agent with role + goal + backstory
  - [x] Define tasks: tailor_cv_task, write_cover_letter_task
  - [x] Create Crew that runs both agents in sequence
  - [x] Input: user CV + job description + research report
  - [x] Output: tailored CV text + cover letter text
- [x] Create `agents/writing/__main__.py`
  - [x] Wrap CrewAI crew as an A2A agent
  - [x] Serve on port 8002
- [ ] Test: send CV + JD, get back tailored CV + cover letter
- [ ] Understand: CrewAI role-based agents and task handoff

## Step 5 — AutoGen Interview Agent (port 8003) ✅
- [x] Create `agents/interview/agent.py`
  - [x] Define MockInterviewer agent (asks tough questions based on JD)
  - [x] Define CareerCoach agent (gives feedback + model answers)
  - [x] Set up conversation loop between them (3-4 rounds)
  - [x] Input: job description + tailored CV
  - [x] Output: interview Q&A transcript
- [x] Create `agents/interview/__main__.py`
  - [x] Wrap AutoGen conversation as an A2A agent
  - [x] Serve on port 8003
- [ ] Test: send JD + CV, get back interview transcript
- [ ] Understand: AutoGen conversational multi-agent pattern

## Step 6 — A2A Client Orchestrator (Smart LLM-powered) ✅
- [x] Create `client/__main__.py`
  - [x] Accept `--cv` and `--job` CLI arguments
  - [x] Discover all 3 agents via their AgentCards
  - [x] Send to Research Agent → get research report
  - [x] Send to Writing Agent (with research) → get CV + cover letter
  - [x] Send to Interview Agent (with tailored CV) → get interview Q&A
  - [x] Combine all outputs into one result
  - [x] Save to `output/result.txt`
- [ ] Test: full end-to-end with dummy CV + real job description
- [ ] Understand: A2A orchestration across different frameworks

## Step 7 — End-to-End Test
- [ ] Start all 4 servers (MCP + 3 agents)
- [ ] Run client with dummy CV + a real job posting
- [ ] Check `output/result.txt`
- [ ] Verify each agent did its job correctly
- [ ] Debug any issues

## Step 8 — Study Checkpoint (after everything works)
- [ ] Can you explain what MCP does without looking at notes?
- [ ] Can you explain the difference between A2A and MCP?
- [ ] Can you explain when to use LangGraph vs CrewAI vs AutoGen?
- [ ] Read the LangGraph supervisor code and trace the flow
- [ ] Read the CrewAI crew code and identify where the handoff happens
- [ ] Read the AutoGen code and see where the conversation loop is
