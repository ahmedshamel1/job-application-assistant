"""
A2A Client — LLM-powered Smart Orchestrator

How this works:
  1. DISCOVER — fetch the agent card from each known agent URL
     (agent card = the agent's "business card": name, description, what it can do)
  2. PLAN — send all agent cards + user request to an LLM
     LLM reads the cards and decides: which agents to call, in what order, with what input
  3. EXECUTE — run the plan step by step
     Each step's output is stored and can be used as input to the next step
  4. SAVE — write everything to output/result.txt

This is the "smart orchestrator" pattern:
  MCP = LLM chooses which TOOL to call
  A2A = LLM chooses which AGENT to call
The LLM reads the agent cards the same way it reads tool descriptions in MCP.
"""

import asyncio
import json
import os
import re
import uuid
from pathlib import Path

import click
import httpx
from dotenv import load_dotenv
from openai import OpenAI

from a2a.client import ClientConfig, create_client
from a2a.types.a2a_pb2 import Message, Part, Role, SendMessageRequest

load_dotenv()

KNOWN_AGENT_URLS = [
    "http://localhost:8001",   # Research Agent (LangGraph)
    "http://localhost:8002",   # Writing Agent (CrewAI)
    "http://localhost:8003",   # Interview Agent (AutoGen)
]

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "openai/gpt-4o-mini")


# ── Step 1: Discover ────────────────────────────────────────────────────────

async def fetch_agent_cards(http: httpx.AsyncClient) -> list[dict]:
    """
    Fetch the agent card from every known agent URL.
    Agent cards live at /.well-known/agent-card.json — standard A2A path.
    Skips agents that are not running (connection error).
    """
    cards = []
    for url in KNOWN_AGENT_URLS:
        try:
            resp = await http.get(f"{url}/.well-known/agent-card.json", timeout=5)
            resp.raise_for_status()
            card = resp.json()
            card["_url"] = url  # attach the base URL so the LLM can reference it
            cards.append(card)
            print(f"  Discovered: {card.get('name', url)} at {url}")
        except Exception:
            print(f"  Skipped {url} — not reachable")
    return cards


# ── Step 2: Plan ────────────────────────────────────────────────────────────

def plan_with_llm(user_request: str, has_cv: bool, agent_cards: list[dict]) -> list[dict]:
    """
    Send agent cards + user request to an LLM.
    LLM returns an ordered plan: which agents to call and what input to send each.

    Input templates use {{variable}} placeholders.
    Available variables:
      {{cv}}                  — user's CV text (if provided)
      {{job_description}}     — user's job description
      {{research_report}}     — output from Research Agent
      {{tailored_cv}}         — output from Writing Agent
      {{cover_letter}}        — output from Writing Agent
      {{interview_transcript}}— output from Interview Agent
    """
    # Build a readable summary of each agent for the LLM
    agents_text = ""
    for card in agent_cards:
        name = card.get("name", "Unknown Agent")
        description = card.get("description", "")
        url = card["_url"]
        skills = card.get("skills", [])
        skill_lines = "\n".join(
            f"    - {s.get('name', '')}: {s.get('description', '')}"
            for s in skills
        )
        agents_text += f"\n[{name}]\nURL: {url}\nDescription: {description}\nSkills:\n{skill_lines}\n"

    user_inputs = "job description"
    if has_cv:
        user_inputs = "CV text and job description"

    prompt = f"""You are an orchestration planner for a job application system.

You have access to these A2A agents:
{agents_text}

The user has provided: {user_inputs}
User request: {user_request}

Decide which agents to call and in what order. You do NOT have to call all agents.
Only call agents that are relevant to the user's request.

When writing input templates, use these {{{{variable}}}} placeholders:
  {{{{cv}}}}                   — user's CV text
  {{{{job_description}}}}      — the job description
  {{{{research_report}}}}      — output from Research Agent (available after it runs)
  {{{{tailored_cv}}}}          — output from Writing Agent (available after it runs)
  {{{{cover_letter}}}}         — output from Writing Agent (available after it runs)
  {{{{interview_transcript}}}} — output from Interview Agent (available after it runs)

Respond with ONLY valid JSON in this format:
{{
  "reasoning": "one sentence explaining your plan",
  "steps": [
    {{
      "url": "<agent base url>",
      "name": "<agent name>",
      "input": "<full input text using {{{{variable}}}} placeholders>"
    }}
  ]
}}"""

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    resp = client.chat.completions.create(
        model=ORCHESTRATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content

    # Extract JSON — LLMs sometimes wrap in ```json ... ```
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return valid JSON. Got:\n{raw}")

    plan = json.loads(match.group())
    return plan


# ── Step 3: Execute ─────────────────────────────────────────────────────────

def _fill_template(template: str, context: dict[str, str]) -> str:
    """Replace {{variable}} placeholders with actual values from context."""
    result = template
    for key, value in context.items():
        result = result.replace(f"{{{{{key}}}}}", value)
    return result


async def send_and_collect(agent_url: str, text: str, http: httpx.AsyncClient) -> dict[str, str]:
    """
    Send a text message to one A2A agent and collect all artifact outputs.
    Returns {artifact_name: text_content}.
    """
    client = create_client(
        agent=agent_url,
        client_config=ClientConfig(
            streaming=False,
            httpx_client=http,
        ),
        resolver_http_kwargs={"verify": False},
    )

    request = SendMessageRequest(
        message=Message(
            message_id=str(uuid.uuid4()),
            context_id=str(uuid.uuid4()),
            role=Role.ROLE_USER,
            parts=[Part(text=text)],
        )
    )

    artifacts: dict[str, str] = {}

    async for event in await client.send_message(request):
        which = event.WhichOneof("event")

        if which == "artifact_update":
            art = event.artifact_update.artifact
            name = art.name or "output"
            text_parts = [p.text for p in art.parts if p.HasField("text")]
            if text_parts:
                artifacts[name] = "\n".join(text_parts)

        elif which == "task":
            for art in event.task.artifacts:
                name = art.name or "output"
                text_parts = [p.text for p in art.parts if p.HasField("text")]
                if text_parts:
                    artifacts[name] = "\n".join(text_parts)

    return artifacts


async def execute_plan(
    steps: list[dict],
    cv_text: str,
    job_description: str,
    http: httpx.AsyncClient,
) -> dict[str, str]:
    """
    Run each step in order.
    Context grows as agents return outputs — later steps can use earlier outputs.
    """
    # Start with what the user provided
    context: dict[str, str] = {
        "cv": cv_text,
        "job_description": job_description,
    }

    for i, step in enumerate(steps, 1):
        name = step["name"]
        url = step["url"]
        input_template = step["input"]

        # Fill in any {{variable}} placeholders with what we have so far
        input_text = _fill_template(input_template, context)

        print(f"\n  [{i}/{len(steps)}] Calling {name}...")
        outputs = await send_and_collect(url, input_text, http)

        # Add all outputs to context so next steps can use them
        context.update(outputs)

        for key, val in outputs.items():
            print(f"    Got '{key}' ({len(val)} chars)")

    return context


# ── Step 4: Save ────────────────────────────────────────────────────────────

def save_results(context: dict[str, str], plan: dict) -> Path:
    """Write all collected outputs to output/result.txt."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    result_path = output_dir / "result.txt"

    sections = []
    sections.append(f"{'='*60}\nJOB APPLICATION PACKAGE\n{'='*60}\n")
    sections.append(f"Orchestration plan: {plan.get('reasoning', '')}\n")

    label_map = {
        "research_report":     "RESEARCH REPORT",
        "tailored_cv":         "TAILORED CV",
        "cover_letter":        "COVER LETTER",
        "interview_transcript": "MOCK INTERVIEW TRANSCRIPT",
    }

    for key, label in label_map.items():
        if key in context and context[key].strip():
            sections.append(f"\n{'='*60}\n{label}\n{'='*60}\n\n{context[key]}")

    result_path.write_text("\n".join(sections), encoding="utf-8")
    return result_path


# ── Orchestration entry point ───────────────────────────────────────────────

async def run(cv_text: str, job_description: str) -> None:
    async with httpx.AsyncClient(verify=False, timeout=300) as http:

        # 1. Discover which agents are running
        print("\n[Discover] Fetching agent cards...")
        agent_cards = await fetch_agent_cards(http)
        if not agent_cards:
            print("No agents found. Make sure the agent servers are running.")
            return

        # 2. Ask LLM to plan the execution
        print("\n[Plan] Asking LLM to decide which agents to call...")
        plan = plan_with_llm(
            user_request=job_description,
            has_cv=bool(cv_text.strip()),
            agent_cards=agent_cards,
        )
        print(f"  Reasoning: {plan.get('reasoning', '')}")
        print(f"  Steps: {[s['name'] for s in plan.get('steps', [])]}")

        # 3. Execute the plan
        print("\n[Execute] Running agents...")
        context = await execute_plan(
            steps=plan["steps"],
            cv_text=cv_text,
            job_description=job_description,
            http=http,
        )

        # 4. Save everything
        result_path = save_results(context, plan)
        print(f"\n[Done] Saved to: {result_path.resolve()}")


# ── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--cv", default="", type=click.Path(), help="Path to your CV text file (optional)")
@click.option("--job", required=True, help="Job description text or path to a .txt file")
def main(cv: str, job: str):
    """
    Smart A2A orchestrator — discovers agents, asks an LLM to plan, executes the plan.

    Examples:
        uv run client --cv dummy_cv.txt --job "Senior Backend Engineer at Stripe..."
        uv run client --job "Just research Stripe for me"
    """
    cv_text = ""
    if cv:
        cv_path = Path(cv)
        if cv_path.exists():
            cv_text = cv_path.read_text(encoding="utf-8")

    job_path = Path(job)
    job_description = job_path.read_text(encoding="utf-8") if job_path.exists() else job

    if cv_text:
        print(f"CV: {cv} ({len(cv_text)} chars)")
    print(f"Job: {len(job_description)} chars")

    asyncio.run(run(cv_text, job_description))


if __name__ == "__main__":
    main()