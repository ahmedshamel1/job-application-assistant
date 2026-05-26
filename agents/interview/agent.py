"""
AutoGen Interview Agent

AutoGen works differently from both LangGraph and CrewAI:
  - No graph, no state dict, no tasks
  - You define Agents (ConversableAgents with system messages)
  - You put them in a Team (RoundRobinGroupChat)
  - The team runs a conversation loop automatically
  - Agents take turns speaking until a termination condition is met

Structure here:

  MockInterviewer  ←→  CareerCoach
      asks question        gives model answer + feedback
      ← repeats 4 rounds →

The key difference from CrewAI:
  CrewAI = sequential tasks (Agent A does task 1, Agent B does task 2)
  AutoGen = conversation loop (Agent A talks to Agent B, they go back and forth)
"""

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("INTERVIEW_MODEL", "openai/gpt-4o-mini")


def _make_client() -> OpenAIChatCompletionClient:
    """
    AutoGen 0.7+ uses OpenAIChatCompletionClient for OpenAI-compatible APIs.
    We point it at OpenRouter using base_url override.
    model_info is required when using a non-standard model name (openai/ prefix).
    """
    return OpenAIChatCompletionClient(
        model=MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )


async def run_interview(job_description: str, tailored_cv: str) -> str:
    """
    Entry point called by the A2A executor.

    job_description: the job posting text
    tailored_cv:     the tailored CV from the Writing Agent

    Returns: full interview transcript as a markdown string
    """
    client = _make_client()

    # ── Agents ─────────────────────────────────────────────────────────────
    # Each agent has a system_message that defines its persona and behaviour.
    # Unlike CrewAI where you fill in role/goal/backstory and CrewAI builds the prompt,
    # here you write the full system_message string yourself — more control, less structure.

    interviewer = AssistantAgent(
        name="MockInterviewer",
        model_client=client,
        system_message=f"""You are a tough but fair technical interviewer at a company hiring for this role:

{job_description}

Your job:
- Ask ONE specific interview question per turn
- Alternate between technical questions (coding, system design, architecture) and behavioural questions (STAR-format situations)
- Base your questions on the job description requirements
- After the CareerCoach responds, ask the next question
- Do NOT give feedback yourself — that is the CareerCoach's job
- After 4 questions, say "INTERVIEW_COMPLETE" to end the session

Start by asking your first question now.""",
    )

    coach = AssistantAgent(
        name="CareerCoach",
        model_client=client,
        system_message=f"""You are an experienced career coach helping a candidate prepare for interviews.

The candidate's CV:
{tailored_cv}

Your job:
- For each question the MockInterviewer asks, provide:
  1. A MODEL ANSWER: a strong, specific answer the candidate should give (2-3 sentences, concrete details)
  2. KEY TIPS: 2 bullet points on what makes this answer strong
- Keep each response under 150 words
- Do NOT ask questions yourself — only respond to the interviewer's questions
- When the interviewer says "INTERVIEW_COMPLETE", write a brief summary of the session""",
    )

    # ── Team ────────────────────────────────────────────────────────────────
    # RoundRobinGroupChat: agents take turns in order (interviewer → coach → interviewer → ...)
    # MaxMessageTermination: stops after N total messages (4 questions × 2 agents = 8 messages,
    # plus the initial task message = 9 total, add buffer → 12)
    # We also use the interviewer's "INTERVIEW_COMPLETE" signal as a soft stop.

    termination = MaxMessageTermination(max_messages=12)

    team = RoundRobinGroupChat(
        participants=[interviewer, coach],
        termination_condition=termination,
    )

    # ── Run ─────────────────────────────────────────────────────────────────
    # team.run(task=...) starts the conversation.
    # The task message goes to the first agent (interviewer) as the opening prompt.
    # result.messages contains every message exchanged, in order.

    result = await team.run(
        task="Begin the interview. Ask your first question.",
    )

    # ── Format transcript ───────────────────────────────────────────────────
    lines = ["# Mock Interview Transcript\n"]
    for msg in result.messages:
        # result.messages contains both agent messages and system events.
        # We only want messages that have a source (agent name) and content.
        if hasattr(msg, "source") and hasattr(msg, "content"):
            source = msg.source
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.strip():
                lines.append(f"**{source}:** {content}\n")

    return "\n".join(lines)