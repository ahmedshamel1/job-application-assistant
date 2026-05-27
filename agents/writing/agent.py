"""
CrewAI Writing Agent

CrewAI works differently from LangGraph:
  - No graph, no state dict, no routing
  - You define Agents (personas with role/goal/backstory)
  - You define Tasks (what each agent must produce)
  - You create a Crew that runs tasks in sequence
  - Each task's output is automatically passed to the next task

Structure here:

  cv_writer_agent    → tailor_cv_task    → tailored CV text
        ↓ output passed automatically
  cover_letter_agent → cover_letter_task → cover letter text

The Crew runs both tasks sequentially and returns a CrewOutput object.
"""

import os

from crewai import Agent, Crew, Task, LLM
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("WRITING_MODEL", "openai/gpt-4o-mini")


def _make_llm() -> LLM:
    """
    CrewAI has its own LLM wrapper. We point it at OpenRouter
    by overriding the base_url and using the openai/ model prefix.
    """
    return LLM(
        model=MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


async def run_writing(cv: str, job_description: str, research_report: str) -> tuple[str, str]:
    """
    Entry point called by the A2A executor.

    cv:               raw text of the candidate's current CV
    job_description:  the job posting text
    research_report:  markdown report from the Research Agent

    Returns: (tailored_cv, cover_letter) — both as plain text strings
    """
    llm = _make_llm()

    # ── Agents ─────────────────────────────────────────────────────────────
    # Each agent is a persona. CrewAI uses the role/goal/backstory to
    # build the system prompt for that agent automatically.

    cv_writer = Agent(
        role="Expert CV Writer",
        goal=(
            "Tailor the candidate's CV to match the job description exactly. "
            "Highlight relevant skills and experience. Keep it concise and ATS-friendly."
        ),
        backstory=(
            "You are a senior technical recruiter turned CV coach with 10 years of experience "
            "helping engineers land roles at top tech companies. You know exactly which keywords "
            "ATS systems look for and how to frame backend engineering experience for maximum impact."
        ),
        llm=llm,
        verbose=True,
    )

    cover_letter_writer = Agent(
        role="Professional Cover Letter Writer",
        goal=(
            "Write a compelling, personalised cover letter that connects the candidate's "
            "background to this specific company and role."
        ),
        backstory=(
            "You are a career coach who has helped hundreds of engineers write cover letters "
            "that actually get read. You use research about the company to make every letter "
            "feel personal, not templated. You write in a direct, confident tone — no clichés."
        ),
        llm=llm,
        verbose=True,
    )

    # ── Tasks ───────────────────────────────────────────────────────────────
    # Tasks define WHAT to produce and WHO produces it.
    # The description is the user prompt; expected_output tells CrewAI what
    # a complete result looks like (used for quality checking internally).

    tailor_cv_task = Task(
        description=f"""Tailor the following CV for the job description below.

CURRENT CV:
{cv}

JOB DESCRIPTION:
{job_description}

COMPANY RESEARCH (use this to understand what the company values):
{research_report}

Instructions:
- Reorder and rewrite bullet points to match the JD's priorities
- Add relevant keywords from the JD naturally into the text
- Remove or de-emphasise experience irrelevant to this role
- Keep the same sections: Summary, Experience, Skills, Education, Projects
- Output the full tailored CV as plain text""",
        expected_output="A complete tailored CV in plain text, ready to paste into a document.",
        agent=cv_writer,
    )

    cover_letter_task = Task(
        description=f"""Write a cover letter for the job application below.

JOB DESCRIPTION:
{job_description}

COMPANY RESEARCH (use specific facts from here to personalise the letter):
{research_report}

TAILORED CV (use this to know what to highlight):
{{tailor_cv_task.output}}

Instructions:
- Opening: mention the specific role and one concrete reason you want THIS company
- Middle: connect 2-3 of the candidate's strongest achievements to what the JD asks for
- Closing: confident call to action, no desperate language
- Length: 3 paragraphs, no longer than 300 words
- Tone: direct and professional, not formal or stiff
- Output the full cover letter as plain text""",
        expected_output="A 3-paragraph cover letter in plain text, under 300 words.",
        agent=cover_letter_writer,
        context=[tailor_cv_task],  # explicitly tells CrewAI this task needs tailor_cv_task's output
    )

    # ── Crew ────────────────────────────────────────────────────────────────
    # process="sequential" means tasks run in order: tailor_cv first, then cover_letter.
    # CrewAI automatically passes the output of each task to the next one.

    crew = Crew(
        agents=[cv_writer, cover_letter_writer],
        tasks=[tailor_cv_task, cover_letter_task],
        verbose=True,
    )

    result = await crew.kickoff_async()

    # result.tasks_output is a list of TaskOutput objects, one per task.
    # .raw gives the plain string output.
    tailored_cv = result.tasks_output[0].raw
    cover_letter = result.tasks_output[1].raw

    return tailored_cv, cover_letter
