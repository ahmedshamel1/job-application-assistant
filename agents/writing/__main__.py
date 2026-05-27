"""
A2A server wrapping the CrewAI writing agent.  (a2a-sdk v1.0.3)

Input format (plain text from client):
    CV:
    <cv text>

    JOB DESCRIPTION:
    <jd text>

    RESEARCH REPORT:
    <report text>

Output: two artifacts — tailored_cv and cover_letter
"""

import click
import uvicorn
from starlette.applications import Starlette
from typing import override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue_v2 import EventQueue
from a2a.server.request_handlers.default_request_handler_v2 import DefaultRequestHandlerV2
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)

from agents.writing.agent import run_writing


def _parse_input(text: str) -> tuple[str, str, str]:
    """
    Split the delimited input block into its three parts.
    Falls back to the whole text if delimiters aren't found.
    """
    cv, jd, report = "", "", ""

    if "CV:" in text and "JOB DESCRIPTION:" in text:
        parts = text.split("JOB DESCRIPTION:")
        cv = parts[0].replace("CV:", "").strip()
        remainder = parts[1]

        if "RESEARCH REPORT:" in remainder:
            jd_part, report_part = remainder.split("RESEARCH REPORT:", 1)
            jd = jd_part.strip()
            report = report_part.strip()
        else:
            jd = remainder.strip()
    else:
        # Fallback: treat whole input as job description, no CV or report
        jd = text

    return cv, jd, report


class WritingAgentExecutor(AgentExecutor):
    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        task = Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
        )
        await event_queue.enqueue_event(task)
        await updater.start_work()

        cv, jd, report = _parse_input(user_input)
        tailored_cv, cover_letter = await run_writing(cv, jd, report)

        await updater.add_artifact(
            parts=[Part(text=tailored_cv)],
            name="tailored_cv",
            last_chunk=False,
        )
        await updater.add_artifact(
            parts=[Part(text=cover_letter)],
            name="cover_letter",
            last_chunk=True,
        )

        await updater.complete()

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=8002)
def main(host: str, port: int):
    skill = AgentSkill(
        id="write_application_documents",
        name="Write CV and Cover Letter",
        description=(
            "Takes a candidate CV, job description, and research report. "
            "Returns a tailored CV and a personalised cover letter."
        ),
        tags=["cv", "cover-letter", "writing", "job-application"],
        examples=[
            "Write application documents for a Backend Engineer role at Stripe",
        ],
    )

    agent_card = AgentCard(
        name="Writing Agent",
        description=(
            "CrewAI-powered writing agent. Uses a CV Writer and Cover Letter Writer "
            "working in sequence to produce tailored job application documents."
        ),
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supported_interfaces=[
            AgentInterface(url=f"http://{host}:{port}/", protocol_binding="JSONRPC")
        ],
    )

    task_store = InMemoryTaskStore()
    handler = DefaultRequestHandlerV2(
        agent_executor=WritingAgentExecutor(),
        task_store=task_store,
        agent_card=agent_card,
    )

    routes = [
        *create_agent_card_routes(agent_card),
        *create_jsonrpc_routes(handler, rpc_url="/"),
    ]

    app = Starlette(routes=routes)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()