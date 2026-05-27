"""
A2A server wrapping the AutoGen interview agent.  (a2a-sdk v1.0.3)

Input format (plain text from client):
    JOB DESCRIPTION:
    <jd text>

    TAILORED CV:
    <cv text>

Output: one artifact — interview_transcript (markdown)
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

from agents.interview.agent import run_interview


def _parse_input(text: str) -> tuple[str, str]:
    """Split delimited input into (job_description, tailored_cv)."""
    jd, cv = "", ""

    if "JOB DESCRIPTION:" in text and "TAILORED CV:" in text:
        parts = text.split("TAILORED CV:")
        jd_part = parts[0].replace("JOB DESCRIPTION:", "").strip()
        cv = parts[1].strip()
        jd = jd_part
    else:
        jd = text

    return jd, cv


class InterviewAgentExecutor(AgentExecutor):
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

        jd, cv = _parse_input(user_input)
        transcript = await run_interview(jd, cv)

        await updater.add_artifact(
            parts=[Part(text=transcript)],
            name="interview_transcript",
            last_chunk=True,
        )

        await updater.complete()

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=8003)
def main(host: str, port: int):
    skill = AgentSkill(
        id="mock_interview",
        name="Mock Interview",
        description=(
            "Simulates a job interview. MockInterviewer asks tough questions; "
            "CareerCoach provides model answers and tips. Returns a full transcript."
        ),
        tags=["interview", "mock-interview", "career-coaching"],
        examples=[
            "Run a mock interview for a Backend Engineer role at Stripe",
        ],
    )

    agent_card = AgentCard(
        name="Interview Agent",
        description=(
            "AutoGen-powered interview agent. Uses a RoundRobinGroupChat with "
            "MockInterviewer and CareerCoach agents to simulate 4 rounds of "
            "interview questions with model answers."
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
        agent_executor=InterviewAgentExecutor(),
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