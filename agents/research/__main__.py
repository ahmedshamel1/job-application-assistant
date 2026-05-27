"""
A2A server wrapping the LangGraph research agent.  (a2a-sdk v1.0.3)

The A2A protocol works like this:
  Client sends POST /message:send → DefaultRequestHandlerV2 receives it
  → creates a Task → calls executor.execute()
  → executor uses TaskUpdater to post events (working → artifact → completed)
  → handler returns the Task to the client
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

from agents.research.agent import run_research


class ResearchAgentExecutor(AgentExecutor):
    """
    The bridge between A2A protocol and our LangGraph agent.

    A2A calls execute() for every incoming task. We:
    1. Signal working via TaskUpdater.start_work()
    2. Run the LangGraph research graph
    3. Post the report as an artifact
    4. Signal completed
    """

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # A2A v1.0: executor must enqueue a Task object first before any status events.
        # This signals to the framework that this is an async (task-based) interaction.
        task = Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
        )
        await event_queue.enqueue_event(task)

        await updater.start_work()

        report = await run_research(user_input)

        await updater.add_artifact(
            parts=[Part(text=report)],
            name="research_report",
            last_chunk=True,
        )

        await updater.complete()

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=8001)
def main(host: str, port: int):
    skill = AgentSkill(
        id="research_company_role",
        name="Research Company and Role",
        description=(
            "Researches a company and job role using web search. "
            "Returns company overview, role details, salary range, and recent news."
        ),
        tags=["research", "company", "salary", "job"],
        examples=[
            "Research Google for the role of Software Engineer in London",
            "Research Stripe for the role of Backend Engineer",
        ],
    )

    agent_card = AgentCard(
        name="Research Agent",
        description=(
            "LangGraph-powered research agent. Uses a supervisor graph with "
            "web_search and fetch_page MCP tools to produce a structured "
            "research report for any company and role."
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
        agent_executor=ResearchAgentExecutor(),
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
