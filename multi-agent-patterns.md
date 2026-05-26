# Multi-Agent Patterns

## The core problem every multi-agent framework solves

You have a task too complex for one LLM call. So you break it into pieces and give each piece to an agent. The framework decides **who runs when and in what order**.

That's it. Every pattern is just a different answer to "who runs when."

---

## LangGraph — you draw a flowchart

Think of it like a flowchart you draw yourself. Boxes are nodes (code that runs), arrows are edges (what happens next).

**Sequential** — straight line, no decisions:
```
START → clean_data → analyse → summarise → END
```
Use when the order never changes.

**Supervisor** — one box decides which box runs next:
```
START → supervisor → search → supervisor → read → supervisor → END
```
The supervisor is an LLM call that says "go search" or "go read" or "I'm done." Use when you don't know upfront how many steps you need.

**Reflection** — a loop that keeps going until quality is good enough:
```
START → write → critique → write → critique → write → END
```
Use when you want the agent to improve its own output.

**Map-reduce** — split, work in parallel, combine:
```
START → [search_1, search_2, search_3] → merge → END
```
Use when you have many independent items to process at the same time.

The key thing: **you draw the flowchart in code.** The framework just runs whatever graph you built.

---

## CrewAI — you hire a team and assign jobs

Think of it like a manager giving out tasks to employees.

**Sequential** — employee A does their job, hands result to employee B:
```
CV Writer → Cover Letter Writer
```
Employee B gets employee A's output automatically. Use when jobs depend on each other in a fixed order.

**Hierarchical** — a manager decides who does what:
```
Manager LLM → assigns "write CV" to CV Writer
            → assigns "check tone" to Editor
```
The manager is an LLM that reads the goal and decides which employee to use. Use when you don't know upfront which specialist is needed.

**Parallel** — everyone works at the same time:
```
Researcher + Analyst + Writer → all run together → results merged
```
Use when jobs don't need each other's output — they can all start immediately.

The key thing: **you describe people and their jobs.** CrewAI decides when to run them based on the process you pick.

---

## AutoGen — you put people in a room and let them talk

Think of it like a meeting. Agents are people. You decide how the meeting is structured.

**RoundRobin** — they take turns speaking in a fixed order:
```
Interviewer speaks → Coach speaks → Interviewer speaks → Coach speaks → done
```
Nobody decides who speaks — it just rotates. Use for structured conversations like an interview or a debate.

**Selector** — an LLM decides who speaks next based on what was just said:
```
User asks a question → LLM picks the best expert → that expert answers → LLM picks next → ...
```
Use when the right speaker depends on context.

**Swarm** — agents hand off to each other by calling a tool:
```
Agent A handles it → decides to pass to Agent B → B handles it → passes to C → ...
```
The agents themselves decide who goes next, not a central router. Use when each agent knows its own limits and when to escalate.

The key thing: **you describe the people and the meeting format.** AutoGen drives the conversation.

---

## Why Supervisor, Hierarchical, and Selector look the same (but aren't)

They all have "one thing that decides who goes next." The difference is **what that thing is and what information it uses.**

| | What drives the routing decision |
|---|---|
| LangGraph Supervisor | Your accumulated state dict — everything collected so far |
| CrewAI Hierarchical | The overall goal + available agents — manager assigns upfront |
| AutoGen Selector | Only the most recent conversation messages |

---

## What map-reduce actually means

Imagine researching 5 companies.

**Without it (sequential):**
```
research Google → research Stripe → research Apple → ...
Total time: 5 minutes
```

**With it (parallel):**
```
research Google ─┐
research Stripe  ├─→ all running at the same time → merge results
research Apple  ─┘
Total time: 1 minute
```

Use it when tasks are completely independent — one result doesn't affect another.

---

## What reflection actually is

A loop where one agent improves its output based on a critic's feedback:
```
Writer writes → Critic finds flaws → Writer fixes → Critic checks → Writer fixes → Critic: "good" → STOP
```

Different from the others because the loop exists **purely to improve one piece of output**, not to accomplish different tasks. The critic's only job is to find flaws. The writer's only job is to fix them.

Use it when the first attempt is never good enough — writing, code generation, report quality.

---

## One-line summary

| Framework | What you define | Framework controls |
|---|---|---|
| LangGraph | The flowchart (boxes + arrows) | Running each box in order |
| CrewAI | The people + their job descriptions | When each person does their job |
| AutoGen | The people + the meeting format | Who speaks and when |

---

## Pattern decision guide

| Problem shape | Use |
|---|---|
| Don't know how many steps upfront | LangGraph Supervisor |
| Process many items at the same time | LangGraph Map-reduce |
| Improve one output through feedback | LangGraph Reflection |
| Fixed pipeline where each step depends on the previous | CrewAI Sequential |
| LLM should decide which specialist handles each subtask | CrewAI Hierarchical |
| Independent jobs that can all start at once | CrewAI Parallel |
| Strictly turn-based conversation | AutoGen RoundRobin |
| Right speaker depends on what was just said | AutoGen Selector |
| Agents themselves decide who handles what next | AutoGen Swarm |