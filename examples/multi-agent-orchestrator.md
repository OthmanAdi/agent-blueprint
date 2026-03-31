# Example: Multi-Agent Orchestrator

An agent that spawns sub-agents for parallel research and synthesis.

## Pattern

```
User Request
     │
     ▼
Orchestrator Agent (plans and delegates)
     │
     ├──→ Sub-Agent 1 (research React)
     ├──→ Sub-Agent 2 (research Vue)     ← Parallel execution
     └──→ Sub-Agent 3 (research Svelte)
     │
     ▼
Synthesize into unified report
```

## Implementation

```python
import asyncio
from agent.loop import agent_loop, spawn_sub_agent
from agent.tools import create_default_tool_registry
from agent.permissions import PermissionSystem
from agent.context import ContextManager
from agent.prompt import build_system_prompt
from agent.api import APIClient

async def orchestrate(task: str):
    api = APIClient()
    tools = create_default_tool_registry()

    # Step 1: Decompose the task
    decomposition_prompt = f"""Break this task into 3 independent sub-tasks.
    Return ONLY a JSON array of objects with "task" and "focus" fields.

    Task: {task}"""

    decomposition = await spawn_sub_agent(
        prompt=decomposition_prompt,
        tools=tools,
        api_client=api,
    )

    import json
    subtasks = json.loads(decomposition)

    # Step 2: Execute sub-tasks in parallel
    results = await asyncio.gather(*[
        spawn_sub_agent(
            prompt=f"Research and analyze: {st['task']}\nFocus on: {st['focus']}",
            tools=tools,
            api_client=api,
        )
        for st in subtasks
    ])

    # Step 3: Synthesize
    synthesis_prompt = f"""Synthesize these research results into a unified report.

    Original task: {task}

    Sub-task results:
    {chr(10).join(f'## {st["task"]}{chr(10)}{r}' for st, r in zip(subtasks, results))}

    Write a comprehensive report combining all findings."""

    report = await spawn_sub_agent(
        prompt=synthesis_prompt,
        tools=tools,
        api_client=api,
        system_prompt=["You are a report writer. Produce clear, structured reports."],
    )

    return report

if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:])
    report = asyncio.run(orchestrate(task))
    print(report)
```

## When to Use This Pattern

- Research tasks with multiple independent angles
- Code reviews across multiple files/modules
- Documentation generation for multi-component systems
- Any task that benefits from parallel execution

## Cost Considerations

Sub-agents each consume tokens independently. For 3 sub-agents with 20 turns each:
- Total turns: ~60
- Estimated cost: ~$0.30-1.00 (Sonnet)
- Time savings: ~3x faster than sequential
