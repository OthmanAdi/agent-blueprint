# Task Agent Template

Multi-step orchestration agent that spawns sub-agents for parallel work.

## Project Structure

```
my-task-agent/
  main.py                # Entry point
  agent/
    __init__.py
    loop.py              # Agent loop with sub-agent support
    tools.py             # All tools + AgentTool (spawn sub-agents)
    permissions.py       # Permission system
    context.py           # Context management
    prompt.py            # Orchestrator system prompt
    coordinator.py       # Task decomposition + sub-agent management
  requirements.txt
```

## Orchestrator System Prompt

```python
ORCHESTRATOR_PROMPT = """You are a task orchestrator that breaks complex work into parallel sub-tasks.

## Your Process
1. Analyze the user's request
2. Break it into independent sub-tasks
3. Spawn sub-agents for each sub-task using the agent tool
4. Collect and synthesize results
5. Present a unified output

## Sub-Agent Rules
- Each sub-agent gets a focused, self-contained prompt
- Include all context the sub-agent needs (file paths, requirements)
- Never assume a sub-agent can see the parent conversation
- Collect results from all sub-agents before synthesizing

## Agent Tool
Use the "agent" tool to spawn sub-agents:
{
  "prompt": "Specific task description with all needed context",
  "description": "Brief task name"
}"""
```

## AgentTool Implementation

```python
class AgentTool(Tool):
    name = "agent"
    description = "Spawn a sub-agent to complete a specific task."
    input_schema = {
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Complete task description for the sub-agent",
            },
            "description": {
                "type": "string",
                "description": "Brief name for this sub-task",
            },
        },
        "required": ["prompt", "description"],
    }
    is_read_only = False
    is_concurrency_safe = True

    def __init__(self, api_client, tools, system_prompt):
        self.api_client = api_client
        self.tools = tools
        self.system_prompt = system_prompt

    async def call(self, input: dict, context: dict) -> ToolResult:
        messages = [{"role": "user", "content": input["prompt"]}]
        result_text = []

        async for event in agent_loop(
            messages=messages,
            tools=self.tools,
            permissions=PermissionSystem.auto_allow(),
            context_manager=ContextManager(),
            system_prompt=self.system_prompt,
            api_client=self.api_client,
            max_turns=20,
        ):
            if event.type == "streaming_text" and event.text:
                result_text.append(event.text)
            if event.type == "done":
                break

        return ToolResult(
            success=True,
            content="".join(result_text),
            metadata={"description": input["description"]},
        )
```

## Coordinator Pattern

```python
async def coordinate(task: str, tools, api_client) -> str:
    sub_agent_tool = AgentTool(
        api_client=api_client,
        tools=tools,
        system_prompt=["You are a focused sub-agent. Complete your task and return results."],
    )

    tools_with_agent = ToolRegistry()
    for tool in tools.all_tools():
        tools_with_agent.register(tool)
    tools_with_agent.register(sub_agent_tool)

    system_prompt = [
        ORCHESTRATOR_PROMPT,
        "You have access to an 'agent' tool that spawns sub-agents.",
        "Break complex tasks into 2-5 parallel sub-tasks.",
    ]

    messages = []
    async for event in agent_loop(
        messages=messages,
        tools=tools_with_agent,
        permissions=PermissionSystem.auto_allow(),
        context_manager=ContextManager(),
        system_prompt=system_prompt,
        api_client=api_client,
    ):
        if event.type == "streaming_text" and event.text:
            print(event.text, end="", flush=True)
        if event.type == "done":
            break

    return messages
```

## Usage

```bash
python main.py "Research React vs Vue vs Svelte, then write a comparison report with code examples for each"
```

This spawns sub-agents to:
1. Research React best practices
2. Research Vue best practices
3. Research Svelte best practices
4. Synthesize into a comparison report
