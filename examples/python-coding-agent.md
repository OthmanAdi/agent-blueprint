# Example: Building a Full Claude Code Clone in Python

A complete walkthrough of building a production-grade coding agent.

## What We're Building

A terminal-based coding assistant that:
- Reads, writes, and edits files
- Executes bash commands
- Searches code with glob and grep
- Manages context for long conversations
- Tracks token costs
- Asks for permission before destructive actions

## Step 1: Project Setup

```bash
mkdir my-claude-clone && cd my-claude-clone
mkdir agent
pip install anthropic pydantic rich
```

## Step 2: Copy Reference Implementations

Copy these files from the skill references:

| Reference File | Copy To |
|---------------|---------|
| `references/agent-loop.md` | `agent/loop.py` — extract the Python code |
| `references/tool-system.md` | `agent/tools.py` — extract all tool classes |
| `references/permission-system.md` | `agent/permissions.py` — extract PermissionSystem |
| `references/context-management.md` | `agent/context.py` — extract ContextManager |
| `references/system-prompts.md` | `agent/prompt.py` — extract SystemPromptBuilder |

## Step 3: Create the API Client Wrapper

```python
# agent/api.py
import anthropic
from typing import AsyncGenerator

class APIClient:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.AsyncAnthropic()
        self.model = model

    async def stream(self, system, messages, tools, stream=True):
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=8192,
            system=system,
            messages=messages,
            tools=tools,
        ) as stream:
            async for event in stream:
                yield event
```

## Step 4: Create the Entry Point

```python
# main.py
import asyncio
import sys
from agent.api import APIClient
from agent.loop import agent_loop
from agent.tools import create_default_tool_registry
from agent.permissions import PermissionSystem
from agent.context import ContextManager
from agent.prompt import build_system_prompt
from agent.cost import CostTracker
from rich.console import Console

console = Console()

async def main():
    api = APIClient()
    tools = create_default_tool_registry()
    permissions = PermissionSystem()
    context = ContextManager(api_client=api)
    cost = CostTracker()
    system_prompt = build_system_prompt(tools)

    # Interactive mode
    console.print("[bold blue]My Claude Clone[/] — type your prompt (Ctrl+C to exit)")
    
    while True:
        try:
            prompt = console.input("[bold green]> [/]")
            if prompt.strip() in ("exit", "quit", "q"):
                break

            messages = [{"role": "user", "content": prompt}]
            
            async for event in agent_loop(
                messages=messages,
                tools=tools,
                permissions=permissions,
                context_manager=context,
                system_prompt=system_prompt,
                api_client=api,
                cost_tracker=cost,
            ):
                if event.type == "streaming_text" and event.text:
                    console.print(event.text, end="")
                elif event.type == "tool_result":
                    icon = "[green]OK[/]" if event.result.success else "[red]FAIL[/]"
                    console.print(f"\n[{event.tool_name}] {icon}")
                elif event.type == "done":
                    console.print()
                elif event.type == "cost":
                    console.print(
                        f"[dim]Tokens: {event.input_tokens + event.output_tokens} "
                        f"Cost: ${event.total_cost_usd:.4f}[/]",
                    )

        except KeyboardInterrupt:
            console.print("\n[bold]Goodbye![/]")
            break

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 5: Run It

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python main.py
```

## What You Get

```
My Claude Clone — type your prompt (Ctrl+C to exit)
> List all Python files and find any TODO comments

[Glob] OK
[Grep] OK

I found 3 Python files with 5 TODO comments:

1. **agent/loop.py:42** - `# TODO: add retry logic for rate limits`
2. **agent/tools.py:128** - `# TODO: support concurrent tool execution`
3. **agent/context.py:67** - `# TODO: implement micro-compact`

Tokens: 4231 Cost: $0.0127
```

## Next Steps

- Add streaming output with Rich live display
- Implement interactive permission prompts
- Add `/compact`, `/cost`, `/help` slash commands
- Add session save/restore
- Add sub-agent support for parallel work
