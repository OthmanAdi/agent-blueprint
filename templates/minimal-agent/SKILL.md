# Minimal Agent Template

A single-purpose agent with 2 tools. ~200 lines of Python.

## Files

```
my-agent/
  agent.py          # Everything in one file
  requirements.txt  # anthropic>=0.39.0
```

## agent.py

```python
#!/usr/bin/env python3
"""Minimal AI agent with bash and file tools."""
import asyncio
import sys
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

SYSTEM_PROMPT = """You are a helpful AI assistant with access to bash and file tools.
Be concise and direct. Use tools to accomplish tasks."""

TOOLS = [
    {
        "name": "bash",
        "description": "Execute a shell command",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file's contents",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
            },
            "required": ["path"],
        },
    },
]

async def execute_tool(name: str, input: dict) -> str:
    if name == "bash":
        proc = await asyncio.create_subprocess_shell(
            input["command"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (stdout + stderr).decode(errors="replace")
    elif name == "read_file":
        try:
            return open(input["path"]).read()
        except Exception as e:
            return f"Error: {e}"
    return f"Unknown tool: {name}"

async def agent(prompt: str):
    messages = [{"role": "user", "content": prompt}]

    while True:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for block in assistant_content:
            if block.type == "text":
                print(block.text)
            elif block.type == "tool_use":
                result = await execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if not tool_results:
            break

        messages.append({"role": "user", "content": tool_results})

if __name__ == "__main__":
    asyncio.run(agent(" ".join(sys.argv[1:]) or input("You: ")))
```

## requirements.txt

```
anthropic>=0.39.0
```

## Usage

```bash
pip install -r requirements.txt
python agent.py "List all Python files in the current directory and count their lines"
```
