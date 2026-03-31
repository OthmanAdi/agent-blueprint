#!/usr/bin/env python3
"""Scaffold a new agent project from a template."""

import sys
import os
import shutil
from pathlib import Path

TEMPLATES = {
    "minimal": "A single-purpose agent with 2 tools (~200 lines)",
    "coding": "Full coding assistant with 9 tools, permissions, context management",
    "research": "Research agent with web search and report generation",
    "task": "Multi-step orchestration agent with sub-agent spawning",
}

TEMPLATE_FILES = {
    "minimal": {
        "agent.py": '''#!/usr/bin/env python3
"""Minimal AI agent with bash and file tools."""
import asyncio
import sys
from anthropic import AsyncAnthropic

client = AsyncAnthropic()
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise."
TOOLS = [
    {{"name": "bash", "description": "Execute a shell command",
      "input_schema": {{"type": "object", "properties": {{"command": {{"type": "string"}}}}}, "required": ["command"]}}},
    {{"name": "read_file", "description": "Read file contents",
      "input_schema": {{"type": "object", "properties": {{"path": {{"type": "string"}}}}}, "required": ["path"]}}},
]

async def execute_tool(name, input_data):
    if name == "bash":
        proc = await asyncio.create_subprocess_shell(input_data["command"],
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        return (stdout + stderr).decode(errors="replace")
    elif name == "read_file":
        try: return open(input_data["path"]).read()
        except Exception as e: return f"Error: {{e}}"

async def agent(prompt):
    messages = [{{"role": "user", "content": prompt}}]
    while True:
        response = await client.messages.create(model="claude-sonnet-4-6",
            max_tokens=4096, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages)
        messages.append({{"role": "assistant", "content": response.content}})
        tool_results = []
        for block in response.content:
            if block.type == "text": print(block.text)
            elif block.type == "tool_use":
                result = await execute_tool(block.name, block.input)
                tool_results.append({{"type": "tool_result", "tool_use_id": block.id, "content": result}})
        if not tool_results: break
        messages.append({{"role": "user", "content": tool_results}})

if __name__ == "__main__":
    asyncio.run(agent(" ".join(sys.argv[1:]) or input("You: ")))
''',
        "requirements.txt": "anthropic>=0.39.0\n",
    },
}

def scaffold(template: str, name: str, output_dir: str):
    if template not in TEMPLATES:
        print(f"Unknown template: {{template}}")
        print(f"Available: {{', '.join(TEMPLATES.keys())}}")
        return False

    target = Path(output_dir) / name
    target.mkdir(parents=True, exist_ok=True)

    files = TEMPLATE_FILES.get(template, TEMPLATE_FILES["minimal"])
    for filename, content in files.items():
        filepath = target / filename
        filepath.write_text(content, encoding="utf-8")
        print(f"  Created {{filepath}}")

    print(f"\nScaffolded '{{name}}' using '{{template}}' template")
    print(f"Location: {{target}}")
    print(f"\nNext steps:")
    print(f"  cd {{target}}")
    print(f"  pip install -r requirements.txt")
    print(f"  export ANTHROPIC_API_KEY=sk-ant-...")
    print(f"  python agent.py 'Your prompt here'")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python scaffold.py <template> [name] [output-dir]")
        print(f"\nAvailable templates:")
        for name, desc in TEMPLATES.items():
            print(f"  {{name:12}} {{desc}}")
        return

    template = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else f"my-{{template}}-agent"
    output = sys.argv[3] if len(sys.argv) > 3 else "."

    scaffold(template, name, output)

if __name__ == "__main__":
    main()
