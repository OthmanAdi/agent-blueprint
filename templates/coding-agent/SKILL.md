# Coding Agent Template

Full coding assistant with 6 tools, permissions, and context management. Copy-paste and run.

## Project Structure

```
my-coding-agent/
  main.py
  agent/
    __init__.py
    loop.py
    tools.py
    permissions.py
    context.py
    prompt.py
    cost.py
  requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python main.py
```

---

## requirements.txt

```
anthropic>=0.39.0
pydantic>=2.0
rich>=13.0
```

---

## agent/__init__.py

```python
# agent package
```

---

## agent/cost.py

```python
from dataclasses import dataclass, field

# Pricing per million tokens (as of claude-sonnet-4-6)
PRICING = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}


@dataclass
class CostTracker:
    model: str = "claude-sonnet-4-6"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    _history: list[dict] = field(default_factory=list)

    def add(self, usage) -> None:
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        self.input_tokens += inp
        self.output_tokens += out
        self.cache_read_tokens += cache_read
        self._history.append({"input": inp, "output": out, "cache_read": cache_read})

    def total_cost(self) -> float:
        rates = PRICING.get(self.model, {"input": 3.00, "output": 15.00})
        # Cache read tokens cost ~10x less
        billable_input = self.input_tokens - self.cache_read_tokens
        cache_cost = self.cache_read_tokens * (rates["input"] / 10) / 1_000_000
        input_cost = billable_input * rates["input"] / 1_000_000
        output_cost = self.output_tokens * rates["output"] / 1_000_000
        return input_cost + output_cost + cache_cost

    def summary(self) -> str:
        return (
            f"Tokens: {self.input_tokens + self.output_tokens:,} "
            f"(in={self.input_tokens:,} out={self.output_tokens:,}) "
            f"Cost: ${self.total_cost():.4f}"
        )
```

---

## agent/permissions.py

```python
from dataclasses import dataclass, field
from fnmatch import fnmatch

SENSITIVE_PATHS = [".git/", ".ssh/", ".env", "credentials", "id_rsa", "id_ed25519"]


@dataclass
class PermissionDecision:
    behavior: str  # "allow" | "deny" | "ask"
    reason: str = ""


class PermissionSystem:
    def __init__(self, mode: str = "ask"):
        # mode: "ask" (interactive), "auto" (allow all), "strict" (deny destructive)
        self.mode = mode
        self.allow_rules: list[str] = [
            "read", "glob", "grep",
        ]
        self.deny_rules: list[str] = []

    def check(self, tool_name: str, tool_input: dict) -> PermissionDecision:
        # Hard deny list
        if tool_name in self.deny_rules:
            return PermissionDecision("deny", "Matched deny rule")

        # Sensitive path check — always ask regardless of mode
        input_str = str(tool_input)
        for sensitive in SENSITIVE_PATHS:
            if sensitive in input_str:
                return PermissionDecision("ask", f"Sensitive path detected: {sensitive}")

        # Allow rules (read-only tools always pass)
        if tool_name in self.allow_rules:
            return PermissionDecision("allow", "Read-only tool")

        # Auto mode — allow everything not explicitly denied
        if self.mode == "auto":
            return PermissionDecision("allow", "Auto mode")

        # Default: ask for write/execute tools
        return PermissionDecision("ask", f"Confirm: {tool_name}")

    def ask_user(self, tool_name: str, tool_input: dict) -> bool:
        """Prompt the user interactively. Returns True to allow."""
        cmd = tool_input.get("command", tool_input.get("file_path", str(tool_input)))
        try:
            answer = input(f"\n[Permission] Allow {tool_name}({cmd!r})? [y/N] ").strip().lower()
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    @classmethod
    def auto_allow(cls) -> "PermissionSystem":
        return cls(mode="auto")
```

---

## agent/context.py

```python
import os
import hashlib


class ContextManager:
    def __init__(
        self,
        context_window: int = 200_000,
        buffer_tokens: int = 13_000,
        max_inline_size: int = 30_000,
        persist_dir: str = ".agent-results",
        api_client=None,
    ):
        self.context_window = context_window
        self.buffer_tokens = buffer_tokens
        self.max_inline_size = max_inline_size
        self.persist_dir = persist_dir
        self.api_client = api_client

    def apply_budget(self, messages: list[dict]) -> list[dict]:
        """Persist oversized tool results to disk; keep a preview in context."""
        for msg in messages:
            if msg.get("role") != "user":
                continue
            for block in msg.get("content", []):
                if block.get("type") != "tool_result":
                    continue
                content = block.get("content", "")
                if isinstance(content, str) and len(content) > self.max_inline_size:
                    path = self._persist(content)
                    block["content"] = (
                        f"[Result too large ({len(content):,} chars). Saved to: {path}]\n"
                        f"Preview:\n{content[:2000]}..."
                    )
        return messages

    def snip_stale(self, messages: list[dict], recent: int = 5) -> list[dict]:
        """Replace old tool results with a placeholder to free up context."""
        result_count = sum(
            1
            for msg in messages
            if msg.get("role") == "user"
            for block in msg.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        )
        to_snip = max(0, result_count - recent)
        snipped = 0

        for msg in messages:
            if snipped >= to_snip:
                break
            if msg.get("role") != "user":
                continue
            for block in msg.get("content", []):
                if snipped >= to_snip:
                    break
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if len(str(block.get("content", ""))) > 200:
                        block["content"] = "[Old tool result cleared]"
                        snipped += 1
        return messages

    def is_over_limit(self, messages: list[dict], system_prompt: list[str]) -> bool:
        return self._count_tokens(messages, system_prompt) > self.context_window

    def _count_tokens(self, messages: list[dict], system_prompt: list[str]) -> int:
        total = sum(len(p) for p in system_prompt) // 4
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "") or block.get("content", "") or str(block)
                        total += len(text) // 4
        return total

    def _persist(self, content: str) -> str:
        h = hashlib.sha256(content.encode()).hexdigest()[:12]
        os.makedirs(self.persist_dir, exist_ok=True)
        path = os.path.join(self.persist_dir, f"{h}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
```

---

## agent/prompt.py

```python
from datetime import datetime


class SystemPromptBuilder:
    def __init__(self, tools, project_dir: str = "."):
        self.tools = tools
        self.project_dir = project_dir

    def build(self) -> list[str]:
        return [
            self._identity(),
            self._tool_rules(),
            self._output_rules(),
            self._dynamic(),
        ]

    def _identity(self) -> str:
        return (
            "You are an intelligent coding assistant. You help users write, edit, debug, "
            "and understand code.\n\n"
            "Use your tools to accomplish tasks. Read files before editing them. "
            "Run tests after making changes when a test suite exists. "
            "Be concise and direct."
        )

    def _tool_rules(self) -> str:
        return """## Tool Rules
- bash: Always explain what a command does. Prefer non-destructive alternatives.
- read: Use offset/limit for large files instead of reading the whole thing.
- edit: old_string must match exactly (whitespace included). Read the file first.
- write: Overwrites entirely — prefer edit for existing files.
- glob: Use to discover file structure before diving into code.
- grep: Use regex. Pass include='*.py' etc. to narrow scope."""

    def _output_rules(self) -> str:
        return """## Output Rules
- Be concise. 1-3 sentences when possible.
- Use fenced code blocks with language tags.
- Include file paths and line numbers when referencing code.
- Avoid unnecessary preamble."""

    def _dynamic(self) -> str:
        parts = [f"Current date: {datetime.now().strftime('%Y-%m-%d')}"]
        try:
            import subprocess
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=self.project_dir,
            ).stdout.strip()
            if branch:
                parts.append(f"Git branch: {branch}")
        except Exception:
            pass
        return "\n".join(parts)


def build_system_prompt(tools, project_dir: str = ".") -> list[str]:
    return SystemPromptBuilder(tools, project_dir).build()
```

---

## agent/tools.py

```python
import asyncio
import glob as globmod
import re
from pathlib import Path

from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    content: str = ""
    error: str = ""


class Tool:
    name: str = ""
    description: str = ""
    input_schema: dict = {}
    is_read_only: bool = False
    is_concurrency_safe: bool = False

    async def call(self, input: dict, context: dict) -> ToolResult:
        raise NotImplementedError

    def validate_input(self, input: dict) -> tuple[bool, str]:
        return True, ""


# ---------------------------------------------------------------------------
# BashTool
# ---------------------------------------------------------------------------

class BashTool(Tool):
    name = "bash"
    description = "Execute a shell command. Returns stdout and stderr."
    input_schema = {
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "number", "description": "Timeout in seconds (default 120)"},
        },
        "required": ["command"],
    }
    is_read_only = False
    is_concurrency_safe = False

    async def call(self, input: dict, context: dict) -> ToolResult:
        command = input["command"]
        timeout = input.get("timeout", 120)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")
            if proc.returncode != 0:
                return ToolResult(success=False, content=out, error=f"Exit {proc.returncode}: {err}")
            return ToolResult(success=True, content=out + (f"\nSTDERR: {err}" if err.strip() else ""))
        except asyncio.TimeoutError:
            return ToolResult(success=False, error=f"Timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# FileReadTool
# ---------------------------------------------------------------------------

class FileReadTool(Tool):
    name = "read"
    description = "Read file contents with line numbers. Supports offset and limit."
    input_schema = {
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file"},
            "offset": {"type": "number", "description": "Start line (1-indexed)"},
            "limit": {"type": "number", "description": "Max lines to read"},
        },
        "required": ["file_path"],
    }
    is_read_only = True
    is_concurrency_safe = True

    async def call(self, input: dict, context: dict) -> ToolResult:
        path = Path(input["file_path"]).expanduser().resolve()
        if not path.exists():
            return ToolResult(success=False, error=f"File not found: {path}")
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            offset = max(0, input.get("offset", 1) - 1)
            limit = input.get("limit", len(lines))
            selected = lines[offset: offset + limit]
            numbered = [f"{i + offset + 1}: {line}" for i, line in enumerate(selected)]
            return ToolResult(success=True, content="\n".join(numbered))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# FileEditTool
# ---------------------------------------------------------------------------

class FileEditTool(Tool):
    name = "edit"
    description = "Replace exact text in a file. old_string must match exactly (whitespace included)."
    input_schema = {
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"},
        },
        "required": ["file_path", "old_string", "new_string"],
    }
    is_read_only = False
    is_concurrency_safe = False

    def validate_input(self, input: dict) -> tuple[bool, str]:
        if input["old_string"] == input["new_string"]:
            return False, "old_string and new_string are identical"
        path = Path(input["file_path"]).expanduser().resolve()
        if not path.exists():
            return False, f"File not found: {path}"
        return True, ""

    async def call(self, input: dict, context: dict) -> ToolResult:
        path = Path(input["file_path"]).expanduser().resolve()
        content = path.read_text(encoding="utf-8")
        old, new = input["old_string"], input["new_string"]
        count = content.count(old)
        if count == 0:
            return ToolResult(success=False, error="old_string not found in file")
        if count > 1 and not input.get("replace_all"):
            return ToolResult(
                success=False,
                error=f"Found {count} matches. Use replace_all=true or provide more context.",
            )
        new_content = content.replace(old, new) if input.get("replace_all") else content.replace(old, new, 1)
        path.write_text(new_content, encoding="utf-8")
        return ToolResult(success=True, content=f"Replaced {count} occurrence(s) in {path.name}")


# ---------------------------------------------------------------------------
# FileWriteTool
# ---------------------------------------------------------------------------

class FileWriteTool(Tool):
    name = "write"
    description = "Write content to a file. Creates the file (and parent dirs) if needed."
    input_schema = {
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to write to"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["file_path", "content"],
    }
    is_read_only = False
    is_concurrency_safe = False

    async def call(self, input: dict, context: dict) -> ToolResult:
        path = Path(input["file_path"]).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(input["content"], encoding="utf-8")
        return ToolResult(success=True, content=f"Wrote {len(input['content']):,} chars to {path.name}")


# ---------------------------------------------------------------------------
# GlobTool
# ---------------------------------------------------------------------------

class GlobTool(Tool):
    name = "glob"
    description = "Find files matching a glob pattern."
    input_schema = {
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
            "path": {"type": "string", "description": "Base directory (default: cwd)"},
        },
        "required": ["pattern"],
    }
    is_read_only = True
    is_concurrency_safe = True

    async def call(self, input: dict, context: dict) -> ToolResult:
        base = Path(input.get("path", ".")).expanduser().resolve()
        matches = sorted(globmod.glob(str(base / input["pattern"]), recursive=True))[:100]
        rel = [str(Path(m).relative_to(base)) for m in matches]
        return ToolResult(
            success=True,
            content="\n".join(rel) if rel else "No matches found",
        )


# ---------------------------------------------------------------------------
# GrepTool
# ---------------------------------------------------------------------------

class GrepTool(Tool):
    name = "grep"
    description = "Search file contents using regex."
    input_schema = {
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern"},
            "path": {"type": "string", "description": "Directory to search (default: cwd)"},
            "include": {"type": "string", "description": "File glob filter (e.g. '*.py')"},
        },
        "required": ["pattern"],
    }
    is_read_only = True
    is_concurrency_safe = True

    async def call(self, input: dict, context: dict) -> ToolResult:
        try:
            pattern = re.compile(input["pattern"])
        except re.error as e:
            return ToolResult(success=False, error=f"Invalid regex: {e}")

        base = Path(input.get("path", ".")).expanduser().resolve()
        include = input.get("include", "*")
        results: list[str] = []
        skip = {".git", "node_modules", "__pycache__", ".venv"}

        for f in base.rglob(include):
            if not f.is_file() or any(p in f.parts for p in skip):
                continue
            try:
                for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                    if pattern.search(line):
                        results.append(f"{f.relative_to(base)}:{i}: {line.strip()}")
                        if len(results) >= 200:
                            results.append("... (truncated at 200 results)")
                            break
            except (PermissionError, OSError):
                continue
            if len(results) >= 200:
                break

        return ToolResult(
            success=True,
            content="\n".join(results) if results else "No matches found",
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_api_schema(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {"type": "object", **t.input_schema},
            }
            for t in self._tools.values()
        ]


def create_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in [BashTool(), FileReadTool(), FileEditTool(), FileWriteTool(), GlobTool(), GrepTool()]:
        registry.register(tool)
    return registry
```

---

## agent/loop.py

```python
import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator

import anthropic

MODEL = "claude-sonnet-4-6"


# --- Event types ---

@dataclass
class StreamingTextEvent:
    type: str = "streaming_text"
    text: str = ""

@dataclass
class ToolResultEvent:
    type: str = "tool_result"
    tool_name: str = ""
    result: object = None

@dataclass
class CostEvent:
    type: str = "cost"
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0

@dataclass
class DoneEvent:
    type: str = "done"
    turn_count: int = 0

@dataclass
class ErrorEvent:
    type: str = "error"
    message: str = ""


Event = StreamingTextEvent | ToolResultEvent | CostEvent | DoneEvent | ErrorEvent


# --- Main loop ---

async def agent_loop(
    messages: list[dict],
    tools,            # ToolRegistry
    permissions,      # PermissionSystem
    context_manager,  # ContextManager
    system_prompt: list[str],
    api_client,       # anthropic.AsyncAnthropic (passed in or created here)
    cost_tracker=None,
    max_turns: int = 50,
) -> AsyncGenerator[Event, None]:
    client = api_client or anthropic.AsyncAnthropic()
    turn = 0

    while turn < max_turns:
        turn += 1

        # Context management
        messages = context_manager.apply_budget(messages)
        messages = context_manager.snip_stale(messages)

        if context_manager.is_over_limit(messages, system_prompt):
            yield ErrorEvent("Context limit exceeded. Start a new conversation.")
            break

        # Build system blocks for API (plain strings)
        system_blocks = [{"type": "text", "text": p} for p in system_prompt]

        # Streaming call
        text_chunks: list[str] = []
        tool_uses: list[dict] = []
        current_tool: dict = {}
        usage = None

        try:
            async with client.messages.stream(
                model=MODEL,
                max_tokens=8192,
                system=system_blocks,
                messages=messages,
                tools=tools.to_api_schema(),
            ) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        usage = event.message.usage
                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool = {
                                "type": "tool_use",
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "_json": "",
                            }
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            text_chunks.append(event.delta.text)
                            yield StreamingTextEvent(text=event.delta.text)
                        elif event.delta.type == "input_json_delta":
                            current_tool["_json"] = current_tool.get("_json", "") + event.delta.partial_json
                    elif event.type == "content_block_stop":
                        if current_tool.get("id"):
                            current_tool["input"] = __import__("json").loads(current_tool.pop("_json") or "{}")
                            tool_uses.append(current_tool)
                            current_tool = {}
                    elif event.type == "message_delta":
                        if getattr(event, "usage", None):
                            usage = event.usage

        except anthropic.BadRequestError as e:
            yield ErrorEvent(str(e))
            break

        # Track cost
        if cost_tracker and usage:
            cost_tracker.add(usage)
            yield CostEvent(
                input_tokens=getattr(usage, "input_tokens", 0),
                output_tokens=getattr(usage, "output_tokens", 0),
                total_cost_usd=cost_tracker.total_cost(),
            )

        # Build assistant message content
        content: list[dict] = []
        if text_chunks:
            content.append({"type": "text", "text": "".join(text_chunks)})
        for tu in tool_uses:
            content.append({k: v for k, v in tu.items() if not k.startswith("_")})

        messages.append({"role": "assistant", "content": content})

        if not tool_uses:
            yield DoneEvent(turn_count=turn)
            break

        # Execute tools
        tool_results: list[dict] = []
        for tu in tool_uses:
            tool = tools.get(tu["name"])
            if not tool:
                result_content = f"Unknown tool: {tu['name']}"
                success = False
            else:
                decision = permissions.check(tu["name"], tu["input"])
                if decision.behavior == "deny":
                    result_content = "Permission denied"
                    success = False
                elif decision.behavior == "ask":
                    allowed = permissions.ask_user(tu["name"], tu["input"])
                    if not allowed:
                        result_content = "User denied permission"
                        success = False
                    else:
                        res = await tool.call(tu["input"], {})
                        result_content = res.content if res.success else f"Error: {res.error}"
                        success = res.success
                else:
                    res = await tool.call(tu["input"], {})
                    result_content = res.content if res.success else f"Error: {res.error}"
                    success = res.success

            from agent.tools import ToolResult
            yield ToolResultEvent(
                tool_name=tu["name"],
                result=ToolResult(success=success, content=result_content if success else "", error="" if success else result_content),
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result_content,
                "is_error": not success,
            })

        messages.append({"role": "user", "content": tool_results})

    else:
        yield ErrorEvent(f"Reached max turns ({max_turns}). Stopping.")
```

---

## main.py

```python
#!/usr/bin/env python3
"""
Interactive coding agent. Uses claude-sonnet-4-6 with 6 tools.

Usage:
    python main.py                  # interactive REPL
    python main.py "fix the bug"    # single prompt then exit
"""
import asyncio
import sys

import anthropic
from rich.console import Console

from agent.loop import agent_loop
from agent.tools import create_default_tool_registry
from agent.permissions import PermissionSystem
from agent.context import ContextManager
from agent.prompt import build_system_prompt
from agent.cost import CostTracker

console = Console()


async def run_prompt(prompt: str, client: anthropic.AsyncAnthropic) -> None:
    tools = create_default_tool_registry()
    permissions = PermissionSystem(mode="ask")
    context = ContextManager()
    cost = CostTracker()
    system_prompt = build_system_prompt(tools)

    messages = [{"role": "user", "content": prompt}]

    async for event in agent_loop(
        messages=messages,
        tools=tools,
        permissions=permissions,
        context_manager=context,
        system_prompt=system_prompt,
        api_client=client,
        cost_tracker=cost,
    ):
        if event.type == "streaming_text" and event.text:
            console.print(event.text, end="")
        elif event.type == "tool_result":
            icon = "[green]OK[/]" if event.result.success else "[red]FAIL[/]"
            console.print(f"\n[[dim]{event.tool_name}[/]] {icon}")
        elif event.type == "done":
            console.print()  # newline after final text
        elif event.type == "cost":
            console.print(
                f"[dim]{cost.summary()}[/]",
                file=sys.stderr,
            )
        elif event.type == "error":
            console.print(f"\n[bold red]Error:[/] {event.message}")


async def main() -> None:
    client = anthropic.AsyncAnthropic()

    # Single-shot mode: python main.py "do something"
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        await run_prompt(prompt, client)
        return

    # Interactive REPL
    console.print("[bold blue]Coding Agent[/] — [dim]claude-sonnet-4-6 | Ctrl+C to exit[/]")
    console.print("[dim]Commands: exit, quit[/]\n")

    while True:
        try:
            prompt = console.input("[bold green]> [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold]Goodbye![/]")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            console.print("[bold]Goodbye![/]")
            break

        await run_prompt(prompt, client)
        console.print()  # blank line between turns


if __name__ == "__main__":
    asyncio.run(main())
```

---

## What You Get After Running

```
Coding Agent — claude-sonnet-4-6 | Ctrl+C to exit
Commands: exit, quit

> List all Python files and find any TODO comments

[glob] OK
[grep] OK

Found 3 Python files with 5 TODO comments:

1. agent/loop.py:42 — TODO: add retry logic for rate limits
2. agent/tools.py:128 — TODO: support concurrent tool execution
3. agent/context.py:67 — TODO: implement micro-compact

Tokens: 4,231 (in=3,890 out=341) Cost: $0.0168
```

## Customization

- **Add a tool:** Subclass `Tool` in `tools.py`, register it in `create_default_tool_registry()`
- **Change model:** Update `MODEL` in `agent/loop.py` and the pricing entry in `agent/cost.py`
- **Auto-allow tools:** Pass `permissions=PermissionSystem(mode="auto")` in `main.py`
- **Adjust context window:** `ContextManager(context_window=100_000)` for smaller models
- **Change system prompt:** Edit `agent/prompt.py` — the `_identity()` method sets the persona
