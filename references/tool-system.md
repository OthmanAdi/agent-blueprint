# Tool System

Complete tool registry, schemas, and execution pipeline.

## Tool Interface

```python
from pydantic import BaseModel
from typing import Any

class ToolResult(BaseModel):
    success: bool
    content: str = ""
    error: str = ""
    metadata: dict[str, Any] = {}

class Tool:
    name: str
    description: str
    input_schema: dict
    is_read_only: bool = False
    is_concurrency_safe: bool = False

    async def call(self, input: dict, context: dict) -> ToolResult:
        raise NotImplementedError

    def check_permissions(self, input: dict) -> str:
        return "passthrough"

    def validate_input(self, input: dict) -> tuple[bool, str]:
        return True, ""
```

## Tool Registry

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_api_schema(self) -> list[dict]:
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    **tool.input_schema,
                },
            })
        return schemas

    def filter_by_deny_rules(self, deny_rules: list[str]) -> "ToolRegistry":
        filtered = ToolRegistry()
        for name, tool in self._tools.items():
            if not any(matches_rule(name, rule) for rule in deny_rules):
                filtered.register(tool)
        return filtered
```

## Built-in Tool Implementations

### BashTool

```python
import asyncio

class BashTool(Tool):
    name = "bash"
    description = "Execute a shell command and return stdout, stderr, and exit code."
    input_schema = {
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in milliseconds (default 120000)",
            },
            "description": {
                "type": "string",
                "description": "Brief description of what this command does",
            },
        },
        "required": ["command"],
    }
    is_read_only = False
    is_concurrency_safe = False

    def _is_read_only_command(self, command: str) -> bool:
        read_only_prefixes = ["ls", "cat", "grep", "find", "git status",
                              "git diff", "git log", "head", "tail", "wc",
                              "which", "echo", "pwd", "type", "node -e"]
        first_cmd = command.split("&&")[0].split("|")[0].split(";")[0].strip()
        return any(first_cmd.startswith(p) for p in read_only_prefixes)

    def check_permissions(self, input: dict) -> str:
        if self._is_read_only_command(input.get("command", "")):
            self.is_read_only = True
            self.is_concurrency_safe = True
        return "passthrough"

    async def call(self, input: dict, context: dict) -> ToolResult:
        command = input["command"]
        timeout = input.get("timeout", 120000) / 1000

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            output = stdout.decode(errors="replace")
            error = stderr.decode(errors="replace")

            if proc.returncode != 0:
                return ToolResult(
                    success=False,
                    content=output,
                    error=f"Exit code {proc.returncode}: {error}",
                )
            return ToolResult(success=True, content=output + error)
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(success=False, error=f"Command timed out after {timeout}s")
```

### FileReadTool

```python
class FileReadTool(Tool):
    name = "read"
    description = "Read file contents. Supports line ranges via offset and limit."
    input_schema = {
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to read",
            },
            "offset": {
                "type": "number",
                "description": "Line number to start reading from (1-indexed)",
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of lines to read",
            },
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
            offset = input.get("offset", 1) - 1
            limit = input.get("limit", len(lines))
            selected = lines[offset:offset + limit]

            numbered = []
            for i, line in enumerate(selected, start=offset + 1):
                numbered.append(f"{i}: {line}")

            return ToolResult(success=True, content="\n".join(numbered))
        except Exception as e:
            return ToolResult(success=False, error=str(e))
```

### FileEditTool

```python
class FileEditTool(Tool):
    name = "edit"
    description = "Replace exact text in a file. The old_string must match exactly."
    input_schema = {
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to find and replace",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement text",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences (default: false)",
            },
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
        old = input["old_string"]
        new = input["new_string"]

        count = content.count(old)
        if count == 0:
            return ToolResult(success=False, error="old_string not found in file")
        if count > 1 and not input.get("replace_all"):
            return ToolResult(
                success=False,
                error=f"Found {count} matches. Use replace_all=true or provide more context.",
            )

        if input.get("replace_all"):
            new_content = content.replace(old, new)
        else:
            new_content = content.replace(old, new, 1)

        path.write_text(new_content, encoding="utf-8")
        return ToolResult(
            success=True,
            content=f"Replaced {count} occurrence(s) in {path.name}",
        )
```

### FileWriteTool

```python
class FileWriteTool(Tool):
    name = "write"
    description = "Write content to a file. Creates the file if it does not exist."
    input_schema = {
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to write to",
            },
            "content": {
                "type": "string",
                "description": "The content to write",
            },
        },
        "required": ["file_path", "content"],
    }
    is_read_only = False
    is_concurrency_safe = False

    async def call(self, input: dict, context: dict) -> ToolResult:
        path = Path(input["file_path"]).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(input["content"], encoding="utf-8")
        return ToolResult(success=True, content=f"Wrote {len(input['content'])} chars to {path.name}")
```

### GlobTool

```python
from pathlib import Path as PathLib
import glob as globmod

class GlobTool(Tool):
    name = "glob"
    description = "Find files matching a glob pattern."
    input_schema = {
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files against",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: cwd)",
            },
        },
        "required": ["pattern"],
    }
    is_read_only = True
    is_concurrency_safe = True

    async def call(self, input: dict, context: dict) -> ToolResult:
        pattern = input["pattern"]
        base = PathLib(input.get("path", ".")).expanduser().resolve()
        matches = sorted(globmod.glob(str(base / pattern), recursive=True))[:100]
        rel = [str(PathLib(m).relative_to(base)) for m in matches]
        return ToolResult(
            success=True,
            content="\n".join(rel) if rel else "No matches found",
        )
```

### GrepTool

```python
import re

class GrepTool(Tool):
    name = "grep"
    description = "Search file contents using regex patterns."
    input_schema = {
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: cwd)",
            },
            "include": {
                "type": "string",
                "description": "File pattern to include (e.g., '*.py')",
            },
        },
        "required": ["pattern"],
    }
    is_read_only = True
    is_concurrency_safe = True

    async def call(self, input: dict, context: dict) -> ToolResult:
        pattern = re.compile(input["pattern"])
        base = Path(input.get("path", ".")).expanduser().resolve()
        include = input.get("include", "*")
        results = []

        for f in base.rglob(include):
            if f.is_file() and not any(p in str(f) for p in [".git", "node_modules", "__pycache__"]):
                try:
                    for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                        if pattern.search(line):
                            results.append(f"{f.relative_to(base)}:{i}: {line.strip()}")
                            if len(results) >= 200:
                                results.append("... (truncated)")
                                break
                except (PermissionError, OSError):
                    continue
            if len(results) >= 200:
                break

        return ToolResult(
            success=True,
            content="\n".join(results) if results else "No matches found",
        )
```

## Registering All Tools

```python
def create_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(FileReadTool())
    registry.register(FileEditTool())
    registry.register(FileWriteTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    return registry
```
