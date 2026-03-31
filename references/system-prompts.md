# System Prompt Construction

Dynamic system prompt assembly with cache optimization.

## Architecture

```
System Prompt = [
    Static Prefix (cached by API, 10x cheaper)
    ─────────────────────────────────────────
    Identity Section         "You are..."
    Tool Instructions         Per-tool rules
    Output Rules             Formatting, verbosity
    Task Guidelines           How to approach work

    Cache Boundary Marker

    Dynamic Suffix (changes per turn, not cached)
    ─────────────────────────────────────────
    Current Date/Time
    Git Context             Branch, status, recent commits
    User Context            CLAUDE.md, project config
    Memory                  Previous session notes
]
```

## Implementation

```python
from datetime import datetime
import subprocess

class SystemPromptBuilder:
    def __init__(self, tools, project_dir: str = "."):
        self.tools = tools
        self.project_dir = project_dir

    def build(self, context: dict | None = None) -> list[str]:
        parts = []

        # --- Static Prefix (cacheable) ---
        parts.append(self._identity_section())
        parts.append(self._tool_instructions())
        parts.append(self._output_rules())
        parts.append(self._task_guidelines())

        # --- Dynamic Suffix ---
        parts.append(self._dynamic_context(context or {}))

        return parts

    def _identity_section(self) -> str:
        return """You are an intelligent coding assistant. You help users with software engineering tasks including writing code, debugging, refactoring, and explaining code.

You have access to tools that let you read files, write files, execute shell commands, search code, and more. Use these tools to accomplish the user's tasks.

IMPORTANT: You should be concise and direct. Minimize output tokens while maintaining helpfulness. Avoid unnecessary preamble or postamble."""

    def _tool_instructions(self) -> str:
        sections = []

        for tool in self.tools.all_tools():
            if tool.name == "bash":
                sections.append(self._bash_instructions())
            elif tool.name == "read":
                sections.append(self._read_instructions())
            elif tool.name == "edit":
                sections.append(self._edit_instructions())
            elif tool.name == "write":
                sections.append(self._write_instructions())
            elif tool.name == "glob":
                sections.append(self._glob_instructions())
            elif tool.name == "grep":
                sections.append(self._grep_instructions())

        return "\n\n".join(sections)

    def _bash_instructions(self) -> str:
        return """## Bash Tool
- Run shell commands to execute tasks
- Always explain what a command does before running it
- Use descriptive `description` parameter for each command
- Prefer reading files with the Read tool over cat/head/tail
- NEVER run destructive commands without user confirmation"""

    def _read_instructions(self) -> str:
        return """## Read Tool
- Read file contents with line numbers
- Use offset/limit to read specific sections instead of entire files
- If a file is large, read the first 100 lines to understand structure
- Avoid re-reading files you just wrote (content is in context)"""

    def _edit_instructions(self) -> str:
        return """## Edit Tool
- Find and replace exact text in files
- The old_string must match EXACTLY (including whitespace/indentation)
- For large changes, prefer multiple small edits over one giant replacement
- Always read a file before editing it"""

    def _write_instructions(self) -> str:
        return """## Write Tool
- Write content to a file, creating it if needed
- Overwrites existing files - use with caution
- Prefer Edit for modifying existing files
- Creates parent directories automatically"""

    def _output_rules(self) -> str:
        return """## Output Rules
- Be concise. Answer in 1-3 sentences when possible
- Use code blocks with language tags
- Include file paths and line numbers when referencing code
- Avoid unnecessary explanations unless asked
- Keep responses under 4 lines unless user asks for detail"""

    def _task_guidelines(self) -> str:
        return """## Task Guidelines
- Read files before modifying them
- Follow existing code conventions in the project
- Run tests after making changes when possible
- Use grep/glob to search before asking questions
- Commit changes only when explicitly asked"""

    def _dynamic_context(self, context: dict) -> str:
        parts = []
        parts.append(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")

        git_info = self._get_git_context()
        if git_info:
            parts.append(git_info)

        user_ctx = context.get("user_context", "")
        if user_ctx:
            parts.append(user_ctx)

        return "\n\n".join(parts)

    def _get_git_context(self) -> str:
        try:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True,
                cwd=self.project_dir
            ).stdout.strip()

            if branch:
                return f"Current git branch: {branch}"
        except Exception:
            pass
        return ""

    def _glob_instructions(self) -> str:
        return "## Glob Tool\n- Find files by pattern\n- Returns paths relative to search directory"

    def _grep_instructions(self) -> str:
        return "## Grep Tool\n- Search file contents using regex\n- Use include parameter to filter by file type"


def build_system_prompt(
    tools,
    project_dir: str = ".",
    context: dict | None = None,
) -> list[str]:
    builder = SystemPromptBuilder(tools, project_dir)
    return builder.build(context)
```

## Cache Optimization

When using the Anthropic API, split the prompt for prompt caching:

```python
def split_for_cache(system_prompt: list[str]) -> tuple[list[dict], list[dict]]:
    """Split system prompt into cached prefix and dynamic suffix."""
    boundary = 3  # First 3 sections are static

    prefix = []
    for part in system_prompt[:boundary]:
        prefix.append({
            "type": "text",
            "text": part,
            "cache_control": {"type": "ephemeral"},
        })

    suffix = []
    for part in system_prompt[boundary:]:
        suffix.append({
            "type": "text",
            "text": part,
        })

    return prefix, suffix
```

This makes the static prefix eligible for caching. Cached tokens cost ~10x less than uncached tokens.

## Cost Impact

| Scenario | Without Caching | With Caching |
|----------|----------------|--------------|
| System prompt per turn | ~10K input tokens | ~10K cached tokens |
| Cost per 1K tokens (Sonnet) | $3.00/MTok | $0.30/MTok |
| 100-turn conversation | ~$3.00 system cost | ~$0.30 system cost |
| Savings | — | **90%** |
