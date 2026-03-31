# Permission System

The complete 7-layer permission pipeline with hooks.

## Permission Pipeline

```
Tool Call Incoming
       │
       ▼
┌──────────────────────┐
│ Layer 1: Hard Deny   │ ← Safety-critical: .git/, .claude/, shell configs
│ (always enforced)    │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Layer 2: Tool-Specific│ ← Each tool defines its own checks
│ check_permissions()  │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Layer 3: Allow Rules │ ← User-configured: "Bash(git *)" = allow
│ (pattern matching)   │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Layer 4: Safety      │ ← Sensitive paths bypass allow rules
│ (path checks)        │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Layer 5: Pre-Tool    │ ← User hooks can block/modify/allow
│ Hooks                │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Layer 6: Interactive │ ← Ask user with context about the action
│ Prompt               │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Layer 7: Post-Tool   │ ← After execution, can modify output
│ Hooks                │
└──────────────────────┘
```

## Implementation

```python
from dataclasses import dataclass, field
from fnmatch import fnmatch

@dataclass
class PermissionDecision:
    behavior: str  # "allow" | "deny" | "ask"
    reason: str = ""

SENSITIVE_PATHS = [
    ".git/",
    ".claude/",
    ".ssh/",
    "credentials",
    ".env",
    "id_rsa",
    "id_ed25519",
]

class PermissionSystem:
    def __init__(self):
        self.deny_rules: list[str] = []
        self.allow_rules: list[str] = []
        self.ask_rules: list[str] = []
        self.mode: str = "default"  # default | auto | bypass
        self.hooks: list[Hook] = []

    def check(self, tool_name: str, tool_input: dict) -> PermissionDecision:
        # Layer 1: Hard deny
        if self._matches_rules(tool_name, tool_input, self.deny_rules):
            return PermissionDecision("deny", "Matched deny rule")

        # Layer 2: Sensitive path check (bypass-immune)
        input_str = str(tool_input)
        for sensitive in SENSITIVE_PATHS:
            if sensitive in input_str:
                return PermissionDecision("ask", f"Sensitive path: {sensitive}")

        # Layer 3: Allow rules
        if self._matches_rules(tool_name, tool_input, self.allow_rules):
            return PermissionDecision("allow", "Matched allow rule")

        # Layer 4: Bypass mode
        if self.mode == "bypass":
            return PermissionDecision("allow", "Bypass mode enabled")

        # Layer 5: Pre-tool hooks
        for hook in self.hooks:
            if hook.event == "PreToolUse" and hook.matches(tool_name):
                decision = hook.run(tool_name, tool_input)
                if decision:
                    return decision

        # Layer 6: Ask user
        return PermissionDecision("ask", "User confirmation required")

    def _matches_rules(self, tool_name: str, tool_input: dict, rules: list[str]) -> bool:
        for rule in rules:
            if "(" in rule:
                rule_tool, pattern = rule.split("(", 1)
                pattern = pattern.rstrip(")")
                if rule_tool == tool_name:
                    input_str = str(tool_input)
                    if fnmatch(input_str, pattern):
                        return True
            else:
                if rule == tool_name:
                    return True
        return False

    @classmethod
    def auto_allow(cls) -> "PermissionSystem":
        ps = cls()
        ps.mode = "bypass"
        return ps
```

## Hook System

```python
@dataclass
class Hook:
    event: str          # PreToolUse | PostToolUse | PostToolUseFailure
    pattern: str        # Tool name pattern (e.g., "Bash" or "Bash(git *)")
    command: str        # Shell command to execute
    timeout: int = 30   # Seconds

    def matches(self, tool_name: str) -> bool:
        if self.pattern == "*":
            return True
        return fnmatch(tool_name, self.pattern)

    async def run(self, tool_name: str, tool_input: dict) -> PermissionDecision | None:
        import subprocess
        import json

        payload = json.dumps({
            "tool_name": tool_name,
            "tool_input": tool_input,
        })

        try:
            result = subprocess.run(
                self.command,
                shell=True,
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout)
                    decision = output.get("decision")
                    if decision == "approve":
                        return PermissionDecision("allow", output.get("reason", "Hook approved"))
                    elif decision == "block":
                        return PermissionDecision("deny", output.get("reason", "Hook blocked"))
                except json.JSONDecodeError:
                    pass

            return None
        except subprocess.TimeoutExpired:
            return PermissionDecision("deny", "Hook timed out")
```

## Permission Rule Configuration

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Bash(git status*)",
      "Bash(git diff*)",
      "Bash(git log*)",
      "Bash(ls *)",
      "Bash(cat *)"
    ],
    "deny": [
      "Bash(rm -rf /*)",
      "Bash(:(){:|:&};:)",
      "Bash(mkfs*)"
    ],
    "ask": [
      "Bash",
      "Write",
      "Edit"
    ]
  }
}
```

## Rule Pattern Matching

```python
def matches_rule(tool_name: str, tool_input: dict, rule: str) -> bool:
    """Match a permission rule against a tool call.

    Rule formats:
      "Bash"            → matches all Bash calls
      "Bash(git *)"     → matches Bash calls where input contains "git ..."
      "Read"            → matches all Read calls
      "Read(*.ts)"      → matches Read calls for .ts files
      "Edit"            → matches all Edit calls
      "mcp__server"     → matches all tools from MCP server
    """
    if "(" not in rule:
        return rule == tool_name

    rule_tool, pattern = rule.split("(", 1)
    pattern = pattern.rstrip(")")

    if rule_tool != tool_name:
        return False

    # Extract relevant input field
    input_str = ""
    if tool_name == "Bash":
        input_str = tool_input.get("command", "")
    elif tool_name in ("Read", "Edit", "Write"):
        input_str = tool_input.get("file_path", "")
    elif tool_name == "Glob":
        input_str = tool_input.get("pattern", "")
    elif tool_name == "Grep":
        input_str = tool_input.get("pattern", "")
    else:
        input_str = str(tool_input)

    return fnmatch(input_str, pattern)
```
