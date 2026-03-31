# Workflow Agent Template

**Use when:** Someone says "I want to automate a multi-step business process with agents."

**Pattern:** Planner → Executor loop with HITL approval gates, typed outputs, and crash-safe checkpointing.

**Runs after:** `pip install -r requirements.txt && python main.py "your task"`

---

## Project Structure

```
my-workflow-agent/
  main.py
  agent/
    __init__.py
    planner.py
    executor.py
    hitl.py
    checkpoint.py
    tools.py
    cost.py
  requirements.txt
```

---

## File: `requirements.txt`

```
anthropic>=0.40.0
pydantic>=2.0
rich>=13.0
```

---

## File: `agent/__init__.py`

```python
# agent package
```

---

## File: `agent/tools.py`

```python
import subprocess
import glob as glob_module
from pathlib import Path
from typing import Any

MODEL = "claude-sonnet-4-6"

# ── Tool schemas for the Anthropic API ──────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "bash",
        "description": "Run a bash command and return stdout+stderr. Avoid destructive commands.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "file_read",
        "description": "Read a file from disk and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "file_write",
        "description": "Write content to a file (creates or overwrites).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "file_edit",
        "description": "Replace an exact string in a file with new text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "e.g. **/*.py"},
                "root": {"type": "string", "description": "Root directory (default: cwd)"},
            },
            "required": ["pattern"],
        },
    },
]


class ToolRegistry:
    """Execute tool calls returned by the model."""

    def run(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        match tool_name:
            case "bash":
                return self._bash(tool_input)
            case "file_read":
                return self._file_read(tool_input)
            case "file_write":
                return self._file_write(tool_input)
            case "file_edit":
                return self._file_edit(tool_input)
            case "glob":
                return self._glob(tool_input)
            case _:
                return f"ERROR: Unknown tool '{tool_name}'"

    def _bash(self, inp: dict) -> str:
        timeout = inp.get("timeout", 30)
        try:
            result = subprocess.run(
                inp["command"], shell=True, capture_output=True,
                text=True, timeout=timeout
            )
            out = result.stdout + result.stderr
            return out[:8000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {timeout}s"
        except Exception as e:
            return f"ERROR: {e}"

    def _file_read(self, inp: dict) -> str:
        try:
            return Path(inp["path"]).read_text(encoding="utf-8")
        except Exception as e:
            return f"ERROR: {e}"

    def _file_write(self, inp: dict) -> str:
        try:
            p = Path(inp["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(inp["content"], encoding="utf-8")
            return f"Written {len(inp['content'])} bytes to {inp['path']}"
        except Exception as e:
            return f"ERROR: {e}"

    def _file_edit(self, inp: dict) -> str:
        try:
            p = Path(inp["path"])
            text = p.read_text(encoding="utf-8")
            if inp["old_string"] not in text:
                return f"ERROR: old_string not found in {inp['path']}"
            p.write_text(text.replace(inp["old_string"], inp["new_string"], 1), encoding="utf-8")
            return f"Edited {inp['path']}"
        except Exception as e:
            return f"ERROR: {e}"

    def _glob(self, inp: dict) -> str:
        root = inp.get("root", ".")
        matches = glob_module.glob(inp["pattern"], root_dir=root, recursive=True)
        return "\n".join(matches) if matches else "(no matches)"
```

---

## File: `agent/cost.py`

```python
# Pricing as of 2025 — update if Anthropic changes rates
INPUT_COST_PER_1M  = 3.00   # claude-sonnet-4-6
OUTPUT_COST_PER_1M = 15.00

class CostTracker:
    def __init__(self):
        self.input_tokens  = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int):
        self.input_tokens  += input_tokens
        self.output_tokens += output_tokens

    @property
    def total_usd(self) -> float:
        return (
            self.input_tokens  / 1_000_000 * INPUT_COST_PER_1M +
            self.output_tokens / 1_000_000 * OUTPUT_COST_PER_1M
        )

    def summary(self) -> str:
        return (
            f"Tokens: {self.input_tokens:,} in / {self.output_tokens:,} out  |  "
            f"Cost: ${self.total_usd:.4f}"
        )
```

---

## File: `agent/checkpoint.py`

```python
import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class WorkflowCheckpoint:
    workflow_id: str
    task: str
    subtasks: list[dict]
    results: dict[str, str]          # subtask_id → result text
    status: str                       # "planning" | "running" | "done" | "failed"
    created_at: str
    updated_at: str
    completed_ids: list[str] = field(default_factory=list)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = ".workflow-checkpoints"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(exist_ok=True)

    # ── Persist ─────────────────────────────────────────────────────────────

    def save(self, cp: WorkflowCheckpoint) -> str:
        cp.updated_at = datetime.now(timezone.utc).isoformat()
        path = self.dir / f"{cp.workflow_id}.json"
        path.write_text(json.dumps(asdict(cp), indent=2), encoding="utf-8")
        return str(path)

    # ── Load ────────────────────────────────────────────────────────────────

    def load(self, workflow_id: str) -> WorkflowCheckpoint | None:
        path = self.dir / f"{workflow_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return WorkflowCheckpoint(**data)

    # ── Resume helpers ───────────────────────────────────────────────────────

    def list_incomplete(self) -> list[WorkflowCheckpoint]:
        results = []
        for f in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                cp = WorkflowCheckpoint(**data)
                if cp.status not in ("done", "failed"):
                    results.append(cp)
            except Exception:
                continue
        return results

    def new(self, task: str) -> WorkflowCheckpoint:
        now = datetime.now(timezone.utc).isoformat()
        return WorkflowCheckpoint(
            workflow_id=str(uuid.uuid4())[:8],
            task=task,
            subtasks=[],
            results={},
            completed_ids=[],
            status="planning",
            created_at=now,
            updated_at=now,
        )

    def mark_done(self, cp: WorkflowCheckpoint):
        cp.status = "done"
        self.save(cp)
        path = self.dir / f"{cp.workflow_id}.json"
        path.unlink(missing_ok=True)
```

---

## File: `agent/hitl.py`

```python
from enum import Enum
from typing import Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
import json

console = Console()


class HITLDecision(Enum):
    APPROVE = "approve"
    EDIT    = "edit"
    REJECT  = "reject"


class TerminalHITL:
    """
    Gate for tool calls that could have side effects.

    AUTO_APPROVE  — read-only or clearly safe
    ALWAYS_ASK    — writes, deletes, or unknown bash commands
    """

    ALWAYS_ASK  = {
        "bash":       ["rm", "delete", "drop", "truncate", "mkfs", ">"],
        "file_write": ["*"],   # always ask on writes
        "file_edit":  ["*"],
    }
    AUTO_APPROVE = ["file_read", "glob"]
    SAFE_BASH    = ["git status", "git log", "ls", "find", "cat", "grep", "echo"]

    # ── Decision logic ───────────────────────────────────────────────────────

    def should_ask(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        if tool_name in self.AUTO_APPROVE:
            return False
        if tool_name == "bash":
            cmd = tool_input.get("command", "")
            if any(safe in cmd for safe in self.SAFE_BASH):
                return False
            return True
        if tool_name in self.ALWAYS_ASK:
            return True
        return False

    # ── Interactive prompt ───────────────────────────────────────────────────

    def ask(self, tool_name: str, tool_input: dict[str, Any]) -> tuple[HITLDecision, dict]:
        pretty = json.dumps(tool_input, indent=2)
        console.print(Panel(
            Syntax(pretty, "json", theme="monokai", line_numbers=False),
            title=f"[bold yellow]HITL Gate — {tool_name}[/bold yellow]",
            border_style="yellow",
        ))

        while True:
            raw = console.input("[bold]Approve? \\[Y/n/edit][/bold] ").strip().lower()

            if raw in ("", "y", "yes"):
                return HITLDecision.APPROVE, tool_input

            if raw in ("n", "no"):
                console.print("[red]Rejected — skipping this tool call.[/red]")
                return HITLDecision.REJECT, tool_input

            if raw in ("e", "edit"):
                console.print("Paste new JSON input (single line), then press Enter:")
                raw_json = console.input("> ").strip()
                try:
                    edited = json.loads(raw_json)
                    console.print("[green]Using edited input.[/green]")
                    return HITLDecision.EDIT, edited
                except json.JSONDecodeError:
                    console.print("[red]Invalid JSON — try again.[/red]")
                    continue

            console.print("Type Y, n, or edit.")
```

---

## File: `agent/planner.py`

```python
import json
from dataclasses import dataclass
import anthropic
from .tools import MODEL
from .cost import CostTracker


@dataclass
class Subtask:
    id: str                   # e.g. "step_1"
    description: str
    depends_on: list[str]     # ids of subtasks that must complete first
    tools_needed: list[str]   # from: bash, file_read, file_write, file_edit, glob
    expected_output: str
    tier: int                 # 1=autonomous, 2=checkpoint (needs approval), 3=pair


PLANNER_SYSTEM = """You are a workflow planner. Break the user's task into an ordered list of subtasks.

Return ONLY a JSON array, no markdown fences. Each item:
{
  "id": "step_N",
  "description": "...",
  "depends_on": ["step_X"],   // empty list if no deps
  "tools_needed": ["bash", "file_read", ...],
  "expected_output": "one sentence describing what this step produces",
  "tier": 1  // 1=fully autonomous, 2=requires human checkpoint, 3=pair programming
}

Tier guidelines:
- Tier 1: read-only tasks, analysis, report generation
- Tier 2: anything that writes files, calls external APIs, or is irreversible
- Tier 3: architecture decisions, security-sensitive changes

Keep it to 3-6 subtasks. Be concrete. Use real tool names."""


class PlannerAgent:
    def __init__(self, cost_tracker: CostTracker | None = None):
        self.client = anthropic.Anthropic()
        self.cost   = cost_tracker or CostTracker()

    async def plan(self, task: str, context: str = "") -> list[Subtask]:
        user_msg = task
        if context:
            user_msg = f"Context:\n{context}\n\nTask:\n{task}"

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        self.cost.add(response.usage.input_tokens, response.usage.output_tokens)

        raw = response.content[0].text.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        return [Subtask(**item) for item in data]
```

---

## File: `agent/executor.py`

```python
import asyncio
from dataclasses import dataclass
import anthropic
from .tools import TOOLS, ToolRegistry, MODEL
from .hitl import TerminalHITL, HITLDecision
from .checkpoint import CheckpointManager
from .planner import Subtask
from .cost import CostTracker
from rich.console import Console

console = Console()


@dataclass
class ExecutorResult:
    subtask_id: str
    success: bool
    output: str
    cost_usd: float
    turns: int
    error: str | None


EXECUTOR_SYSTEM = """You are an expert software agent. Execute the assigned subtask using the tools available.
Be precise and efficient. When done, output a brief summary of what you accomplished.
Do NOT ask for clarification — make reasonable assumptions and proceed."""


class ExecutorAgent:
    MAX_TURNS = 20

    def __init__(
        self,
        hitl: TerminalHITL,
        checkpoint: CheckpointManager,
        cost_tracker: CostTracker | None = None,
    ):
        self.client     = anthropic.Anthropic()
        self.hitl       = hitl
        self.checkpoint = checkpoint
        self.registry   = ToolRegistry()
        self.cost       = cost_tracker or CostTracker()

    async def execute(self, subtask: Subtask, context: str = "") -> ExecutorResult:
        messages: list[dict] = []
        user_content = f"Subtask: {subtask.description}"
        if context:
            user_content = f"Prior context:\n{context}\n\n{user_content}"
        messages.append({"role": "user", "content": user_content})

        turns = 0
        input_tok = output_tok = 0

        for turn in range(self.MAX_TURNS):
            turns += 1
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=EXECUTOR_SYSTEM,
                tools=TOOLS,
                messages=messages,
            )

            input_tok  += response.usage.input_tokens
            output_tok += response.usage.output_tokens

            # Collect assistant message content blocks
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract final text output
                final = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                self.cost.add(input_tok, output_tok)
                return ExecutorResult(
                    subtask_id=subtask.id,
                    success=True,
                    output=final,
                    cost_usd=self.cost.total_usd,
                    turns=turns,
                    error=None,
                )

            if response.stop_reason != "tool_use":
                break

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name  = block.name
                tool_input = block.input

                # HITL gate
                if self.hitl.should_ask(tool_name, tool_input):
                    decision, tool_input = self.hitl.ask(tool_name, tool_input)
                    if decision == HITLDecision.REJECT:
                        tool_result = "Tool call rejected by user."
                    else:
                        tool_result = self.registry.run(tool_name, tool_input)
                else:
                    console.print(f"  [dim][{tool_name}] auto-approved[/dim]")
                    tool_result = self.registry.run(tool_name, tool_input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(tool_result),
                })

            messages.append({"role": "user", "content": tool_results})

        self.cost.add(input_tok, output_tok)
        return ExecutorResult(
            subtask_id=subtask.id,
            success=False,
            output="",
            cost_usd=self.cost.total_usd,
            turns=turns,
            error="Max turns reached without completion",
        )
```

---

## File: `main.py`

```python
import asyncio
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from agent.planner import PlannerAgent, Subtask
from agent.executor import ExecutorAgent
from agent.hitl import TerminalHITL
from agent.checkpoint import CheckpointManager, WorkflowCheckpoint
from agent.cost import CostTracker

console = Console()

TIER_LABELS = {1: "[green]Auto[/green]", 2: "[yellow]Review[/yellow]", 3: "[red]Pair[/red]"}


# ── Plan display ─────────────────────────────────────────────────────────────

def show_plan(subtasks: list[Subtask]):
    table = Table(title=f"Workflow Plan ({len(subtasks)} subtasks)", show_lines=True)
    table.add_column("#",    style="bold cyan", width=4)
    table.add_column("Task", min_width=40)
    table.add_column("Deps", style="dim", width=12)
    table.add_column("Tier", width=10)

    for i, st in enumerate(subtasks, 1):
        deps = ", ".join(st.depends_on) if st.depends_on else "—"
        table.add_row(
            str(i),
            st.description,
            deps,
            TIER_LABELS.get(st.tier, str(st.tier)),
        )
    console.print(table)


# ── Dependency-aware execution ────────────────────────────────────────────────

async def execute_with_deps(
    subtasks: list[Subtask],
    executor: ExecutorAgent,
    cp: WorkflowCheckpoint,
    cm: CheckpointManager,
):
    completed: dict[str, str] = dict(cp.results)   # id → output
    pending = [st for st in subtasks if st.id not in cp.completed_ids]
    total   = len(subtasks)
    done    = len(cp.completed_ids)

    while pending:
        # Find subtasks whose deps are all satisfied
        ready = [
            st for st in pending
            if all(dep in completed for dep in st.depends_on)
        ]
        if not ready:
            console.print("[red]Dependency deadlock — cannot proceed.[/red]")
            break

        # Run ready subtasks in parallel
        context = "\n\n".join(
            f"[{sid}]: {out}" for sid, out in completed.items()
        )

        tasks = [executor.execute(st, context) for st in ready]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for st, result in zip(ready, results):
            done += 1
            if isinstance(result, Exception):
                console.print(f"[red][{done}/{total}] {st.description} — FAILED: {result}[/red]")
                cp.results[st.id]      = f"ERROR: {result}"
                cp.completed_ids.append(st.id)
                completed[st.id] = cp.results[st.id]
            else:
                icon = "[green]✓[/green]" if result.success else "[red]✗[/red]"
                console.print(
                    f"{icon} [{done}/{total}] {st.description}  "
                    f"[dim]({result.turns} turns, ${result.cost_usd:.4f})[/dim]"
                )
                if result.output:
                    console.print(Panel(result.output[:600], border_style="dim"))
                cp.results[st.id]       = result.output
                cp.completed_ids.append(st.id)
                completed[st.id]        = result.output

            cm.save(cp)
            pending.remove(st)

    return completed


# ── Main workflow ─────────────────────────────────────────────────────────────

async def run_workflow(task: str, resume: bool = False):
    cm   = CheckpointManager()
    hitl = TerminalHITL()
    cost = CostTracker()

    cp: WorkflowCheckpoint | None = None

    # Resume path
    if resume:
        incomplete = cm.list_incomplete()
        if not incomplete:
            console.print("[yellow]No incomplete workflows found — starting fresh.[/yellow]")
        else:
            console.print("\nIncomplete workflows:")
            for i, c in enumerate(incomplete, 1):
                console.print(f"  {i}. [{c.workflow_id}] {c.task[:60]}  ({len(c.completed_ids)} steps done)")
            idx = console.input("Resume which? [1] ").strip() or "1"
            cp = incomplete[int(idx) - 1]
            task = cp.task
            console.print(f"\n[green]Resuming workflow {cp.workflow_id}[/green]")

    if cp is None:
        cp = cm.new(task)

    console.rule("[bold blue]Workflow Agent[/bold blue]")
    console.print(Panel(task, title="Task", border_style="blue"))

    # Planning
    if not cp.subtasks:
        planner = PlannerAgent(cost)
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as p:
            p.add_task("Planning workflow...")
            subtasks = await planner.plan(task)

        cp.subtasks = [vars(st) for st in subtasks]
        cp.status   = "running"
        cm.save(cp)
    else:
        from agent.planner import Subtask
        subtasks = [Subtask(**s) for s in cp.subtasks]

    # Show plan
    show_plan(subtasks)

    confirm = console.input("\nProceed? [Y/n] ").strip().lower()
    if confirm in ("n", "no"):
        console.print("[yellow]Aborted.[/yellow]")
        return

    # Execute
    executor = ExecutorAgent(hitl, cm, cost)
    console.rule("Executing")
    await execute_with_deps(subtasks, executor, cp, cm)

    # Summary
    console.rule("[bold green]Done[/bold green]")
    console.print(cost.summary())
    cm.mark_done(cp)


async def main():
    args  = sys.argv[1:]
    resume = "--resume" in args
    args  = [a for a in args if a != "--resume"]
    task  = " ".join(args).strip() or input("What workflow should I run? ").strip()

    if not task:
        console.print("[red]No task provided. Exiting.[/red]")
        sys.exit(1)

    await run_workflow(task, resume=resume)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Usage

```bash
# Install
cd my-workflow-agent
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run a new workflow
python main.py "Review all Python files, find any TODO comments, create a summary report"

# Resume a crashed or interrupted workflow
python main.py --resume
```

---

## Example Terminal Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Workflow Agent ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭──────────────────────────────────────────────────────────────────────╮
│ Task                                                                 │
│ Review all Python files, find any TODO comments, create a report    │
╰──────────────────────────────────────────────────────────────────────╯
Planning...

┌──────────────────────────────── Workflow Plan (4 subtasks) ─────────┐
│ #  │ Task                                  │ Deps │ Tier            │
├────┼───────────────────────────────────────┼──────┼─────────────────┤
│ 1  │ Scan all Python files for TODOs       │ —    │ Auto            │
│ 2  │ Parse and categorize TODOs            │ 1    │ Auto            │
│ 3  │ Create GitHub issues via API          │ 2    │ Review          │
│ 4  │ Generate summary report               │ 2    │ Auto            │
└────┴───────────────────────────────────────┴──────┴─────────────────┘

Proceed? [Y/n] Y

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Executing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [glob] auto-approved
  [bash] auto-approved
✓ [1/4] Scan all Python files for TODOs  (3 turns, $0.0041)
✓ [2/4] Parse and categorize TODOs  (2 turns, $0.0028)
╭──────────────────────────────────────────────────────────────────────╮
│ Found 23 TODOs: 12 bugs, 8 features, 3 refactors                    │
╰──────────────────────────────────────────────────────────────────────╯

[3/4] Create GitHub issues via API  (Tier 2 — HITL gate)
╭─────────────────── HITL Gate — bash ──────────────────────────────╮
│ {                                                                  │
│   "command": "gh issue create --title 'Fix auth bug' ..."         │
│ }                                                                  │
╰────────────────────────────────────────────────────────────────────╯
Approve? [Y/n/edit] Y
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| Planner is a single LLM call | Fast, cheap, predictable. No need for planner to use tools. |
| Executor is a full agent loop | Subtasks are open-ended — the model needs tool use iterations. |
| HITL is synchronous | Keeps the UX simple. Agent pauses, human decides, agent continues. |
| Checkpoints after every subtask | Safe to kill at any time. `--resume` picks up exactly where it left off. |
| `asyncio.gather` for parallel steps | Subtasks with no shared deps run concurrently, cutting wall time. |
| Typed `Subtask` / `ExecutorResult` | Catch planning errors early. No stringly-typed dict passing. |
| Cost tracked per executor run | Shows real cost at the end. Forces awareness of token burn. |

---

## Extending This Template

**Add a new tool:** Add schema to `TOOLS` list in `tools.py`, add handler to `ToolRegistry.run()`.

**Change HITL policy:** Edit `TerminalHITL.ALWAYS_ASK` and `AUTO_APPROVE` in `hitl.py`.

**Persist to database:** Replace `CheckpointManager` with a SQLite or Redis-backed version — the interface stays the same.

**Add streaming output:** Pass `stream=True` to the Anthropic client and iterate over `stream.text_stream`.

**Add email/Slack notifications:** Hook into `execute_with_deps()` after each subtask completes.
