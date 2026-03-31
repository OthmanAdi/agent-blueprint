# File-Based Planning for Long-Running Tasks

The pattern that prevents context loss on complex, 100+ step work.
Any session can resume exactly where the last one left off.

---

## 1. The Problem

Context windows are finite. Claude's context fills up after 50–150 operations
depending on verbosity of tool results. When that happens:

- The agent loses track of what it already did
- It re-does completed work
- It forgets decisions and makes inconsistent choices
- The task never actually finishes

The fix is obvious in retrospect: **write everything to files**. Files survive
context resets. Files are queryable. Files are the agent's external memory.

This is the Manus pattern. Three planning files, updated continuously, make any
task resumable from a cold start.

---

## 2. The Three Planning Files

| File | Purpose |
|------|---------|
| `task_plan.md` | Phase breakdown, completion status, what's next |
| `progress_log.md` | Action-by-action log — prevents re-doing work |
| `decisions.md` | Why choices were made — prevents re-deciding |

Create them at the start of every long task. Update them after every meaningful
action. They live in the task's working directory.

---

## 3. Complete LongTaskManager

```python
# long_task_manager.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@dataclass
class Phase:
    id: str
    name: str
    description: str
    steps: list[str] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    status: str = "pending"   # pending | in_progress | complete | skipped


@dataclass
class ActionEntry:
    timestamp: str
    phase_id: str
    action: str
    result: str
    files_changed: list[str] = field(default_factory=list)


@dataclass
class DecisionEntry:
    timestamp: str
    decision: str
    rationale: str
    alternatives_rejected: list[str] = field(default_factory=list)


class LongTaskManager:
    """
    Manages file-based planning for tasks with 100+ steps.

    Usage:
        mgr = LongTaskManager("/tmp/my_task")
        mgr.initialize_plan("Migrate 847 files to new format", phases=[...])

        # in agent loop:
        mgr.log_action("Scanned directory", "Found 847 files", [])
        mgr.mark_step_complete("phase_1", "Scan directory")

        if mgr.should_checkpoint(turn=45, token_count=140_000):
            print(mgr.get_resume_context())
            # hand off to new session
    """

    PLAN_FILE = "task_plan.md"
    LOG_FILE = "progress_log.md"
    DECISIONS_FILE = "decisions.md"
    STATE_FILE = ".task_state.json"   # machine-readable mirror

    def __init__(self, work_dir: str):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._state: dict = self._load_state()

    # ── Init ──────────────────────────────────────────────────────────────────

    def initialize_plan(self, task: str, phases: list[Phase]) -> None:
        """
        Call once at the start of a new task.
        Safe to call again — skips if task_plan.md already exists.
        """
        plan_path = self.work_dir / self.PLAN_FILE
        if plan_path.exists():
            return  # resuming — don't overwrite

        self._state = {
            "task": task,
            "status": "in_progress",
            "started_at": _now(),
            "phases": {p.id: asdict(p) for p in phases},
            "current_phase_id": phases[0].id if phases else None,
        }
        self._write_state()
        self._write_plan_md()

        # create log and decisions files with headers
        (self.work_dir / self.LOG_FILE).write_text(
            f"# Progress Log\n## Task: {task}\n## Started: {_now()}\n\n---\n\n",
            encoding="utf-8",
        )
        (self.work_dir / self.DECISIONS_FILE).write_text(
            f"# Decision Log\n## Task: {task}\n\n---\n\n",
            encoding="utf-8",
        )

    # ── Phase control ─────────────────────────────────────────────────────────

    def mark_phase_complete(self, phase_id: str) -> None:
        if phase_id not in self._state.get("phases", {}):
            raise ValueError(f"Unknown phase: {phase_id}")

        self._state["phases"][phase_id]["status"] = "complete"

        # advance to next pending phase
        phases = list(self._state["phases"].values())
        for i, p in enumerate(phases):
            if p["id"] == phase_id and i + 1 < len(phases):
                self._state["current_phase_id"] = phases[i + 1]["id"]
                self._state["phases"][phases[i + 1]["id"]]["status"] = "in_progress"
                break

        self._write_state()
        self._write_plan_md()
        self.log_action(
            f"Phase complete: {phase_id}",
            self._state["phases"][phase_id]["name"],
            [],
        )

    def mark_step_complete(self, phase_id: str, step: str) -> None:
        phase = self._state["phases"].get(phase_id)
        if not phase:
            raise ValueError(f"Unknown phase: {phase_id}")
        if step not in phase["completed_steps"]:
            phase["completed_steps"].append(step)
        self._write_state()
        self._write_plan_md()

    def get_current_phase(self) -> Optional[Phase]:
        pid = self._state.get("current_phase_id")
        if not pid:
            return None
        raw = self._state["phases"].get(pid)
        if not raw:
            return None
        return Phase(**raw)

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_action(
        self,
        action: str,
        result: str,
        files_changed: list[str],
    ) -> None:
        phase = self.get_current_phase()
        phase_id = phase.id if phase else "unknown"

        entry = ActionEntry(
            timestamp=_now(),
            phase_id=phase_id,
            action=action,
            result=result,
            files_changed=files_changed,
        )

        log_path = self.work_dir / self.LOG_FILE
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"### [{entry.timestamp}] {entry.action}\n")
            f.write(f"**Phase:** {entry.phase_id}\n")
            f.write(f"**Result:** {entry.result}\n")
            if entry.files_changed:
                f.write(f"**Files:** {', '.join(entry.files_changed)}\n")
            f.write("\n")

    def log_decision(
        self,
        decision: str,
        rationale: str,
        alternatives_rejected: Optional[list[str]] = None,
    ) -> None:
        entry = DecisionEntry(
            timestamp=_now(),
            decision=decision,
            rationale=rationale,
            alternatives_rejected=alternatives_rejected or [],
        )

        dec_path = self.work_dir / self.DECISIONS_FILE
        with open(dec_path, "a", encoding="utf-8") as f:
            f.write(f"### [{entry.timestamp}] {entry.decision}\n")
            f.write(f"**Rationale:** {entry.rationale}\n")
            if entry.alternatives_rejected:
                f.write(f"**Rejected alternatives:** {', '.join(entry.alternatives_rejected)}\n")
            f.write("\n")

    # ── Context handoff ───────────────────────────────────────────────────────

    def get_resume_context(self) -> str:
        """
        Returns a compact summary of current state.
        Inject this as a user message at the start of a new session.
        """
        task = self._state.get("task", "Unknown task")
        current = self.get_current_phase()
        completed = [
            p["name"]
            for p in self._state.get("phases", {}).values()
            if p["status"] == "complete"
        ]
        pending = [
            p["name"]
            for p in self._state.get("phases", {}).values()
            if p["status"] == "pending"
        ]

        lines = [
            f"# Task Resume Context",
            f"**Task:** {task}",
            f"**Status:** {self._state.get('status', 'unknown')}",
            f"**Work directory:** {self.work_dir}",
            "",
            f"**Completed phases ({len(completed)}):** {', '.join(completed) or 'none'}",
            f"**Current phase:** {current.name if current else 'none'} ({current.id if current else '—'})",
            f"**Pending phases:** {', '.join(pending) or 'none'}",
        ]

        if current and current.steps:
            done = set(current.completed_steps)
            remaining = [s for s in current.steps if s not in done]
            lines += [
                "",
                f"**Steps done in current phase:** {len(current.completed_steps)}/{len(current.steps)}",
                f"**Next step:** {remaining[0] if remaining else '— all done'}",
            ]

        lines += [
            "",
            f"Read `task_plan.md`, `progress_log.md`, and `decisions.md` in `{self.work_dir}` to fully restore context.",
            "Do NOT re-do completed work.",
        ]
        return "\n".join(lines)

    def should_checkpoint(self, turn: int, token_count: int) -> bool:
        """
        Returns True when the agent should stop, write handoff, and signal
        that a new session is needed.

        Conservative thresholds — adjust for your model's context window.
        """
        TOKEN_LIMIT = 150_000   # claude-sonnet-4-6: ~200K, stop well before
        TURN_LIMIT = 60         # sanity cap

        return token_count >= TOKEN_LIMIT or turn >= TURN_LIMIT

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        state_path = self.work_dir / self.STATE_FILE
        if state_path.exists():
            return json.loads(state_path.read_text(encoding="utf-8"))
        return {}

    def _write_state(self) -> None:
        state_path = self.work_dir / self.STATE_FILE
        state_path.write_text(
            json.dumps(self._state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_plan_md(self) -> None:
        task = self._state.get("task", "Unknown")
        status = self._state.get("status", "unknown")
        current_pid = self._state.get("current_phase_id", "")
        phases = self._state.get("phases", {})

        current_name = phases[current_pid]["name"] if current_pid in phases else "—"

        lines = [
            f"# Task: {task}",
            f"## Status: {status}",
            f"## Current Phase: {current_name}",
            "",
            "---",
            "",
        ]

        for p in phases.values():
            status_icon = {"complete": "COMPLETE ✓", "in_progress": "IN PROGRESS", "pending": "PENDING", "skipped": "SKIPPED"}.get(p["status"], p["status"])
            lines.append(f"### Phase {p['id']}: {p['name']} — {status_icon}")
            if p.get("description"):
                lines.append(f"_{p['description']}_")
            lines.append("")

            done = set(p.get("completed_steps", []))
            for step in p.get("steps", []):
                checkbox = "[x]" if step in done else "[ ]"
                lines.append(f"- {checkbox} {step}")
            lines.append("")

        plan_path = self.work_dir / self.PLAN_FILE
        plan_path.write_text("\n".join(lines), encoding="utf-8")
```

---

## 4. task_plan.md Format

After `initialize_plan()` + several `mark_step_complete()` calls, the file looks like:

```markdown
# Task: Migrate 847 files to new format
## Status: in_progress
## Current Phase: Data Processing

---

### Phase 1: Discovery — COMPLETE ✓
_Scan all source files and identify candidates_

- [x] Scan top-level directory
- [x] Recurse into subdirectories
- [x] Identify 23 duplicates
- [x] Write manifest to files.json

### Phase 2: Data Processing — IN PROGRESS
_Transform each file to new schema_

- [x] Processed 412/847 files
- [x] Wrote output to /tmp/output/
- [ ] Process remaining 435 files
- [ ] Validate output checksums
- [ ] Generate migration report

### Phase 3: Validation — PENDING
_Verify all outputs are correct_

- [ ] Run schema validator on all outputs
- [ ] Compare checksums
- [ ] Flag any failures
```

The agent reads this file first in every session. Completed phases are not
re-entered. The current phase is the only active work area.

---

## 5. Context Check and Handoff

```python
# Inside your agent loop:

async def run_long_task(task: str, work_dir: str) -> None:
    long_task = LongTaskManager(work_dir)
    phases = [
        Phase(id="1", name="Discovery", description="Scan all files",
              steps=["Scan directory", "Identify duplicates", "Write manifest"]),
        Phase(id="2", name="Processing", description="Transform files",
              steps=["Process batch 1", "Process batch 2", "Validate"]),
        Phase(id="3", name="Report", description="Generate final report",
              steps=["Aggregate results", "Write report"]),
    ]
    long_task.initialize_plan(task, phases)

    messages = [{"role": "user", "content": task}]
    turn = 0
    total_tokens = 0

    while True:
        turn += 1

        # --- check before calling API ---
        if long_task.should_checkpoint(turn, total_tokens):
            handoff = long_task.get_resume_context()
            print("\n[CHECKPOINT] Context limit approaching.")
            print(handoff)
            print("\nStart a new session with the resume prompt below.")
            return

        # --- call agent ---
        response = await call_claude(messages)
        total_tokens += response.usage.input_tokens + response.usage.output_tokens

        # --- log what happened ---
        long_task.log_action(
            action=f"Turn {turn}: {messages[-1]['content'][:60]}",
            result=response.content[:120],
            files_changed=[],
        )

        # --- check for completion ---
        if is_task_complete(response):
            long_task.mark_phase_complete(long_task.get_current_phase().id)
            break

        messages.append({"role": "assistant", "content": response.content})
```

---

## 6. Resume Prompt Template

When a checkpoint fires, hand this to the user to paste into a new session:

```
Read task_plan.md, progress_log.md, and decisions.md in [work_dir].

Continue from Phase [N]: [Phase Name].
Do NOT re-do any work marked complete in task_plan.md.

[Optional] Here's what happened since the files were last updated:
- I started processing batch 3 but only completed 12/50 files before the
  session ended. The partial output is in /tmp/output/batch3_partial/.

Resume now.
```

The optional section handles the gap between the last file write and the
session end. Without it, the agent re-processes those 12 files. With it, it
picks up exactly at file 13.

---

## 7. Integration with Agent Loop

Minimal integration — 10 lines added to an existing loop:

```python
# Additions highlighted with  ← comments
from long_task_manager import LongTaskManager, Phase

async def agent_loop_with_planning(
    user_message: str,
    work_dir: str,
    phases: list[Phase],
):
    long_task = LongTaskManager(work_dir)            # ← create manager
    long_task.initialize_plan(user_message, phases)   # ← init (safe if resuming)

    messages = [{"role": "user", "content": long_task.get_resume_context()}]  # ← inject state
    messages.append({"role": "user", "content": user_message})

    turn = 0
    tokens = 0

    async for event in _inner_agent_loop(messages):
        turn += 1
        tokens += event.token_count

        long_task.log_action(                         # ← log every action
            action=event.action_description,
            result=event.result_summary,
            files_changed=event.files_changed,
        )

        if long_task.should_checkpoint(turn, tokens): # ← check limit
            yield CheckpointEvent(long_task.get_resume_context())
            return

        yield event
```

The key principle: `log_action` is called on every event, not just on phase
completion. The log is the source of truth for "what did this agent do."
`should_checkpoint` is checked before every API call, not after, so the agent
never gets caught mid-sentence with a full context window.
