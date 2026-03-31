# Human-in-the-Loop (HITL)

In 2026, HITL is a governance requirement for enterprise agents — not an optional feature.
Compliance, insurance, and liability frameworks all demand human approval gates for
high-impact actions.

## The Interrupt/Resume Pattern

The correct pattern is **interrupt → persist → notify → wait → resume**. The wrong pattern is
stopping the agent and restarting from scratch. Restarting loses context, wastes LLM calls,
and breaks auditability.

```
Agent executes
      │
      ▼
 Checkpoint reached? ──No──► Continue execution
      │
     Yes
      │
      ▼
 Persist full state to checkpoint store
      │
      ▼
 Notify human (terminal / webhook / DB)
      │
      ▼
 Wait for decision (approve / edit / reject / escalate)
      │
      ▼
 Load checkpoint, apply decision, resume from exact point
```

The agent must persist state **before** notifying. The system could restart while waiting.

---

## The 4-Dimension Decision Matrix

Four dimensions determine if an action needs human approval.

```python
from dataclasses import dataclass

@dataclass
class AgentAction:
    name: str
    tool_input: dict
    is_irreversible: bool        # Can this be undone?
    affected_records: int        # How many records / people does this touch?
    creates_legal_obligation: bool  # Does this create a legal obligation?
    model_confidence: float      # 0.0 – 1.0, how certain is the model?

def requires_human_approval(action: AgentAction) -> bool:
    """Return True if this action should pause for human review."""
    # Dimension 1 — Irreversibility
    if action.is_irreversible:
        return True
    # Dimension 2 — Blast radius
    if action.affected_records > 100:
        return True
    # Dimension 3 — Compliance exposure
    if action.creates_legal_obligation:
        return True
    # Dimension 4 — Model confidence
    if action.model_confidence < 0.85:
        return True
    return False
```

---

## State Persistence

The agent persists its complete conversation state before handing off to a human.
This enables resume-from-checkpoint even after a server restart.

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import uuid


@dataclass
class CheckpointState:
    session_id: str
    turn: int
    messages: list[dict]
    pending_action: dict      # the tool call awaiting approval
    context: dict             # any extra agent state (memory, scratch pad, etc.)
    created_at: str
    timeout_at: str           # auto-expire after N hours


class CheckpointStore:
    """
    Two-layer storage:
      - Redis  → fast lookup during active sessions
      - PostgreSQL / SQLite → durable record for audit trail

    For simple deployments, disk-backed JSON is enough.
    """

    def __init__(self, path: str = "/tmp/hitl_checkpoints"):
        import os
        self.path = path
        os.makedirs(path, exist_ok=True)

    def save(self, state: CheckpointState) -> str:
        checkpoint_id = str(uuid.uuid4())
        file_path = f"{self.path}/{checkpoint_id}.json"
        with open(file_path, "w") as f:
            json.dump(state.__dict__, f, indent=2)
        return checkpoint_id

    def load(self, checkpoint_id: str) -> CheckpointState:
        file_path = f"{self.path}/{checkpoint_id}.json"
        with open(file_path) as f:
            data = json.load(f)
        return CheckpointState(**data)

    def list_pending(self) -> list[CheckpointState]:
        import os
        states = []
        for fname in os.listdir(self.path):
            if fname.endswith(".json"):
                cid = fname.replace(".json", "")
                states.append(self.load(cid))
        return states

    def delete(self, checkpoint_id: str) -> None:
        import os
        file_path = f"{self.path}/{checkpoint_id}.json"
        if os.path.exists(file_path):
            os.remove(file_path)

    def write_decision(self, checkpoint_id: str, decision: dict) -> None:
        file_path = f"{self.path}/{checkpoint_id}.decision.json"
        with open(file_path, "w") as f:
            json.dump(decision, f)

    def read_decision(self, checkpoint_id: str) -> dict | None:
        import os
        file_path = f"{self.path}/{checkpoint_id}.decision.json"
        if not os.path.exists(file_path):
            return None
        with open(file_path) as f:
            return json.load(f)
```

---

## The Four Approval Outcomes

```python
from enum import Enum
from dataclasses import dataclass


class ApprovalDecision(Enum):
    APPROVE   = "approve"    # Execute exactly as planned
    EDIT      = "edit"       # Human modified the action — execute modified version
    REJECT    = "reject"     # Cancel this action, return error to agent
    ESCALATE  = "escalate"   # Send to a higher authority (manager, security team)


@dataclass
class ApprovalResult:
    outcome: ApprovalDecision
    modified_input: dict | None = None   # set when outcome == EDIT
    reason: str = ""
```

How each outcome flows back into the agent loop:

- **APPROVE** — pass `tool_input` to the tool unchanged, inject the tool result as normal.
- **EDIT** — replace `tool_input` with `decision.modified_input`, then execute.
- **REJECT** — inject a synthetic tool result: `{"error": "Action rejected by human: <reason>"}`.
  The agent receives this and must decide what to do next (retry differently, abort, ask user).
- **ESCALATE** — same as REJECT for the current agent turn, but the checkpoint is forwarded
  to a secondary approval queue (e.g., a Slack channel for the security team).

---

## Python Implementation — HITLMiddleware

```python
import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Callable


class HITLMiddleware:
    """
    Wraps any tool call with an interrupt/resume approval gate.

    Usage:
        hitl = HITLMiddleware(store=CheckpointStore(), notification=TerminalNotification())
        # In your agent loop — before every tool execution:
        if hitl.requires_approval(tool_name, tool_input):
            checkpoint_id = hitl.create_checkpoint(messages, tool_use)
            hitl.notify(checkpoint_id, f"Waiting for approval: {tool_name}")
            decision = await hitl.wait_for_approval(checkpoint_id, timeout=3600)
            ...
    """

    def __init__(
        self,
        store: CheckpointStore,
        notification: "NotificationBackend",
        rules: dict | None = None,
    ):
        self.store = store
        self.notification = notification
        self.rules = rules or {
            "auto_approve": ["Read", "Glob", "Grep", "Bash(git status*)", "Bash(git log*)"],
            "always_ask":   ["Bash(git push*)", "Bash(rm *)", "Write"],
            "auto_reject":  ["Bash(rm -rf /*)"],
        }

    def requires_approval(self, tool_name: str, tool_input: dict) -> bool:
        """True if this tool call must pause for human review."""
        if self._matches(tool_name, tool_input, self.rules["auto_reject"]):
            return True  # auto-reject also pauses to tell the human
        if self._matches(tool_name, tool_input, self.rules["auto_approve"]):
            return False
        if self._matches(tool_name, tool_input, self.rules["always_ask"]):
            return True
        # Fall back to 4-dimension matrix
        action = self._infer_action(tool_name, tool_input)
        return requires_human_approval(action)

    def create_checkpoint(
        self,
        messages: list[dict],
        pending_action: dict,
        context: dict | None = None,
        timeout_hours: int = 8,
    ) -> str:
        now = datetime.utcnow()
        state = CheckpointState(
            session_id=str(__import__("uuid").uuid4()),
            turn=len(messages),
            messages=messages,
            pending_action=pending_action,
            context=context or {},
            created_at=now.isoformat(),
            timeout_at=(now + timedelta(hours=timeout_hours)).isoformat(),
        )
        return self.store.save(state)

    def notify(self, checkpoint_id: str, message: str) -> None:
        state = self.store.load(checkpoint_id)
        asyncio.get_event_loop().run_until_complete(
            self.notification.notify(checkpoint_id, state.pending_action)
        )

    async def wait_for_approval(
        self, checkpoint_id: str, timeout: int = 3600, poll_interval: int = 2
    ) -> ApprovalResult:
        """Poll the checkpoint store until a decision arrives or timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self.store.read_decision(checkpoint_id)
            if raw:
                self.store.delete(checkpoint_id)
                return ApprovalResult(
                    outcome=ApprovalDecision(raw["outcome"]),
                    modified_input=raw.get("modified_input"),
                    reason=raw.get("reason", ""),
                )
            await asyncio.sleep(poll_interval)
        # Timeout — treat as rejection
        self.store.delete(checkpoint_id)
        return ApprovalResult(outcome=ApprovalDecision.REJECT, reason="Approval timed out")

    def _matches(self, tool_name: str, tool_input: dict, rules: list[str]) -> bool:
        for rule in rules:
            if "(" in rule:
                rule_tool, pattern = rule.split("(", 1)
                pattern = pattern.rstrip(")")
                if rule_tool == tool_name:
                    cmd = tool_input.get("command", str(tool_input))
                    if fnmatch(cmd, pattern):
                        return True
            elif rule == tool_name:
                return True
        return False

    def _infer_action(self, tool_name: str, tool_input: dict) -> AgentAction:
        irreversible = tool_name in ("Write", "Edit", "Bash")
        return AgentAction(
            name=tool_name,
            tool_input=tool_input,
            is_irreversible=irreversible,
            affected_records=0,
            creates_legal_obligation=False,
            model_confidence=0.90,
        )
```

### Agent Loop Integration

```python
async def run_agent_loop(messages: list[dict], hitl: HITLMiddleware) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic()

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=messages,
            tools=TOOLS,
        )

        if response.stop_reason == "end_turn":
            break

        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name  = block.name
            tool_input = block.input

            # HITL gate
            if hitl.requires_approval(tool_name, tool_input):
                checkpoint_id = hitl.create_checkpoint(messages, {"name": tool_name, "input": tool_input})
                hitl.notify(checkpoint_id, f"Waiting for approval: {tool_name}")
                decision = await hitl.wait_for_approval(checkpoint_id, timeout=3600)

                if decision.outcome == ApprovalDecision.REJECT:
                    tool_result = {"error": f"Action rejected: {decision.reason}"}
                elif decision.outcome == ApprovalDecision.EDIT:
                    tool_input  = decision.modified_input
                    tool_result = execute_tool(tool_name, tool_input)
                else:
                    tool_result = execute_tool(tool_name, tool_input)
            else:
                tool_result = execute_tool(tool_name, tool_input)

            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "content": str(tool_result)}]})

    return messages
```

---

## TypeScript Equivalent

```typescript
import Anthropic from "@anthropic-ai/sdk";

interface ApprovalResult {
  outcome: "approve" | "edit" | "reject" | "escalate";
  modifiedInput?: Record<string, unknown>;
  reason?: string;
}

class HITLMiddleware {
  private store: Map<string, unknown> = new Map();
  private decisions: Map<string, ApprovalResult> = new Map();

  requiresApproval(toolName: string, _toolInput: Record<string, unknown>): boolean {
    const alwaysAsk = ["Write", "Edit"];
    return alwaysAsk.includes(toolName);
  }

  createCheckpoint(messages: unknown[], pendingAction: unknown): string {
    const id = crypto.randomUUID();
    this.store.set(id, { messages, pendingAction, createdAt: new Date().toISOString() });
    return id;
  }

  notify(checkpointId: string, message: string): void {
    console.log(`\n[HITL] ${message}`);
    console.log(`[HITL] Checkpoint: ${checkpointId}`);
    console.log(`[HITL] Approve with: hitl.submitDecision("${checkpointId}", "approve")`);
  }

  submitDecision(checkpointId: string, result: ApprovalResult): void {
    this.decisions.set(checkpointId, result);
  }

  async waitForApproval(checkpointId: string, timeoutMs = 3_600_000): Promise<ApprovalResult> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const decision = this.decisions.get(checkpointId);
      if (decision) {
        this.store.delete(checkpointId);
        this.decisions.delete(checkpointId);
        return decision;
      }
      await new Promise((r) => setTimeout(r, 2000));
    }
    return { outcome: "reject", reason: "Approval timed out" };
  }
}
```

---

## Notification Patterns

```python
from typing import Protocol


class NotificationBackend(Protocol):
    async def notify(self, checkpoint_id: str, action: dict) -> None: ...
    async def poll_for_decision(self, checkpoint_id: str) -> ApprovalResult: ...


class TerminalNotification:
    """Prints to stdout. Human types their decision in the terminal."""

    async def notify(self, checkpoint_id: str, action: dict) -> None:
        print(f"\n{'='*60}")
        print(f"[HITL] Action requires approval")
        print(f"  checkpoint : {checkpoint_id}")
        print(f"  tool       : {action.get('name')}")
        print(f"  input      : {action.get('input')}")
        print(f"  decide     : approve | edit | reject | escalate")
        print(f"{'='*60}\n")

    async def poll_for_decision(self, checkpoint_id: str) -> ApprovalResult:
        raw = input("Decision [approve/edit/reject/escalate]: ").strip().lower()
        modified = None
        if raw == "edit":
            modified = eval(input("Modified input (dict): "))
        return ApprovalResult(outcome=ApprovalDecision(raw), modified_input=modified)


class WebhookNotification:
    """POSTs to a URL — works for Slack, Teams, email relay, custom UI."""

    def __init__(self, webhook_url: str, poll_base_url: str):
        self.webhook_url  = webhook_url
        self.poll_base_url = poll_base_url

    async def notify(self, checkpoint_id: str, action: dict) -> None:
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json={
                "checkpoint_id": checkpoint_id,
                "action":        action,
                "approve_url":   f"{self.poll_base_url}/approve/{checkpoint_id}",
                "reject_url":    f"{self.poll_base_url}/reject/{checkpoint_id}",
            })

    async def poll_for_decision(self, checkpoint_id: str) -> ApprovalResult:
        import httpx
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{self.poll_base_url}/decision/{checkpoint_id}")
            data = r.json()
            return ApprovalResult(
                outcome=ApprovalDecision(data["outcome"]),
                modified_input=data.get("modified_input"),
                reason=data.get("reason", ""),
            )


class DatabaseNotification:
    """Writes to a DB table. Your app polls and renders a UI for the approver."""

    def __init__(self, connection_string: str):
        self.dsn = connection_string

    async def notify(self, checkpoint_id: str, action: dict) -> None:
        import asyncpg, json
        conn = await asyncpg.connect(self.dsn)
        await conn.execute(
            "INSERT INTO hitl_pending (id, action, status) VALUES ($1, $2, 'pending')",
            checkpoint_id, json.dumps(action)
        )
        await conn.close()

    async def poll_for_decision(self, checkpoint_id: str) -> ApprovalResult:
        import asyncpg
        conn = await asyncpg.connect(self.dsn)
        while True:
            row = await conn.fetchrow(
                "SELECT status, modified_input, reason FROM hitl_pending WHERE id = $1",
                checkpoint_id,
            )
            if row and row["status"] != "pending":
                await conn.close()
                return ApprovalResult(
                    outcome=ApprovalDecision(row["status"]),
                    modified_input=row["modified_input"],
                    reason=row["reason"] or "",
                )
            await asyncio.sleep(2)
```

---

## Pre-Configured Approval Rules

Users define rules once. The agent never interrupts for routine, low-risk calls.

```json
{
  "hitl": {
    "auto_approve": [
      "Read",
      "Glob",
      "Grep",
      "Bash(git status*)",
      "Bash(git log*)",
      "Bash(git diff*)"
    ],
    "always_ask": [
      "Bash(git push*)",
      "Bash(rm *)",
      "Write",
      "Edit"
    ],
    "auto_reject": [
      "Bash(rm -rf /*)"
    ]
  }
}
```

Rule matching order: `auto_reject` → `always_ask` → `auto_approve` → 4-dimension matrix.

---

## The Tiered Delegation Model

Three tiers, chosen per task. Matches the Kilo Speed framework.

| Tier | Name | Interrupt Strategy | When to Use |
|------|------|--------------------|-------------|
| 1 | Autonomous | Never | Read-only analysis, safe codegen, research |
| 2 | Checkpoints | At defined milestones | Multi-step builds, data migrations |
| 3 | Pair | Every significant decision | Architecture changes, security-critical ops |

```python
from enum import Enum
from dataclasses import dataclass, field


class AgentTier(Enum):
    AUTONOMOUS   = "tier1"
    CHECKPOINTS  = "tier2"
    PAIR         = "tier3"


@dataclass
class HITLConfig:
    mode: str                    # auto_allow | milestone | ask_always
    milestones: list[str] = field(default_factory=list)


def build_hitl_config(tier: AgentTier) -> HITLConfig:
    return {
        AgentTier.AUTONOMOUS:  HITLConfig(mode="auto_allow"),
        AgentTier.CHECKPOINTS: HITLConfig(
            mode="milestone",
            milestones=["planning_done", "before_deploy"],
        ),
        AgentTier.PAIR: HITLConfig(mode="ask_always"),
    }[tier]


async def run_agent(task: str, tier: AgentTier = AgentTier.AUTONOMOUS) -> None:
    config = build_hitl_config(tier)
    store  = CheckpointStore()
    notif  = TerminalNotification()

    if config.mode == "auto_allow":
        rules = {"auto_approve": ["*"], "always_ask": [], "auto_reject": []}
    elif config.mode == "ask_always":
        rules = {"auto_approve": [], "always_ask": ["*"], "auto_reject": []}
    else:
        # milestone mode: auto-approve everything, pause at named checkpoints
        rules = {
            "auto_approve": ["Read", "Glob", "Grep", "Bash(git *)"],
            "always_ask":   ["Write", "Edit", "Bash(git push*)"],
            "auto_reject":  [],
        }

    hitl = HITLMiddleware(store=store, notification=notif, rules=rules)
    messages = [{"role": "user", "content": task}]
    await run_agent_loop(messages, hitl)
```

Milestone pauses in Tier 2 are implemented by injecting named checkpoint calls at key moments
in the agent logic (e.g., after the planning step, before any deployment command). The agent
calls `hitl.create_checkpoint(...)` explicitly at those points, independent of tool-level rules.
