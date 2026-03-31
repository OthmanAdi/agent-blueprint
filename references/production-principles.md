# Production Principles for AI Agents

> Sources: Manus engineering blog ("Building Production AI Agents"), Lance Martin's agent reliability research (LangChain), and patterns extracted from the 12-Factor Agents framework.

This document covers what separates a demo agent from one that survives contact with production. Read it before you write a single line of agent code.

---

## 1. The 85% Compounding Problem

A single LLM call that achieves 85% accuracy sounds acceptable. Chain ten of them together and the math becomes brutal.

**Compounding failure rate: `accuracy^steps`**

| Steps | Success Rate |
|-------|-------------|
| 5     | 44%         |
| 10    | 20%         |
| 15    | 9%          |
| 20    | 4%          |

A 10-step agent with 85% per-step accuracy succeeds roughly 1 in 5 runs. At 20 steps, it fails 96% of the time. This is not a model quality problem — it is an architecture problem.

### Solutions

**1. Reduce step count.** Consolidate multi-step reasoning into a single well-structured prompt where possible. Every step you remove multiplies your success rate.

**2. Add verification nodes.** After high-risk steps (external API calls, writes to disk, irreversible actions), insert a lightweight verification call that checks the output before continuing.

```python
# Bad: fire-and-forget step chain
result_a = step_a(input)
result_b = step_b(result_a)
result_c = step_c(result_b)

# Better: verify after each high-risk step
result_a = step_a(input)
assert_valid(result_a, schema=StepAOutput)   # deterministic, zero LLM cost

result_b = step_b(result_a)
verify_b = verification_node(result_b)       # cheap LLM call with narrow task
if not verify_b.passed:
    result_b = step_b(result_a, retry=True)  # one retry with context
```

**3. Use deterministic routing.** LLM reasoning should not decide what happens next in business-critical paths. If the decision can be expressed as Python `if/else`, it should be. See section 3 (Neuro-Symbolic Split) for the full rule.

---

## 2. The 6 Manus Production Principles

These come directly from Manus's engineering blog post on production agent architecture. They apply regardless of which LLM or framework you use.

---

### Principle 1: Design Around KV-Cache

In agent workflows, input tokens dominate. The real-world ratio is roughly **100:1 input-to-output tokens**. This makes cache hit rate the single biggest cost lever in production.

**The price difference:**

| Token type    | Approximate cost (claude-sonnet-4-6) |
|---------------|--------------------------------------|
| Cached input  | ~$0.30 / MTok                        |
| Uncached input| ~$3.00 / MTok                        |

A 10x cost difference. On a high-traffic agent, this compounds into the primary infrastructure line item.

**Cache invalidation rules:**
- A single character change anywhere in the static prefix invalidates the entire cache for every turn.
- Timestamps, request IDs, and session UUIDs in the system prompt kill the cache on every call.
- Append-only context (never reordering prior messages) maximizes cache hits.

**Implementation: split system prompt into static + dynamic sections**

```python
import anthropic

client = anthropic.Anthropic()

# Static prefix — always identical, will be cached after first call
STATIC_SYSTEM = """You are a production AI agent for document processing.

Your capabilities:
- Read and parse structured documents (PDF, DOCX, JSON, CSV)
- Extract entities, dates, and monetary values
- Classify documents into predefined categories
- Flag documents that require human review

Output format: always valid JSON matching the DocumentResult schema.
Never include commentary outside the JSON block.
When uncertain, set confidence < 0.7 and add reasoning to the review_notes field."""

def build_system_prompt(active_tools: list[str], user_tier: str) -> str:
    # Dynamic suffix appended after the stable prefix
    dynamic_suffix = f"""
Active tool permissions: {", ".join(sorted(active_tools))}
User tier: {user_tier}
"""
    return STATIC_SYSTEM + dynamic_suffix

def process_document(doc_content: str, active_tools: list[str], user_tier: str):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=build_system_prompt(active_tools, user_tier),
        messages=[{"role": "user", "content": doc_content}]
    )
    return response
```

Note that `sorted(active_tools)` is deliberate — deterministic serialization keeps the dynamic suffix identical across calls with the same logical state, maximizing partial cache hits.

---

### Principle 2: Mask, Don't Remove

A common pattern: dynamically remove tools from the tool list when they are not relevant to the current state. This feels smart. It destroys KV-cache.

**Never remove tools. Mask them instead.**

The tool list is part of the system prompt. Changing it breaks the cache. Instead, keep the full tool list stable and use consistent naming conventions that allow the model and your routing code to select appropriate subsets.

**Recommended approach: action namespace prefixes**

```python
ALL_TOOLS = [
    # Always present, always in this order
    {"name": "bash_run",         "description": "Execute a bash command"},
    {"name": "bash_test",        "description": "Run unit tests"},
    {"name": "file_read",        "description": "Read file contents"},
    {"name": "file_write",       "description": "Write content to a file"},
    {"name": "file_list",        "description": "List directory contents"},
    {"name": "browser_navigate", "description": "Navigate to a URL"},
    {"name": "browser_extract",  "description": "Extract content from current page"},
    {"name": "api_get",          "description": "Make a GET request"},
    {"name": "api_post",         "description": "Make a POST request"},
]

# To restrict behavior: use system prompt instructions, not tool removal
RESTRICTED_SUFFIX = """
In this session only file_read and file_list are authorized.
Do not call any other tools even if they appear in your capabilities.
"""
```

**Tool count sweet spot: under 30.**

Performance degrades measurably above 30 tools. The model's attention is diluted across too many options. For larger capability sets, use dynamic tool loadouts — load the appropriate subset at session start based on the task type, then keep that set stable for the entire session.

```python
TOOL_LOADOUTS = {
    "document_processing": ["file_read", "file_write", "file_list", "bash_run"],
    "web_research":        ["browser_navigate", "browser_extract", "api_get"],
    "code_generation":     ["file_read", "file_write", "bash_run", "bash_test"],
}

def get_tools_for_task(task_type: str) -> list:
    tool_names = TOOL_LOADOUTS.get(task_type, list(TOOL_LOADOUTS["document_processing"]))
    return [t for t in ALL_TOOLS if t["name"] in tool_names]
```

---

### Principle 3: Filesystem as Context

Think of the context window as RAM and the filesystem as disk.

**Context window (RAM):**
- Fast to access
- Volatile — lost at session end
- Finite — consuming it with large tool outputs crowds out reasoning

**Filesystem (Disk):**
- Persistent across turns and sessions
- Effectively unlimited
- Queryable with `glob` and `grep` — often faster than a vector search for most tasks

**The rule:** never put large tool outputs into context. Persist them to disk immediately and keep only a summary + file path in context.

**Critical constraint:** compression must be restorable. When dropping web content, keep the URL. When dropping document contents, keep the file path. Never lose the pointer to the full data.

```python
import json
from pathlib import Path
from datetime import datetime

WORK_DIR = Path("/tmp/agent_workspace")
WORK_DIR.mkdir(exist_ok=True)

def persist_large_result(result: dict, step_name: str) -> dict:
    """Persist result to disk, return lean summary for context."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{step_name}_{timestamp}.json"
    file_path = WORK_DIR / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Return only what the next step needs from context
    return {
        "status": "persisted",
        "file_path": str(file_path),
        "summary": {
            "record_count": len(result.get("items", [])),
            "top_result": result.get("items", [{}])[0],
            "metadata": result.get("metadata", {}),
        },
        "note": "Full result at file_path. Use file_read tool to retrieve if needed.",
    }

# Usage in agent step
raw_api_response = call_external_api(query)

if len(str(raw_api_response)) > 2000:
    context_entry = persist_large_result(raw_api_response, "api_search")
else:
    context_entry = raw_api_response  # small enough to carry directly
```

For most retrieval tasks, prefer `glob`/`grep` on disk over spinning up a vector store. Vector stores add latency, cost, and operational complexity. A well-structured workspace directory with consistent naming is often sufficient.

---

### Principle 4: Manipulate Attention Through Recitation

LLMs suffer from "lost in the middle" degradation — as context grows, attention to the original task goal weakens. In a long agentic run, the model starts optimizing for the most recent observations rather than the original objective.

**The data point from Manus:** complex tasks average around 50 tool calls. Without goal re-injection, the model drifts.

**Solution:** re-read the task plan every N turns. Explicitly inject the plan into context as an observation, not just leave it in the original prompt.

```python
PLAN_REFRESH_INTERVAL = 10  # turns

def run_agent_loop(task: str, plan_file: str = "task_plan.md") -> str:
    messages = [{"role": "user", "content": task}]
    turn = 0

    while True:
        # Inject plan reminder at regular intervals
        if turn > 0 and turn % PLAN_REFRESH_INTERVAL == 0:
            plan_content = Path(plan_file).read_text(encoding="utf-8")
            plan_reminder = {
                "role": "user",
                "content": f"[PLAN REMINDER — turn {turn}]\n\n{plan_content}\n\nContinue working toward the plan above."
            }
            messages.append(plan_reminder)

        response = call_llm(messages)
        action = parse_action(response)

        if action.type == "finish":
            return action.result

        observation = execute_action(action)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": observation})
        turn += 1
```

The plan file itself should be written at task start and updated as subtasks complete. See `planning-with-files` skill for the full Manus-style file-based planning pattern.

---

### Principle 5: Keep Failures In Context

When a tool call fails — network error, wrong argument, unexpected output — the instinct is to clean it up. Remove it from context, try again with a fresh state. This is wrong.

**Failed actions with stack traces let the model implicitly update its beliefs.**

The model sees: "I tried X with arguments Y. It returned error Z." This is information. It constrains the search space for the next attempt. Removing it forces the model to rediscover the same constraint, often by making the same mistake.

Manus describes this as "a clear signal of TRUE agentic behavior" — the model learning within a session from its own failures without requiring explicit error handling logic.

```python
def execute_with_failure_retention(action, messages: list) -> list:
    """Execute action and keep result in context regardless of success."""
    try:
        result = execute_action(action)
        observation = {
            "status": "success",
            "output": result,
        }
    except Exception as e:
        # Do NOT retry silently. Keep the failure.
        observation = {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": format_exc(),
            "action_attempted": action.to_dict(),
        }

    # Always append, never replace or delete
    messages.append({
        "role": "user",
        "content": f"[TOOL RESULT]\n{json.dumps(observation, indent=2)}"
    })
    return messages
```

**What not to do:**
```python
# Bad: silent retry wipes the failure from context
try:
    result = execute_action(action)
except Exception:
    result = execute_action(action)  # context has no record of first failure
```

---

### Principle 6: Break Pattern Mimicry

LLMs are pattern-completion machines. In a long agentic run, repetitive action-observation pairs cause the model to start mimicking the pattern of recent turns rather than reasoning about the task. This manifests as:

- Repeating the same tool call with slightly different arguments
- Generating structurally identical responses even when content should differ
- Getting stuck in local loops

**Solution:** vary the serialization templates and response framing across turns. Break the visual rhythm of the context.

```python
import random

OBSERVATION_TEMPLATES = [
    "[TOOL RESULT]\n{content}",
    "[STEP OUTPUT — turn {turn}]\n{content}",
    "[RESULT]\n{content}",
    "[OBSERVATION]\nStep {turn} completed.\n{content}",
]

def format_observation(content: str, turn: int) -> str:
    # Vary template to break pattern mimicry
    template = OBSERVATION_TEMPLATES[turn % len(OBSERVATION_TEMPLATES)]
    return template.format(content=content, turn=turn)
```

Similarly, vary how you present the plan reminder across injections, and avoid identical preambles on every assistant turn.

---

## 3. The Neuro-Symbolic Split

For any agent operating on business-critical or safety-sensitive paths, separate the architecture into two explicit layers:

**Thinking Layer (LLM):** handles ambiguity, natural language, interpretation, and generation. Anything that requires actual reasoning.

**Orchestration Layer (deterministic code):** handles routing, validation, state transitions, and action execution. Anything that can be expressed as logic.

**The rule:** if the decision can be expressed as Python `if/else`, it must not be delegated to the LLM.

```python
# Wrong: LLM decides whether to escalate
response = llm.invoke("Should I escalate this ticket? Ticket: " + ticket_content)
if "yes" in response.lower():
    escalate(ticket)

# Right: deterministic rule, LLM only extracts structured data
ticket_data = llm.invoke(
    "Extract: priority (low/medium/high/critical), category, sentiment_score (0-1). "
    "Return JSON only. Ticket: " + ticket_content
)
parsed = json.loads(ticket_data)

# Routing is pure Python — auditable, testable, deterministic
if parsed["priority"] == "critical" or parsed["sentiment_score"] < 0.2:
    escalate(ticket)
elif parsed["priority"] == "high":
    assign_to_senior(ticket)
else:
    assign_to_queue(ticket)
```

This pattern makes your agent auditable. Every routing decision has an explicit, inspectable code path. The LLM's role is limited to extraction and generation — the domains where it adds actual value.

---

## 4. The Context Discipline Rules

These are hard constraints, not suggestions.

**Carry only what the current step needs.**
Not full history. Not all prior tool outputs. The current step's reasoning context. Prior turns are already in the message list — do not duplicate them as observations.

**Large tool results go to disk immediately.**
Define "large" as anything over ~2000 tokens. Persist, keep summary + path.

**Use `glob`/`grep` on disk instead of vector stores for most tasks.**
Vector stores are warranted when you need semantic similarity over a large corpus you cannot enumerate. For typical agent workspaces (a few hundred files from one session), they are operational overhead with no benefit.

**Maximum 30 tools in registry at any time.**
If your agent needs more than 30 capabilities, use dynamic loadouts selected at session start by task type. The loadout is then stable for the session.

**Rotate out stale tool results after N turns.**
Tool output from 40 turns ago about a web page that no longer matters is noise. Implement a sliding window or explicit result expiry:

```python
RESULT_TTL_TURNS = 20

def prune_stale_results(messages: list, current_turn: int) -> list:
    pruned = []
    for msg in messages:
        if msg.get("result_turn") and (current_turn - msg["result_turn"]) > RESULT_TTL_TURNS:
            # Replace content with a pointer stub
            pruned.append({
                **msg,
                "content": f"[RESULT PRUNED — stale after {RESULT_TTL_TURNS} turns. "
                           f"File path if persisted: {msg.get('file_path', 'N/A')}]"
            })
        else:
            pruned.append(msg)
    return pruned
```

---

## 5. Quick Production Checklist

Run this before deploying any agent to production. Every unchecked item is a known failure mode.

- [ ] System prompt is split: stable static prefix (cached) + dynamic suffix (uncached)
- [ ] No timestamps, request IDs, or session UUIDs in the static system prompt
- [ ] Tool count is 30 or fewer in any single session
- [ ] Tool list order is deterministic and never changes mid-session
- [ ] Large tool outputs (> 2000 tokens) are persisted to disk; only summary + path in context
- [ ] Failed steps are kept in context with full error detail — never silently deleted or retried
- [ ] Verification node present after each high-risk, irreversible, or external-write step
- [ ] All routing decisions that can be expressed as `if/else` are implemented in Python, not delegated to LLM
- [ ] Plan is re-injected into context every 10 turns for tasks expected to exceed 20 steps
- [ ] Observation templates vary across turns to prevent pattern mimicry
- [ ] Step count for the core workflow is audited — target single digits, flag anything above 15
- [ ] Agent has been tested with deliberate mid-run failures to verify failure retention behavior

---

## Further Reading

- Manus engineering blog: "Building Production AI Agents" — source for the 6 principles above
- Lance Martin (LangChain): agent reliability research, compounding error analysis
- `planning-with-files` skill — Manus-style file-based planning implementation
- `agent-blueprint/references/context-management.md` — context window discipline in depth
- `agent-blueprint/references/architecture.md` — system architecture patterns
