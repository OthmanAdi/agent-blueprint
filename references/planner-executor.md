# Planner + Executor Architecture

The dominant multi-agent pattern in 2026. The most important architectural upgrade from single-agent loops.

## 1. Why Single Agents Fail at Scale

A single agent running a complex task tries to do three things simultaneously:
- **Plan**: decide what needs doing and in what order
- **Execute**: run tools, call APIs, write files
- **Verify**: check that results match the goal

These three responsibilities pollute each other. Planning requires a broad view of the entire task. Execution requires narrow, focused context on a single subtask. Verification requires a skeptical, adversarial lens. Running all three in one context window means none of them get done well.

The numbers back this up. Manus analyzed their production single-agent system and found that **33% of all tool calls were just updating `todo.md`**. One in three tool calls was bookkeeping — not real work. That overhead compounds with every turn because plan state is mixed into the same context as execution history.

This connects to the **85% compounding problem** from `production-principles.md`: if each step has an 85% success rate, a 10-step task succeeds only 20% of the time. When planning, executing, and verifying compete for the same context window, the per-step reliability drops below 85% — and the math gets brutal fast.

The fix is separation of concerns. Dedicated agents for each role means:
- The planner sees only the task and its decomposition
- Each executor sees only its subtask and the tools it needs
- The verifier sees only results, not how they were produced

## 2. The Three-Agent Pattern

```
User Request
     │
     ▼
┌─────────────────┐
│  PLANNER AGENT  │  Decomposes task → typed subtask list
│  No tools       │  Output: JSON array of subtasks
└────────┬────────┘
         │
    ┌────┴────┬────────┐
    ▼         ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│WORKER1│ │WORKER2│ │WORKER3│  Run in parallel
│ bash  │ │ web   │ │ files │  Focused tool sets
│ files │ │ fetch │ │ only  │
└───┬───┘ └───┬───┘ └───┬───┘
    └────┬────┘
         ▼
┌─────────────────┐
│ VERIFIER AGENT  │  Checks all results, flags failures
│ read-only tools │  Output: pass/fail per subtask
└────────┬────────┘
         ▼
    Final result
```

Key properties:
- **Planner has no tools.** It only reasons and produces a structured plan. Tool access would pollute its planning context with execution noise.
- **Each executor gets only the tools it needs.** A web-fetching executor has no file write tools. A file-writing executor has no bash exec tools. Least-privilege for agents.
- **Verifier has read-only tools only.** It cannot modify state. It can only inspect and report.
- **Independent subtasks run in parallel.** Dependency resolution determines what can be parallelized.

## 3. Python Implementation

### Data Structures

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Subtask:
    id: str
    description: str
    tools_needed: list[str]
    depends_on: list[str]  # IDs of subtasks this one depends on
    expected_output: str

@dataclass
class SubtaskResult:
    subtask_id: str
    status: Literal["success", "failure", "skipped"]
    output: str
    error: str | None = None

@dataclass
class VerificationResult:
    subtask_id: str
    passed: bool
    reason: str

@dataclass
class WorkflowResult:
    task: str
    subtasks: list[Subtask]
    results: list[SubtaskResult]
    verification: list[VerificationResult]
    overall_passed: bool
    summary: str
```

### PlannerAgent

```python
import json
import anthropic

class PlannerAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model

    def plan(self, task: str) -> list[Subtask]:
        system = """You are a task decomposition agent. Given a task, decompose it into
concrete subtasks. Each subtask must be independently executable.

Respond ONLY with a JSON array. No explanation, no markdown, no wrapper object.

Each element must have:
- id: string, unique, e.g. "t1", "t2"
- description: what to do, specific enough that an agent with no other context can do it
- tools_needed: array of tool names from: ["bash", "read_file", "write_file", "web_fetch", "search"]
- depends_on: array of IDs that must complete before this starts (empty if independent)
- expected_output: what a successful result looks like, in one sentence"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": task}],
        )

        raw = response.content[0].text.strip()
        data = json.loads(raw)

        return [
            Subtask(
                id=item["id"],
                description=item["description"],
                tools_needed=item["tools_needed"],
                depends_on=item["depends_on"],
                expected_output=item["expected_output"],
            )
            for item in data
        ]
```

### ExecutorAgent

```python
TOOL_IMPLEMENTATIONS = {
    "bash": run_bash,
    "read_file": read_file,
    "write_file": write_file,
    "web_fetch": web_fetch,
    "search": search,
}

class ExecutorAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model

    async def execute(self, subtask: Subtask, context: str = "") -> SubtaskResult:
        # Build tool list from only what this subtask needs
        available_tools = {
            name: TOOL_IMPLEMENTATIONS[name]
            for name in subtask.tools_needed
            if name in TOOL_IMPLEMENTATIONS
        }
        tool_schemas = [get_tool_schema(name) for name in available_tools]

        system = f"""You are an execution agent. Complete the assigned subtask exactly.
Do not do anything outside the scope of the subtask.
Expected output: {subtask.expected_output}"""

        user_message = subtask.description
        if context:
            user_message = f"Context from previous steps:\n{context}\n\nTask: {subtask.description}"

        messages = [{"role": "user", "content": user_message}]

        # Run the agent loop (bounded)
        for _ in range(20):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=messages,
                tools=tool_schemas,
            )

            # Collect text and tool calls
            tool_calls = [b for b in response.content if b.type == "tool_use"]

            if not tool_calls:
                # Agent finished — extract final text
                text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                return SubtaskResult(
                    subtask_id=subtask.id,
                    status="success",
                    output=text,
                )

            # Execute tools
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for call in tool_calls:
                fn = available_tools.get(call.name)
                if fn is None:
                    result = f"Error: tool {call.name} not available to this executor"
                else:
                    try:
                        result = await fn(**call.input)
                    except Exception as e:
                        result = f"Error: {e}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": str(result),
                })

            messages.append({"role": "user", "content": tool_results})

        return SubtaskResult(
            subtask_id=subtask.id,
            status="failure",
            output="",
            error="Max turns exceeded without completing subtask",
        )
```

### VerifierAgent

```python
class VerifierAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model

    def verify(
        self,
        subtasks: list[Subtask],
        results: list[SubtaskResult],
    ) -> list[VerificationResult]:
        results_by_id = {r.subtask_id: r for r in results}
        verifications = []

        for subtask in subtasks:
            result = results_by_id.get(subtask.id)
            if result is None or result.status == "skipped":
                verifications.append(
                    VerificationResult(
                        subtask_id=subtask.id,
                        passed=False,
                        reason="Subtask was skipped or has no result",
                    )
                )
                continue

            prompt = f"""Subtask: {subtask.description}
Expected output: {subtask.expected_output}
Actual output: {result.output}
Execution status: {result.status}
Error (if any): {result.error or "none"}

Did this subtask succeed? Reply with JSON: {{"passed": true/false, "reason": "one sentence"}}"""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = json.loads(response.content[0].text)
            verifications.append(
                VerificationResult(
                    subtask_id=subtask.id,
                    passed=raw["passed"],
                    reason=raw["reason"],
                )
            )

        return verifications
```

### Orchestrator

```python
import asyncio

async def run_workflow(task: str) -> WorkflowResult:
    planner = PlannerAgent()
    executor = ExecutorAgent()
    verifier = VerifierAgent()

    # 1. Plan
    subtasks = planner.plan(task)

    # 2. Build dependency graph
    subtask_map = {s.id: s for s in subtasks}
    completed: dict[str, SubtaskResult] = {}

    # 3. Execute with topological ordering
    results: list[SubtaskResult] = []
    remaining = list(subtasks)

    while remaining:
        # Find all subtasks whose dependencies are satisfied
        ready = [
            s for s in remaining
            if all(dep in completed for dep in s.depends_on)
        ]

        if not ready:
            # Circular dependency or blocked by failure
            for s in remaining:
                results.append(SubtaskResult(
                    subtask_id=s.id,
                    status="skipped",
                    output="",
                    error="Dependency not satisfied",
                ))
            break

        # Build context from completed results
        context = "\n".join(
            f"[{r.subtask_id}] {r.output}"
            for r in results
            if r.status == "success"
        )

        # 4. Run independent subtasks in parallel
        batch_results = await asyncio.gather(
            *[executor.execute(s, context) for s in ready]
        )

        for result in batch_results:
            completed[result.subtask_id] = result
            results.append(result)

        remaining = [s for s in remaining if s.id not in completed]

    # 5. Verify
    verification = verifier.verify(subtasks, results)

    # 6. Return structured result
    overall_passed = all(v.passed for v in verification)
    failed = [v for v in verification if not v.passed]
    summary = (
        "All subtasks completed successfully."
        if overall_passed
        else f"{len(failed)} subtask(s) failed: {', '.join(v.subtask_id for v in failed)}"
    )

    return WorkflowResult(
        task=task,
        subtasks=subtasks,
        results=results,
        verification=verification,
        overall_passed=overall_passed,
        summary=summary,
    )
```

## 4. TypeScript/Bun Implementation

```typescript
import Anthropic from "@anthropic-ai/sdk";

interface Subtask {
  id: string;
  description: string;
  toolsNeeded: string[];
  dependsOn: string[];
  expectedOutput: string;
}

interface SubtaskResult {
  subtaskId: string;
  status: "success" | "failure" | "skipped";
  output: string;
  error?: string;
}

interface WorkflowResult {
  task: string;
  results: SubtaskResult[];
  overallPassed: boolean;
  summary: string;
}

const client = new Anthropic();

async function planTask(task: string): Promise<Subtask[]> {
  const response = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 2048,
    system: `Decompose the task into subtasks. Respond ONLY with a JSON array.
Each item: { id, description, toolsNeeded, dependsOn, expectedOutput }`,
    messages: [{ role: "user", content: task }],
  });

  const raw = (response.content[0] as { text: string }).text.trim();
  return JSON.parse(raw) as Subtask[];
}

async function executeSubtask(
  subtask: Subtask,
  context: string
): Promise<SubtaskResult> {
  const messages: Anthropic.MessageParam[] = [
    {
      role: "user",
      content: context
        ? `Context:\n${context}\n\nTask: ${subtask.description}`
        : subtask.description,
    },
  ];

  // Simplified: real implementation runs tool loop here
  const response = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 4096,
    system: `Execute the task. Expected output: ${subtask.expectedOutput}`,
    messages,
  });

  const text = response.content
    .filter((b) => b.type === "text")
    .map((b) => (b as { text: string }).text)
    .join("\n");

  return {
    subtaskId: subtask.id,
    status: "success",
    output: text,
  };
}

async function runWorkflow(task: string): Promise<WorkflowResult> {
  // 1. Plan
  const subtasks = await planTask(task);
  const subtaskMap = new Map(subtasks.map((s) => [s.id, s]));

  const completed = new Map<string, SubtaskResult>();
  const results: SubtaskResult[] = [];
  let remaining = [...subtasks];

  // 2. Topological execution with parallel batches
  while (remaining.length > 0) {
    const ready = remaining.filter((s) =>
      s.dependsOn.every((dep) => completed.has(dep))
    );

    if (ready.length === 0) {
      for (const s of remaining) {
        results.push({
          subtaskId: s.id,
          status: "skipped",
          output: "",
          error: "Dependency not satisfied",
        });
      }
      break;
    }

    const context = results
      .filter((r) => r.status === "success")
      .map((r) => `[${r.subtaskId}] ${r.output}`)
      .join("\n");

    // 3. Run ready subtasks in parallel
    const batchResults = await Promise.all(
      ready.map((s) => executeSubtask(s, context))
    );

    for (const result of batchResults) {
      completed.set(result.subtaskId, result);
      results.push(result);
    }

    remaining = remaining.filter((s) => !completed.has(s.id));
  }

  const failed = results.filter((r) => r.status === "failure");
  const overallPassed = failed.length === 0;

  return {
    task,
    results,
    overallPassed,
    summary: overallPassed
      ? "All subtasks completed."
      : `${failed.length} failed: ${failed.map((r) => r.subtaskId).join(", ")}`,
  };
}
```

## 5. When to Use Which Pattern

| Scenario | Pattern |
|----------|---------|
| Simple task, fewer than 5 steps | Single agent loop |
| Complex task with independent subtasks | Planner + Executor |
| Business workflow needing approval gates | Planner + Executor + Verifier + HITL |
| Research with synthesis | Planner → parallel researchers → synthesizer |
| Untrusted or expensive tool calls | Planner + Executor + Verifier |
| Real-time streaming output required | Single agent loop (simpler) |

Rule of thumb: if you catch yourself writing `if step > 3: plan again`, you need a Planner.

## 6. Dependency Resolution

The topological batch approach above handles most cases. For strict topological sort:

```python
def topological_batches(subtasks: list[Subtask]) -> list[list[Subtask]]:
    """Returns subtasks grouped into batches that can run in parallel."""
    remaining = {s.id: s for s in subtasks}
    done: set[str] = set()
    batches: list[list[Subtask]] = []

    while remaining:
        batch = [
            s for s in remaining.values()
            if all(dep in done for dep in s.depends_on)
        ]
        if not batch:
            raise ValueError(f"Circular dependency in: {list(remaining.keys())}")
        for s in batch:
            del remaining[s.id]
            done.add(s.id)
        batches.append(batch)

    return batches
```

Usage in orchestrator:

```python
for batch in topological_batches(subtasks):
    context = build_context(completed_results)
    batch_results = await asyncio.gather(
        *[executor.execute(s, context) for s in batch]
    )
    completed_results.extend(batch_results)
```

This gives you guaranteed parallel execution within each batch and correct sequential ordering between batches with no polling or sleep.

## 7. The Supervisor Pattern (LangGraph)

LangGraph implements this architecture via graph nodes and directed edges. The planner node outputs a `Plan` object into graph state. A `supervisor` node reads the plan and routes to the correct executor node. Each executor node writes its result back into state. The verifier node reads all results and decides whether to terminate or re-plan.

```python
from langgraph.graph import StateGraph, END

builder = StateGraph(WorkflowState)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("verifier", verifier_node)

builder.add_edge("planner", "executor")
builder.add_conditional_edges(
    "verifier",
    lambda state: END if state.all_passed else "planner",
)

graph = builder.compile(interrupt_before=["executor"])  # HITL gate
```

The `interrupt_before=["executor"]` line inserts a human-in-the-loop checkpoint before any execution starts. The human sees the plan and approves or modifies it before the executors run. This is the standard pattern for high-stakes workflows.

## 8. Common Mistakes

**Conflating planner and verifier.** If the same LLM call plans the task and checks the results, it will rationalize failures. The planner invented the plan; it is psychologically committed to it. Use a separate verifier with a separate system prompt that only sees outputs, not intentions.

**Giving executors all 40 tools.** An executor with access to `delete_database` while doing a web search is a liability. The tool list IS the executor's permission system. If it doesn't need it, don't give it. Narrow tool sets also reduce the chance of the model choosing the wrong tool.

**Giving executors insufficient context.** Executors cannot see the parent conversation. They only see their subtask description and the context string you explicitly pass. If subtask `t3` depends on a file path produced by `t1`, you must pass that path explicitly in the context string — the executor will not find it by looking at previous messages.
