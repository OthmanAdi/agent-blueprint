# Observability for Production AI Agents

The difference between "logs exist" and "logs are actionable" is the difference between a system that
fails silently and one that tells you exactly what broke, when, and why.

---

## 1. What to Trace (The Minimum)

Every production agent must emit these on every event — no exceptions:

- **Session ID + turn number** — correlates every event back to a user interaction
- **Tool name + input + output + duration_ms** — for every tool call
- **Token counts** — input, output, cache_read, cache_creation — per turn
- **Cost in USD** — per turn and cumulative for the session
- **Error type + stack trace + recovery status** — was this recovered or fatal?

If your agent doesn't emit these, you have logs. You don't have observability.

---

## 2. The Trace Structure

```python
from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class ToolCallTrace:
    tool_name: str
    tool_id: str
    input: dict
    output_preview: str      # first 500 chars — never store full output
    success: bool
    error: Optional[str]
    duration_ms: int
    permission: str          # "auto_allow" | "user_approved" | "user_denied"


@dataclass
class TurnTrace:
    turn: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cost_usd: float
    duration_ms: int
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class AgentTrace:
    session_id: str
    trace_id: str             # UUID for this specific run
    model: str
    total_turns: int
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    duration_ms: int
    status: str               # "success" | "error" | "interrupted"
    turns: list[TurnTrace] = field(default_factory=list)
    error: Optional[str] = None

    @classmethod
    def new(cls, session_id: str, model: str) -> "AgentTrace":
        return cls(
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            model=model,
            total_turns=0,
            total_cost_usd=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cache_read_tokens=0,
            duration_ms=0,
            status="running",
        )
```

The `output_preview` cap matters. Full tool outputs can be megabytes. Storing them
in trace records destroys your disk and your wallet if you're sending to a trace backend.
Store a preview, store a reference path if you need the full output.

---

## 3. Python OpenTelemetry Integration

OpenTelemetry is the vendor-neutral standard. Use it and you can ship to Jaeger,
Grafana Tempo, Honeycomb, or Datadog without changing instrumentation code.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from contextlib import contextmanager
import json


def setup_tracing(service_name: str = "agent-blueprint", endpoint: str = "http://localhost:4317"):
    """Initialize the tracer — call once at startup."""
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


tracer = trace.get_tracer("agent-blueprint")


@contextmanager
def trace_turn(turn: int, session_id: str):
    """Context manager for one agent turn."""
    with tracer.start_as_current_span(f"agent_turn_{turn}") as span:
        span.set_attribute("session.id", session_id)
        span.set_attribute("turn.number", turn)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            raise


@contextmanager
def trace_tool_call(tool_name: str, tool_input: dict):
    """Context manager for one tool execution."""
    with tracer.start_as_current_span(f"tool.{tool_name}") as span:
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.input", json.dumps(tool_input)[:500])
        start = time.monotonic()
        try:
            yield span
            duration_ms = int((time.monotonic() - start) * 1000)
            span.set_attribute("tool.duration_ms", duration_ms)
            span.set_attribute("tool.success", True)
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("tool.success", False)
            span.set_status(trace.StatusCode.ERROR, str(e))
            raise
```

Usage in the agent loop:

```python
async def run_turn(agent, messages, turn: int, session_id: str):
    with trace_turn(turn, session_id) as span:
        response = await agent.step(messages)
        span.set_attribute("tokens.input", response.usage.input_tokens)
        span.set_attribute("tokens.output", response.usage.output_tokens)
        span.set_attribute("tokens.cache_read", response.usage.cache_read_input_tokens)
        return response
```

---

## 4. LangSmith Integration (2 Environment Variables)

If you're on LangChain or LangGraph, instrumentation is a two-liner:

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__your_key_here"

# That's it. Every LangChain/LangGraph call is now traced automatically.
# You'll see the full trace in the LangSmith dashboard.
```

For non-LangChain agents, use the RunTree API directly:

```python
from langsmith import Client
from datetime import datetime, timezone

client = Client()

def run_agent_with_tracing(prompt: str) -> str:
    # Start the run
    run = client.create_run(
        name="agent_session",
        run_type="chain",
        inputs={"query": prompt},
        start_time=datetime.now(timezone.utc),
    )

    try:
        result = execute_agent(prompt)  # your agent call
        client.update_run(
            run.id,
            outputs={"result": result},
            end_time=datetime.now(timezone.utc),
        )
        return result
    except Exception as e:
        client.update_run(
            run.id,
            error=str(e),
            end_time=datetime.now(timezone.utc),
        )
        raise


def log_tool_call_to_langsmith(parent_run_id, tool_name: str, inputs: dict, output: str):
    """Log a child tool run under the parent session."""
    tool_run = client.create_run(
        name=tool_name,
        run_type="tool",
        inputs=inputs,
        parent_run_id=parent_run_id,
        start_time=datetime.now(timezone.utc),
    )
    client.update_run(
        tool_run.id,
        outputs={"result": output[:500]},
        end_time=datetime.now(timezone.utc),
    )
```

---

## 5. Simple Local File Tracer (No External Dependencies)

For development, local testing, or environments where you can't run an OTLP collector.
Writes JSONL — one JSON object per line, easy to tail, grep, and analyze.

```python
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone


class FileTracer:
    def __init__(self, trace_dir: str = ".agent-traces"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.current_trace: AgentTrace | None = None
        self.trace_file: Path | None = None
        self._session_start: float = 0.0

    def start_session(self, session_id: str, model: str = "claude-sonnet-4-6") -> str:
        """Start a new trace session. Returns the trace file path."""
        self.current_trace = AgentTrace.new(session_id, model)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        filename = f"{timestamp}_{session_id[:8]}.jsonl"
        self.trace_file = self.trace_dir / filename
        self._session_start = time.monotonic()
        self._write({"event": "session_start", "session_id": session_id, "model": model})
        return str(self.trace_file)

    def log_turn(self, turn_trace: TurnTrace) -> None:
        """Append a completed turn to the trace."""
        if self.current_trace is None:
            raise RuntimeError("Call start_session() before log_turn()")
        self.current_trace.turns.append(turn_trace)
        self.current_trace.total_turns += 1
        self.current_trace.total_cost_usd += turn_trace.cost_usd
        self.current_trace.total_input_tokens += turn_trace.input_tokens
        self.current_trace.total_output_tokens += turn_trace.output_tokens
        self.current_trace.total_cache_read_tokens += turn_trace.cache_read_tokens
        self._write({"event": "turn", **asdict(turn_trace)})

    def log_tool_call(self, tool_trace: ToolCallTrace) -> None:
        """Log a tool call (call before log_turn to keep ordered)."""
        self._write({"event": "tool_call", **asdict(tool_trace)})

    def end_session(self, status: str, error: str | None = None) -> AgentTrace:
        """Finalize the trace. Returns the completed AgentTrace."""
        if self.current_trace is None:
            raise RuntimeError("No active session")
        elapsed_ms = int((time.monotonic() - self._session_start) * 1000)
        self.current_trace.status = status
        self.current_trace.duration_ms = elapsed_ms
        self.current_trace.error = error
        self._write({"event": "session_end", **asdict(self.current_trace)})
        trace = self.current_trace
        self.current_trace = None
        return trace

    def summarize(self) -> str:
        """Human-readable summary of the completed trace."""
        if not self.trace_file or not self.trace_file.exists():
            return "No trace found."
        events = [json.loads(line) for line in self.trace_file.read_text().splitlines()]
        end_event = next((e for e in reversed(events) if e.get("event") == "session_end"), None)
        if not end_event:
            return "Session not yet ended."
        lines = [
            f"Session: {end_event['session_id']}",
            f"Status:  {end_event['status']}",
            f"Turns:   {end_event['total_turns']}",
            f"Cost:    ${end_event['total_cost_usd']:.4f}",
            f"Tokens:  {end_event['total_input_tokens']} in / {end_event['total_output_tokens']} out / {end_event['total_cache_read_tokens']} cache_read",
            f"Time:    {end_event['duration_ms']}ms",
        ]
        if end_event.get("error"):
            lines.append(f"Error:   {end_event['error']}")
        return "\n".join(lines)

    def _write(self, data: dict) -> None:
        with self.trace_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")
```

---

## 6. Surfacing Actionable Failures

Raw traces are not enough. You need automated analysis that tells you when something
is wrong before a user complains.

The three failure patterns that show up in production traces most often:

**Pattern 1: Repeated tool failure**
The same tool fails 3+ times in one session. This means the tool is broken, not the agent.

**Pattern 2: Context explosion**
`input_tokens` doubling each turn. Context is not being managed — old results are
accumulating and the session will hit the context limit and crash or hallucinate.

**Pattern 3: Cost runaway**
Session cost is 10x the session average. This almost always means an infinite loop
or a tool that returns a huge payload that gets fed back into context every turn.

```python
def analyze_trace(trace: AgentTrace) -> list[str]:
    """Return list of actionable failure signals."""
    warnings = []

    # Pattern 1: Repeated tool failure
    tool_failure_counts: dict[str, int] = {}
    for turn in trace.turns:
        for tc in turn.tool_calls:
            if not tc.success:
                tool_failure_counts[tc.tool_name] = tool_failure_counts.get(tc.tool_name, 0) + 1
    for tool_name, count in tool_failure_counts.items():
        if count >= 3:
            warnings.append(
                f"TOOL_BROKEN: '{tool_name}' failed {count} times in this session. "
                f"Check tool implementation, not agent logic."
            )

    # Pattern 2: Context explosion
    if len(trace.turns) >= 3:
        token_counts = [t.input_tokens for t in trace.turns]
        for i in range(1, len(token_counts)):
            if token_counts[i] > token_counts[i - 1] * 1.8:
                warnings.append(
                    f"CONTEXT_EXPLOSION: Input tokens grew {token_counts[i-1]} → {token_counts[i]} "
                    f"at turn {i+1}. Tool results are not being trimmed."
                )
                break  # one warning is enough

    # Pattern 3: Cost runaway — compare to a $2 threshold for a single session
    RUNAWAY_THRESHOLD_USD = 2.0
    if trace.total_cost_usd > RUNAWAY_THRESHOLD_USD:
        warnings.append(
            f"COST_RUNAWAY: Session cost ${trace.total_cost_usd:.2f} exceeds "
            f"${RUNAWAY_THRESHOLD_USD:.2f} threshold. Possible loop detected."
        )

    # Pattern 4: Excessive turns
    if trace.total_turns > 30:
        warnings.append(
            f"EXCESSIVE_TURNS: {trace.total_turns} turns in one session. "
            f"Agent may be stuck or task decomposition is wrong."
        )

    return warnings
```

Wire this into your `end_session()` call:

```python
trace = tracer.end_session(status="success")
for warning in analyze_trace(trace):
    logger.warning(warning)
    # Also fire to your alerting system (Slack, PagerDuty, etc.)
```

---

## 7. Key Metrics Dashboard

The 8 metrics every production agent deployment must track. Set alerts on the Warning
thresholds, wake someone up at Critical.

| Metric | Good | Warning | Critical |
|---|---|---|---|
| Success rate per session | >90% | 80–90% | <80% |
| Avg cost per session | <$0.50 | $0.50–$2.00 | >$2.00 |
| Avg turns per session | <20 | 20–40 | >40 |
| Cache hit rate | >70% | 50–70% | <50% |
| Tool error rate | <5% | 5–15% | >15% |
| P95 latency (seconds) | <30 | 30–60 | >60 |
| Context compactions/session | 0 | 1 | >1 |
| Human approval rate | <10% | 10–30% | >30% |

How to compute these from your JSONL traces:

```python
from pathlib import Path
import json
from statistics import mean, quantiles


def compute_dashboard_metrics(trace_dir: str = ".agent-traces") -> dict:
    traces = []
    for path in Path(trace_dir).glob("*.jsonl"):
        lines = path.read_text().splitlines()
        for line in reversed(lines):
            event = json.loads(line)
            if event.get("event") == "session_end":
                traces.append(event)
                break

    if not traces:
        return {}

    success_count = sum(1 for t in traces if t["status"] == "success")
    all_costs = [t["total_cost_usd"] for t in traces]
    all_turns = [t["total_turns"] for t in traces]
    all_durations_s = [t["duration_ms"] / 1000 for t in traces]

    # Cache hit rate: cache_read / (cache_read + input)
    cache_rates = []
    for t in traces:
        total_in = t["total_input_tokens"] + t["total_cache_read_tokens"]
        if total_in > 0:
            cache_rates.append(t["total_cache_read_tokens"] / total_in)

    p95_latency = quantiles(all_durations_s, n=20)[18] if len(all_durations_s) >= 2 else all_durations_s[0]

    return {
        "success_rate": success_count / len(traces),
        "avg_cost_usd": mean(all_costs),
        "avg_turns": mean(all_turns),
        "cache_hit_rate": mean(cache_rates) if cache_rates else 0.0,
        "p95_latency_s": p95_latency,
        "session_count": len(traces),
    }
```

---

## Quick Reference

```python
# Minimal instrumented agent loop
tracer = FileTracer(".agent-traces")
session_id = str(uuid.uuid4())
tracer.start_session(session_id)

try:
    for turn_num in range(1, MAX_TURNS + 1):
        turn_start = time.monotonic()
        response = await client.messages.create(...)

        tool_traces = []
        for tool_call in response.tool_calls:
            t_start = time.monotonic()
            result, success, error = await execute_tool(tool_call)
            tool_traces.append(ToolCallTrace(
                tool_name=tool_call.name,
                tool_id=tool_call.id,
                input=tool_call.input,
                output_preview=str(result)[:500],
                success=success,
                error=error,
                duration_ms=int((time.monotonic() - t_start) * 1000),
                permission="auto_allow",
            ))

        cost = compute_turn_cost(response.usage, model="claude-sonnet-4-6")
        tracer.log_turn(TurnTrace(
            turn=turn_num,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_input_tokens or 0,
            cache_creation_tokens=response.usage.cache_creation_input_tokens or 0,
            cost_usd=cost,
            duration_ms=int((time.monotonic() - turn_start) * 1000),
            tool_calls=tool_traces,
        ))

        if response.stop_reason == "end_turn":
            break

    trace = tracer.end_session("success")
except Exception as e:
    trace = tracer.end_session("error", error=str(e))
    raise
finally:
    print(tracer.summarize())
    for w in analyze_trace(trace):
        print(f"WARNING: {w}")
```
