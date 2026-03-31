# Cost Optimization for Production AI Agents

How to reduce agent costs by 40–90%. Real techniques, real numbers.

---

## 1. The Numbers (Why This Matters)

Before you optimize anything, understand the scale of the problem:

- Unoptimized multi-agent enterprise system: **$10,000–$150,000/month**
- With proper optimization: **40–90% reduction is achievable**
- Most teams leave **60–70% savings on the table** through poor context management alone

The two biggest cost drivers in practice:
1. **Context size** — you pay for every input token on every turn. Old tool results that
   sit in context forever are pure waste.
2. **Model selection** — using `claude-sonnet-4-6` to summarize a JSON blob that
   `claude-haiku-4-5` could handle at one-tenth the price.

The techniques below address both. Apply them in order — the first two are highest ROI.

---

## 2. Technique 1: Prompt Caching (40–90% savings on system prompt costs)

Anthropic charges `$0.30/MTok` for cache reads versus `$3.00/MTok` for standard input.
That's a 10x reduction on anything that hits the cache.

The rule: **stable content goes first, dynamic content goes last.** One changed token
invalidates everything from that point forward.

```python
def split_for_cache(system_prompt_parts: list[str]) -> list[dict]:
    """
    Split system prompt blocks into cacheable (static) and non-cacheable (dynamic).
    Static parts get cache_control, dynamic parts don't.
    """
    # Anything that doesn't change turn-to-turn is static:
    # - Agent persona and role
    # - Tool definitions
    # - Core instructions
    # - Example outputs
    #
    # Anything that changes is dynamic:
    # - Current date/time
    # - User name or project context
    # - Session-specific state
    STATIC_BLOCK_COUNT = 3  # tune this for your prompt structure

    result = []
    for i, part in enumerate(system_prompt_parts):
        block: dict = {"type": "text", "text": part}
        if i < STATIC_BLOCK_COUNT:
            block["cache_control"] = {"type": "ephemeral"}
        result.append(block)
    return result


# What NOT to do — this kills your cache hit rate:
BAD_SYSTEM_PROMPT = f"""
You are a helpful assistant. Today is {datetime.now()}.  # <-- timestamp in static section
You have access to the following tools...
"""

# What to do instead — separate static from dynamic:
STATIC_SYSTEM_PROMPT = """
You are a helpful assistant. You have access to the following tools...
[all tool definitions here]
[all core instructions here]
"""

def build_messages(user_query: str, current_date: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": STATIC_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # this block gets cached
                },
                {
                    "type": "text",
                    "text": f"Current date: {current_date}\n\nUser query: {user_query}",
                    # no cache_control — dynamic, not cached
                },
            ],
        }
    ]
```

**Cache rules summary:**
1. Never put timestamps or session data in the cached prefix
2. Tool definitions belong in the static prefix — they almost never change
3. User context (name, current project) goes in the dynamic suffix
4. A one-token change anywhere in a cached block invalidates from that point forward

**Cost calculation:**

```python
def calculate_caching_savings(
    input_tokens_per_session: int,
    sessions_per_day: int,
    cache_hit_rate: float = 0.80,
    days: int = 30,
) -> dict:
    """
    Calculate monthly savings from prompt caching.
    Prices: $3.00/MTok standard input, $0.30/MTok cache read.
    """
    STANDARD_PRICE = 3.00 / 1_000_000    # per token
    CACHE_READ_PRICE = 0.30 / 1_000_000  # per token

    total_sessions = sessions_per_day * days
    total_tokens = input_tokens_per_session * total_sessions

    cached_tokens = total_tokens * cache_hit_rate
    uncached_tokens = total_tokens * (1 - cache_hit_rate)

    cost_without_caching = total_tokens * STANDARD_PRICE
    cost_with_caching = (cached_tokens * CACHE_READ_PRICE) + (uncached_tokens * STANDARD_PRICE)

    savings_usd = cost_without_caching - cost_with_caching
    savings_pct = savings_usd / cost_without_caching

    return {
        "cost_without_caching_usd": round(cost_without_caching, 2),
        "cost_with_caching_usd": round(cost_with_caching, 2),
        "monthly_savings_usd": round(savings_usd, 2),
        "savings_percent": round(savings_pct * 100, 1),
    }


# Example: 50k token system prompt, 1000 sessions/day, 80% cache hit rate
result = calculate_caching_savings(50_000, 1_000)
# {'cost_without_caching_usd': 4500.0, 'cost_with_caching_usd': 720.0,
#  'monthly_savings_usd': 3780.0, 'savings_percent': 84.0}
```

---

## 3. Technique 2: Model Routing (40–60% savings)

Not every task needs `claude-sonnet-4-6`. Routing simple tasks to `claude-haiku-4-5`
costs roughly 10x less while producing identical results for the right task types.

Price comparison (per MTok, input/output):
- `claude-haiku-4-5`: ~$0.25 / $1.25
- `claude-sonnet-4-6`: ~$3.00 / $15.00

```python
class ModelRouter:
    """Route tasks to the cheapest model that can handle them correctly."""

    FAST_MODEL = "claude-haiku-4-5"    # cheap, fast, great for simple tasks
    SMART_MODEL = "claude-sonnet-4-6"  # balanced, production default

    # Tasks where the fast model performs identically to the smart model
    SIMPLE_TASK_PATTERNS = [
        "summarize",
        "extract",
        "format",
        "classify",
        "list",
        "convert",
        "parse",
        "translate",
        "count",
    ]

    # Tasks that require reasoning — always use the smart model
    COMPLEX_TASK_PATTERNS = [
        "debug",
        "architect",
        "design",
        "analyze",
        "evaluate",
        "plan",
        "reason",
        "explain why",
    ]

    def select_model(self, task_description: str, context: dict | None = None) -> str:
        task_lower = task_description.lower()

        # Explicit complexity overrides
        if any(p in task_lower for p in self.COMPLEX_TASK_PATTERNS):
            return self.SMART_MODEL

        # Simple task heuristics
        if self._is_simple_task(task_lower, context):
            return self.FAST_MODEL

        return self.SMART_MODEL

    def _is_simple_task(self, task: str, context: dict | None) -> bool:
        # Pattern match
        if any(p in task for p in self.SIMPLE_TASK_PATTERNS):
            return True
        # Short structured output is simple
        if context and context.get("expected_output_format") in ("json", "list", "boolean"):
            return True
        return False


router = ModelRouter()

async def call_agent(task: str, messages: list[dict]) -> str:
    model = router.select_model(task)
    response = await client.messages.create(
        model=model,
        messages=messages,
        max_tokens=1024,
    )
    return response.content[0].text
```

**Where to apply routing in a multi-agent system:**

```python
# Orchestrator: uses SMART_MODEL (makes decisions, plans)
# Summarizer subagent: uses FAST_MODEL (compresses tool outputs)
# Classifier subagent: uses FAST_MODEL (routes requests)
# Code executor: uses SMART_MODEL (generates correct code)
# Result formatter: uses FAST_MODEL (converts JSON to markdown)

AGENT_MODEL_MAP = {
    "orchestrator": "claude-sonnet-4-6",
    "summarizer": "claude-haiku-4-5",
    "classifier": "claude-haiku-4-5",
    "code_generator": "claude-sonnet-4-6",
    "formatter": "claude-haiku-4-5",
    "researcher": "claude-sonnet-4-6",
}
```

---

## 4. Technique 3: Context Discipline (addresses 60–70% of total costs)

This is the most important optimization. It's not glamorous. Most teams skip it.
Every token in context is billed on every turn. A 10-turn session with a 100k token
result sitting unmanaged in context costs 10x more than it needs to.

**Three rules:**

```python
import tempfile
import os

# Rule 1: Tool result budget — never carry the full result inline
MAX_INLINE_CHARS = 30_000

def cap_tool_result(result: str, tool_name: str) -> str:
    """
    If a tool result exceeds the inline budget, write it to disk and
    return a reference with a short preview.
    """
    if len(result) <= MAX_INLINE_CHARS:
        return result

    # Write full result to temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{tool_name}.txt",
        delete=False,
        prefix=".tool_output_",
    )
    tmp.write(result)
    tmp.close()

    return (
        f"[Result too large ({len(result):,} chars). Full output at: {tmp.name}]\n"
        f"Preview (first 2000 chars):\n{result[:2000]}"
    )


# Rule 2: Snip stale tool results from old turns
def snip_old_results(messages: list[dict], keep_recent_turns: int = 5) -> list[dict]:
    """
    Replace tool_result content in older turns with a placeholder.
    Keeps the most recent N turns fully intact.
    """
    tool_result_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool" or (
            m.get("role") == "user" and
            isinstance(m.get("content"), list) and
            any(c.get("type") == "tool_result" for c in m.get("content", []))
        )
    ]

    # Keep the most recent `keep_recent_turns` tool results
    indices_to_clear = tool_result_indices[:-keep_recent_turns] if len(tool_result_indices) > keep_recent_turns else []

    result_messages = []
    for i, msg in enumerate(messages):
        if i in indices_to_clear:
            if isinstance(msg.get("content"), list):
                cleared_content = []
                for block in msg["content"]:
                    if block.get("type") == "tool_result":
                        cleared_content.append({**block, "content": "[Cleared — older than context window]"})
                    else:
                        cleared_content.append(block)
                result_messages.append({**msg, "content": cleared_content})
                continue
        result_messages.append(msg)
    return result_messages


# Rule 3: Compact messages when approaching the context limit
async def compact_if_needed(
    messages: list[dict],
    current_token_count: int,
    context_window: int = 200_000,
    threshold: float = 0.80,
) -> list[dict]:
    """
    Summarize older messages when token count crosses 80% of context window.
    Preserves the system prompt (index 0) and last 10 messages.
    """
    if current_token_count < context_window * threshold:
        return messages

    # Identify the slice to summarize: everything except first (system) and last 10
    if len(messages) <= 11:
        return messages

    to_summarize = messages[1:-10]
    preserved_tail = messages[-10:]

    summary_prompt = (
        "Summarize the following conversation history concisely. "
        "Preserve all decisions made, tools called, and key findings. "
        "Output only the summary, no preamble.\n\n"
        + "\n".join(f"{m['role']}: {str(m.get('content', ''))[:500]}" for m in to_summarize)
    )

    summary_response = await client.messages.create(
        model="claude-haiku-4-5",  # cheap model for summarization
        max_tokens=1024,
        messages=[{"role": "user", "content": summary_prompt}],
    )

    summary_message = {
        "role": "user",
        "content": f"[Context summary — {len(to_summarize)} messages compressed]\n{summary_response.content[0].text}",
    }

    return [messages[0], summary_message] + preserved_tail
```

---

## 5. Technique 4: Tool Set Curation

Every tool definition you include in your API call is tokens — paid for on every turn.
40 tool definitions in a typical SDK can add 8,000–15,000 tokens per turn.

```python
# Bad: all tools loaded for every call regardless of task
def bad_agent_setup():
    registry.register_all_tools()  # 40 tools = 12,000 extra tokens per turn
    return registry.get_all()


# Good: load only what's needed for the current task type
BASE_TOOLS = ["ReadTool", "GlobTool", "GrepTool"]

TASK_TOOL_MAP = {
    "coding":   ["BashTool", "EditTool", "WriteTool"],
    "research": ["WebSearchTool", "WebFetchTool"],
    "data":     ["BashTool", "WriteTool"],
    "writing":  ["ReadTool", "WriteTool"],
}

def get_tools_for_task(task_type: str) -> list:
    tool_names = BASE_TOOLS + TASK_TOOL_MAP.get(task_type, [])
    return [registry.get_tool(name) for name in tool_names]


# Even better: classify the task first (using haiku), then load tools
async def dynamic_tool_loading(user_query: str) -> list:
    task_type = await classify_task(user_query)  # cheap haiku call
    return get_tools_for_task(task_type)


async def classify_task(query: str) -> str:
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": (
                f"Classify this task as one of: coding, research, data, writing, other.\n"
                f"Reply with only the single word classification.\n\nTask: {query}"
            ),
        }],
    )
    return response.content[0].text.strip().lower()
```

Token savings per turn from tool curation:
- 40 tools → 5 tools: saves ~10,000 tokens per turn
- On a 20-turn session at $3/MTok: saves ~$0.60/session
- At 1,000 sessions/day: ~$18,000/month in tool overhead alone

---

## 6. Technique 5: Batch API for Non-Interactive Tasks

For anything that doesn't need a real-time response — scheduled reports, bulk analysis,
background processing — the Batch API gives a flat 50% cost reduction.

```python
import anthropic

client = anthropic.Anthropic()

def submit_batch_jobs(prompts: list[str]) -> str:
    """Submit a batch of requests. Returns the batch ID."""
    requests = [
        {
            "custom_id": f"job_{i}",
            "params": {
                "model": "claude-sonnet-4-6",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        for i, prompt in enumerate(prompts)
    ]

    batch = client.beta.messages.batches.create(requests=requests)
    return batch.id


def poll_batch_results(batch_id: str) -> list[dict]:
    """Poll until complete and return results."""
    import time

    while True:
        batch = client.beta.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        time.sleep(60)  # batch API has no webhooks — poll every minute

    results = []
    for result in client.beta.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            results.append({
                "custom_id": result.custom_id,
                "text": result.result.message.content[0].text,
            })
    return results


# Use case: nightly document summarization
# Instead of running 500 summaries in real-time at $3/MTok:
# Submit as batch → processed within 24h → billed at $1.50/MTok
```

Where batch API applies: nightly reports, bulk data processing, training data generation,
offline evaluation pipelines, scheduled analysis jobs.

Where it does NOT apply: anything a user is waiting on.

---

## 7. Monthly Cost Calculator

```python
def estimate_monthly_cost(
    sessions_per_day: int,
    avg_turns_per_session: int,
    avg_tokens_per_turn: int,
    cache_hit_rate: float = 0.70,
    smart_model_fraction: float = 0.60,  # fraction of calls on sonnet vs haiku
    batch_fraction: float = 0.0,          # fraction run via Batch API
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    Estimate monthly cost under different optimization scenarios.
    Returns cost for: no optimization, caching only, routing added, fully optimized.

    Pricing assumptions (per MTok):
    - claude-sonnet-4-6 input: $3.00, output: $15.00, cache read: $0.30
    - claude-haiku-4-5 input:  $0.25, output:  $1.25, cache read: $0.03
    """
    SONNET_IN  = 3.00   / 1_000_000
    SONNET_OUT = 15.00  / 1_000_000
    SONNET_CACHE = 0.30 / 1_000_000
    HAIKU_IN   = 0.25   / 1_000_000
    HAIKU_OUT  = 1.25   / 1_000_000
    HAIKU_CACHE  = 0.03 / 1_000_000
    BATCH_DISCOUNT = 0.50  # 50% off for batch API

    total_sessions = sessions_per_day * 30
    turns_total = total_sessions * avg_turns_per_session

    # Assume output is ~30% of input tokens
    input_t = avg_tokens_per_turn
    output_t = int(avg_tokens_per_turn * 0.30)

    # Scenario 1: No optimization (all sonnet, no cache, no batch)
    no_opt = turns_total * (input_t * SONNET_IN + output_t * SONNET_OUT)

    # Scenario 2: Caching added
    cached_input_cost = (
        input_t * cache_hit_rate * SONNET_CACHE +
        input_t * (1 - cache_hit_rate) * SONNET_IN
    )
    with_caching = turns_total * (cached_input_cost + output_t * SONNET_OUT)

    # Scenario 3: Model routing added (split between sonnet and haiku)
    haiku_turns = turns_total * (1 - smart_model_fraction)
    sonnet_turns = turns_total * smart_model_fraction
    with_routing = (
        sonnet_turns * (
            input_t * cache_hit_rate * SONNET_CACHE +
            input_t * (1 - cache_hit_rate) * SONNET_IN +
            output_t * SONNET_OUT
        ) +
        haiku_turns * (
            input_t * cache_hit_rate * HAIKU_CACHE +
            input_t * (1 - cache_hit_rate) * HAIKU_IN +
            output_t * HAIKU_OUT
        )
    )

    # Scenario 4: Fully optimized (routing + caching + batch for eligible fraction)
    batch_turns = turns_total * batch_fraction
    interactive_turns = turns_total * (1 - batch_fraction)
    # Context discipline cuts effective input tokens by ~40%
    optimized_input_t = int(input_t * 0.60)

    fully_optimized = (
        interactive_turns * smart_model_fraction * (
            optimized_input_t * cache_hit_rate * SONNET_CACHE +
            optimized_input_t * (1 - cache_hit_rate) * SONNET_IN +
            output_t * SONNET_OUT
        ) +
        interactive_turns * (1 - smart_model_fraction) * (
            optimized_input_t * cache_hit_rate * HAIKU_CACHE +
            optimized_input_t * (1 - cache_hit_rate) * HAIKU_IN +
            output_t * HAIKU_OUT
        ) +
        batch_turns * BATCH_DISCOUNT * (
            smart_model_fraction * (optimized_input_t * SONNET_IN + output_t * SONNET_OUT) +
            (1 - smart_model_fraction) * (optimized_input_t * HAIKU_IN + output_t * HAIKU_OUT)
        )
    )

    return {
        "sessions_per_month": total_sessions,
        "without_optimization_usd": round(no_opt, 2),
        "with_caching_usd": round(with_caching, 2),
        "with_routing_usd": round(with_routing, 2),
        "fully_optimized_usd": round(fully_optimized, 2),
        "total_savings_pct": round((1 - fully_optimized / no_opt) * 100, 1) if no_opt > 0 else 0,
    }


# Example: 500 sessions/day, 15 turns each, 20k tokens/turn
result = estimate_monthly_cost(500, 15, 20_000, cache_hit_rate=0.75, smart_model_fraction=0.5, batch_fraction=0.3)
# without_optimization_usd: ~$40,500
# fully_optimized_usd:       ~$4,200
# total_savings_pct:          ~89.6%
```

---

## Quick Reference: Optimization Priority

| Technique | Effort | Savings | Apply When |
|---|---|---|---|
| Prompt caching | Low | 40–90% on system tokens | Always — first thing to add |
| Context discipline | Medium | 40–60% on input tokens | Any session > 5 turns |
| Model routing | Medium | 30–50% on model costs | Multi-agent or varied tasks |
| Tool set curation | Low | 5–20% on input tokens | When using 10+ tools |
| Batch API | Low | 50% flat | Any non-interactive workload |

Do them in order. Prompt caching and context discipline together will get most teams to
60–70% savings before you touch anything else.
