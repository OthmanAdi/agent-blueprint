# Context Management

Strategies for managing the context window in long-running agent conversations.

## The Problem

LLMs have fixed context windows. In a coding agent, conversations grow fast:
- System prompt: ~10K tokens
- Each tool call + result: ~1-5K tokens
- User messages: ~0.5-2K tokens each
- After 20 turns: ~50-100K tokens used

Without context management, the agent crashes when it exceeds the limit.

## Three-Layer Strategy

### Layer 1: Tool Result Budget

Persist large outputs to disk, keep references in context:

```python
class ContextManager:
    max_inline_size = 30_000  # characters
    context_window = 200_000  # tokens
    buffer_tokens = 13_000    # safety margin
    persist_dir = ".agent-results"

    def apply_budget(self, messages: list[dict]) -> list[dict]:
        for msg in messages:
            if msg.get("role") == "user":
                for block in msg.get("content", []):
                    if block.get("type") == "tool_result":
                        content = block.get("content", "")
                        if len(content) > self.max_inline_size:
                            path = self._persist(content)
                            block["content"] = (
                                f"[Result too large ({len(content)} chars). "
                                f"Full output saved to: {path}]\n"
                                f"Preview: {content[:2000]}..."
                            )
        return messages

    def _persist(self, content: str) -> str:
        import hashlib, os
        h = hashlib.sha256(content.encode()).hexdigest()[:12]
        os.makedirs(self.persist_dir, exist_ok=True)
        path = os.path.join(self.persist_dir, f"{h}.txt")
        with open(path, "w") as f:
            f.write(content)
        return path
```

### Layer 2: Snip/Micro-Compact

Replace old tool results with placeholders:

```python
    def snip_stale(self, messages: list[dict], recent: int = 5) -> list[dict]:
        tool_result_count = 0
        for msg in reversed(messages):
            if msg.get("role") == "user":
                for block in msg.get("content", []):
                    if block.get("type") == "tool_result":
                        tool_result_count += 1

        results_to_snip = max(0, tool_result_count - recent)
        snipped = 0

        for msg in messages:
            if snipped >= results_to_snip:
                break
            if msg.get("role") == "user":
                for block in msg.get("content", []):
                    if block.get("type") == "tool_result" and snipped < results_to_snip:
                        original = block.get("content", "")
                        if len(original) > 200:
                            block["content"] = "[Old tool result content cleared]"
                            snipped += 1

        return messages
```

### Layer 3: Auto-Compact (Full Summarization)

When approaching the limit, summarize the conversation:

```python
    async def auto_compact_if_needed(
        self,
        messages: list[dict],
        system_prompt: list[str],
        tools,
    ) -> list[dict]:
        token_count = self._count_tokens(messages, system_prompt)
        threshold = self.context_window - self.buffer_tokens

        if token_count < threshold:
            return messages

        return await self._summarize(messages, system_prompt)

    async def _summarize(
        self,
        messages: list[dict],
        system_prompt: list[str],
    ) -> list[dict]:
        summary_prompt = [
            "Summarize the conversation so far. Preserve:",
            "1. Primary request and intent",
            "2. Key technical decisions made",
            "3. Files created/modified (with paths)",
            "4. Errors encountered and their fixes",
            "5. Pending tasks",
            "6. Current work in progress",
            "",
            "Be specific. Include file paths, line numbers, and code snippets.",
        ]

        # Call LLM to summarize (separate API call)
        summary = await self._call_for_summary(messages, summary_prompt)

        # Replace old messages with summary
        return [
            {
                "role": "user",
                "content": f"[Conversation compacted]\n\n{summary}",
            },
            messages[-1],  # Keep the most recent message intact
        ]

    def _count_tokens(self, messages: list[dict], system_prompt: list[str]) -> int:
        # Rough estimate: 1 token ≈ 4 chars
        total = sum(len(p) for p in system_prompt) // 4
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "") or block.get("content", "") or str(block)
                        total += len(text) // 4
        return total
```

## Complete ContextManager

```python
import os
import hashlib

class ContextManager:
    def __init__(
        self,
        context_window: int = 200_000,
        buffer_tokens: int = 13_000,
        max_inline_size: int = 30_000,
        persist_dir: str = ".agent-results",
        api_client = None,
    ):
        self.context_window = context_window
        self.buffer_tokens = buffer_tokens
        self.max_inline_size = max_inline_size
        self.persist_dir = persist_dir
        self.api_client = api_client

    def apply_budget(self, messages: list[dict]) -> list[dict]:
        return self._apply_budget_impl(messages)

    def snip_stale(self, messages: list[dict], recent: int = 5) -> list[dict]:
        return self._snip_stale_impl(messages, recent)

    async def auto_compact_if_needed(
        self, messages: list[dict], system_prompt: list[str], tools=None
    ) -> list[dict]:
        return await self._auto_compact_impl(messages, system_prompt)

    def is_over_limit(self, messages: list[dict], system_prompt: list[str]) -> bool:
        return self._count_tokens(messages, system_prompt) > self.context_window

    def persist_result(self, content: str) -> str:
        return self._persist(content)
```
