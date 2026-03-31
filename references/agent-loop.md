# The Agent Loop

The complete streaming agent loop with error recovery, parallel tool execution, and cost tracking.

## Complete Implementation

```python
import asyncio
import json
from typing import AsyncGenerator

async def agent_loop(
    messages: list[dict],
    tools: ToolRegistry,
    permissions: PermissionSystem,
    context_manager: ContextManager,
    system_prompt: list[str],
    api_client: APIClient,
    max_turns: int = 50,
    cost_tracker: CostTracker | None = None,
) -> AsyncGenerator[Event, None]:
    turn = 0
    consecutive_errors = 0

    while turn < max_turns:
        turn += 1

        # --- Context Management Pipeline ---
        messages = context_manager.apply_budget(messages)
        messages = context_manager.snip_stale(messages)
        messages = context_manager.auto_compact_if_needed(
            messages, system_prompt, tools
        )

        # Check hard context limit
        if context_manager.is_over_limit(messages, system_prompt):
            yield ErrorEvent("Context limit exceeded. Use /compact to summarize.")
            break

        # --- Call LLM with Streaming ---
        try:
            stream = api_client.stream(
                system=system_prompt,
                messages=messages,
                tools=tools.to_api_schema(),
                stream=True,
            )

            # Collect response blocks
            text_blocks = []
            tool_blocks = []
            thinking_blocks = []
            usage = None

            async for event in stream:
                if event.type == "message_start":
                    usage = event.message.usage
                elif event.type == "content_block_start":
                    if event.content_block.type == "text":
                        yield StreamingTextEvent(start=True)
                    elif event.content_block.type == "tool_use":
                        yield ToolStartEvent(
                            tool_name=event.content_block.name,
                            tool_id=event.content_block.id,
                        )
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        text_blocks.append(event.delta.text)
                        yield StreamingTextEvent(text=event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        yield ToolInputEvent(partial_json=event.delta.partial_json)
                    elif event.delta.type == "thinking_delta":
                        thinking_blocks.append(event.delta.thinking)
                        yield ThinkingEvent(thinking=event.delta.thinking)
                elif event.type == "message_delta":
                    if event.delta.stop_reason:
                        stop_reason = event.delta.stop_reason
                    if event.usage:
                        usage = event.usage

            consecutive_errors = 0

        except PromptTooLongError:
            # Context too large - compact and retry
            messages = await context_manager.reactive_compact(
                messages, system_prompt
            )
            continue
        except APIError as e:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                yield ErrorEvent(f"API error after 3 retries: {e}")
                break
            yield RetryEvent(error=str(e), attempt=consecutive_errors)
            await asyncio.sleep(2 ** consecutive_errors)
            continue

        # --- Track Costs ---
        if cost_tracker and usage:
            cost_tracker.add(usage)
            yield CostEvent(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read=usage.cache_read_input_tokens,
                cache_creation=usage.cache_creation_input_tokens,
                total_cost_usd=cost_tracker.total_cost(),
            )

        # --- Build Assistant Message ---
        content = []
        if thinking_blocks:
            content.append({"type": "thinking", "thinking": "".join(thinking_blocks)})
        if text_blocks:
            content.append({"type": "text", "text": "".join(text_blocks)})

        for tb in tool_blocks:
            content.append(tb)

        messages.append({"role": "assistant", "content": content})

        # --- No tools used = done ---
        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if not tool_uses:
            yield DoneEvent(stop_reason=stop_reason, turn_count=turn)
            break

        # --- Execute Tools ---
        # Separate into concurrent and sequential
        concurrent = []
        sequential = []
        for tu in tool_uses:
            tool = tools.get(tu["name"])
            if tool and tool.is_concurrency_safe:
                concurrent.append(tu)
            else:
                sequential.append(tu)

        # Run concurrent tools in parallel
        if concurrent:
            results = await asyncio.gather(*[
                execute_tool_with_permissions(
                    tu, tools, permissions, context_manager
                )
                for tu in concurrent
            ])
            for tu, result in zip(concurrent, results):
                yield ToolResultEvent(tool_name=tu["name"], result=result)
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": result.content if result.success else f"Error: {result.error}",
                        "is_error": not result.success,
                    }],
                })

        # Run sequential tools one at a time
        for tu in sequential:
            result = await execute_tool_with_permissions(
                tu, tools, permissions, context_manager
            )
            yield ToolResultEvent(tool_name=tu["name"], result=result)
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result.content if result.success else f"Error: {result.error}",
                    "is_error": not result.success,
                }],
            })

    else:
        yield MaxTurnsEvent(max_turns=max_turns)


async def execute_tool_with_permissions(
    tool_use: dict,
    tools: ToolRegistry,
    permissions: PermissionSystem,
    context_manager: ContextManager,
) -> ToolResult:
    tool = tools.get(tool_use["name"])
    if not tool:
        return ToolResult(success=False, error=f"Unknown tool: {tool_use['name']}")

    # Validate input
    validation = tool.validate_input(tool_use["input"])
    if not validation.valid:
        return ToolResult(success=False, error=validation.error)

    # Check permissions
    decision = permissions.check(tool.name, tool_use["input"])
    if decision == "deny":
        return ToolResult(success=False, error="Permission denied")

    # Execute
    try:
        result = await tool.call(tool_use["input"], context={})

        # Persist large results
        if len(str(result.content)) > context_manager.max_inline_size:
            path = context_manager.persist_result(result.content)
            result.content = f"[Result too large. Full output saved to: {path}]"

        return result
    except Exception as e:
        return ToolResult(success=False, error=str(e))
```

## Event Types

```python
from dataclasses import dataclass

@dataclass
class Event:
    type: str

@dataclass
class StreamingTextEvent(Event):
    type: str = "streaming_text"
    text: str = ""
    start: bool = False

@dataclass
class ToolStartEvent(Event):
    type: str = "tool_start"
    tool_name: str = ""
    tool_id: str = ""

@dataclass
class ToolInputEvent(Event):
    type: str = "tool_input"
    partial_json: str = ""

@dataclass
class ThinkingEvent(Event):
    type: str = "thinking"
    thinking: str = ""

@dataclass
class ToolResultEvent(Event):
    type: str = "tool_result"
    tool_name: str = ""
    result: "ToolResult" = None

@dataclass
class CostEvent(Event):
    type: str = "cost"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_creation: int = 0
    total_cost_usd: float = 0.0

@dataclass
class DoneEvent(Event):
    type: str = "done"
    stop_reason: str = ""
    turn_count: int = 0

@dataclass
class ErrorEvent(Event):
    type: str = "error"
    message: str = ""

@dataclass
class RetryEvent(Event):
    type: str = "retry"
    error: str = ""
    attempt: int = 0

@dataclass
class MaxTurnsEvent(Event):
    type: str = "max_turns"
    max_turns: int = 0
```

## Sub-Agent Pattern

Spawn child agents for parallel work:

```python
async def spawn_sub_agent(
    prompt: str,
    tools: ToolRegistry,
    api_client: APIClient,
    system_prompt: list[str] | None = None,
) -> str:
    sub_system = system_prompt or [
        "You are a sub-agent. Complete the task and return results.",
    ]
    messages = [{"role": "user", "content": prompt}]
    result_text = []

    async for event in agent_loop(
        messages=messages,
        tools=tools,
        permissions=PermissionSystem.auto_allow(),
        context_manager=ContextManager(),
        system_prompt=sub_system,
        api_client=api_client,
        max_turns=20,
    ):
        if isinstance(event, StreamingTextEvent) and event.text:
            result_text.append(event.text)
        if isinstance(event, DoneEvent):
            break

    return "".join(result_text)
```
