# The Agent Loop — TypeScript/Bun

The complete streaming agent loop in idiomatic TypeScript/Bun: typed discriminated-union events,
concurrent tool execution with `Promise.all`, exponential-backoff retry, prompt-too-long
auto-compact, and full prompt-cache cost tracking.

This is the TypeScript equivalent of `references/agent-loop.md`.

---

## Event Type Discriminated Union

No class inheritance. Every consumer gets exhaustive-check safety via the `type` discriminant.

```typescript
// events.ts
export type AgentEvent =
  | StreamingTextEvent
  | ToolStartEvent
  | ToolInputEvent
  | ThinkingEvent
  | ToolResultEvent
  | CostEvent
  | DoneEvent
  | ErrorEvent
  | RetryEvent
  | MaxTurnsEvent;

export interface StreamingTextEvent {
  type: "streaming_text";
  text: string;
  start: boolean;
}

export interface ToolStartEvent {
  type: "tool_start";
  toolName: string;
  toolId: string;
}

export interface ToolInputEvent {
  type: "tool_input";
  partialJson: string;
}

export interface ThinkingEvent {
  type: "thinking";
  thinking: string;
}

export interface ToolResultEvent {
  type: "tool_result";
  toolName: string;
  result: ToolResult;
}

export interface CostEvent {
  type: "cost";
  inputTokens: number;
  outputTokens: number;
  cacheRead: number;
  cacheCreation: number;
  totalCostUsd: number;
}

export interface DoneEvent {
  type: "done";
  stopReason: string;
  turnCount: number;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export interface RetryEvent {
  type: "retry";
  error: string;
  attempt: number;
}

export interface MaxTurnsEvent {
  type: "max_turns";
  maxTurns: number;
}
```

---

## Tool Interface and Registry

```typescript
// tools.ts
import type Anthropic from "@anthropic-ai/sdk";

export interface ToolResult {
  success: boolean;
  content: string;
  error?: string;
  metadata?: Record<string, unknown>;
}

export interface Tool {
  name: string;
  description: string;
  /** JSON Schema properties object passed directly to the Anthropic API. */
  inputSchema: Record<string, unknown>;
  isReadOnly: boolean;
  isConcurrencySafe: boolean;
  call(input: Record<string, unknown>): Promise<ToolResult>;
  /** Return "allow" to auto-approve, "deny" to block, "ask" to prompt user. */
  checkPermissions(input: Record<string, unknown>): "allow" | "deny" | "ask";
}

export class ToolRegistry {
  private tools = new Map<string, Tool>();

  register(tool: Tool): void {
    this.tools.set(tool.name, tool);
  }

  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }

  all(): Tool[] {
    return [...this.tools.values()];
  }

  /** Format the registry for the Anthropic messages API `tools` parameter. */
  toApiSchema(): Anthropic.Tool[] {
    return this.all().map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: {
        type: "object" as const,
        ...tool.inputSchema,
      },
    }));
  }

  filterByDenyRules(denyList: string[]): ToolRegistry {
    const filtered = new ToolRegistry();
    for (const tool of this.tools.values()) {
      if (!denyList.includes(tool.name)) {
        filtered.register(tool);
      }
    }
    return filtered;
  }
}
```

---

## Built-in Tool Implementations

### BashTool — `Bun.spawn`

```typescript
// tools/bash.ts
import type { Tool, ToolResult } from "../tools.js";

const READ_ONLY_PREFIXES = [
  "ls", "cat", "grep", "find", "git status",
  "git diff", "git log", "head", "tail", "wc",
  "which", "echo", "pwd", "type",
];

export class BashTool implements Tool {
  name = "bash";
  description = "Execute a shell command and return stdout, stderr, and exit code.";
  isReadOnly = false;
  isConcurrencySafe = false;

  inputSchema = {
    properties: {
      command: {
        type: "string",
        description: "The shell command to execute",
      },
      timeout: {
        type: "number",
        description: "Timeout in milliseconds (default 120000)",
      },
      description: {
        type: "string",
        description: "Brief description of what this command does",
      },
    },
    required: ["command"],
  };

  checkPermissions(input: Record<string, unknown>): "allow" | "deny" | "ask" {
    const command = String(input.command ?? "");
    const first = command.split(/&&|\||;/)[0].trim();
    if (READ_ONLY_PREFIXES.some((p) => first.startsWith(p))) {
      // Promote to safe for this call — read-only commands can run concurrently
      this.isReadOnly = true;
      this.isConcurrencySafe = true;
      return "allow";
    }
    return "ask";
  }

  async call(input: Record<string, unknown>): Promise<ToolResult> {
    const command = String(input.command);
    const timeoutMs = typeof input.timeout === "number" ? input.timeout : 120_000;

    const proc = Bun.spawn(["sh", "-c", command], {
      stdout: "pipe",
      stderr: "pipe",
    });

    const timer = setTimeout(() => proc.kill(), timeoutMs);

    try {
      const [stdout, stderr, exitCode] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
      ]);
      clearTimeout(timer);

      if (exitCode !== 0) {
        return {
          success: false,
          content: stdout,
          error: `Exit code ${exitCode}: ${stderr}`,
        };
      }
      return { success: true, content: stdout + (stderr ? `\n${stderr}` : "") };
    } catch (err) {
      clearTimeout(timer);
      return { success: false, content: "", error: `Spawn error: ${String(err)}` };
    }
  }
}
```

### FileReadTool — `Bun.file()`

```typescript
// tools/file-read.ts
import type { Tool, ToolResult } from "../tools.js";

export class FileReadTool implements Tool {
  name = "read";
  description = "Read file contents with optional line ranges.";
  isReadOnly = true;
  isConcurrencySafe = true;

  inputSchema = {
    properties: {
      file_path: { type: "string", description: "Absolute path to the file" },
      offset: { type: "number", description: "First line to read (1-indexed)" },
      limit: { type: "number", description: "Maximum lines to return" },
    },
    required: ["file_path"],
  };

  checkPermissions(_input: Record<string, unknown>): "allow" {
    return "allow";
  }

  async call(input: Record<string, unknown>): Promise<ToolResult> {
    const filePath = String(input.file_path);
    const file = Bun.file(filePath);

    const exists = await file.exists();
    if (!exists) {
      return { success: false, content: "", error: `File not found: ${filePath}` };
    }

    try {
      const text = await file.text();
      const lines = text.split("\n");
      const offset = typeof input.offset === "number" ? input.offset - 1 : 0;
      const limit = typeof input.limit === "number" ? input.limit : lines.length;
      const selected = lines.slice(offset, offset + limit);

      const numbered = selected
        .map((line, i) => `${offset + i + 1}\t${line}`)
        .join("\n");

      return { success: true, content: numbered };
    } catch (err) {
      return { success: false, content: "", error: String(err) };
    }
  }
}
```

### FileEditTool — exact-string replacement

```typescript
// tools/file-edit.ts
import type { Tool, ToolResult } from "../tools.js";

export class FileEditTool implements Tool {
  name = "edit";
  description = "Replace exact text in a file. old_string must match precisely.";
  isReadOnly = false;
  isConcurrencySafe = false;

  inputSchema = {
    properties: {
      file_path: { type: "string", description: "Absolute path to the file" },
      old_string: { type: "string", description: "Exact text to replace" },
      new_string: { type: "string", description: "Replacement text" },
      replace_all: { type: "boolean", description: "Replace all occurrences (default false)" },
    },
    required: ["file_path", "old_string", "new_string"],
  };

  checkPermissions(_input: Record<string, unknown>): "ask" {
    return "ask";
  }

  async call(input: Record<string, unknown>): Promise<ToolResult> {
    const filePath = String(input.file_path);
    const oldStr = String(input.old_string);
    const newStr = String(input.new_string);
    const replaceAll = Boolean(input.replace_all);

    if (oldStr === newStr) {
      return { success: false, content: "", error: "old_string and new_string are identical" };
    }

    const file = Bun.file(filePath);
    if (!(await file.exists())) {
      return { success: false, content: "", error: `File not found: ${filePath}` };
    }

    const content = await file.text();
    // Count occurrences without regex to avoid escaping issues
    let count = 0;
    let pos = 0;
    while ((pos = content.indexOf(oldStr, pos)) !== -1) {
      count++;
      pos += oldStr.length;
    }

    if (count === 0) {
      return { success: false, content: "", error: "old_string not found in file" };
    }
    if (count > 1 && !replaceAll) {
      return {
        success: false,
        content: "",
        error: `Found ${count} matches. Set replace_all: true or add more context to old_string.`,
      };
    }

    const newContent = replaceAll
      ? content.split(oldStr).join(newStr)
      : content.replace(oldStr, newStr); // first occurrence only

    await Bun.write(filePath, newContent);
    return { success: true, content: `Replaced ${count} occurrence(s) in ${filePath}` };
  }
}
```

---

## `executeToolWithPermissions` Helper

```typescript
// execute-tool.ts
import type { Tool, ToolRegistry, ToolResult } from "./tools.js";

const MAX_INLINE_SIZE = 50_000; // chars before we truncate

export async function executeToolWithPermissions(
  toolUse: { id: string; name: string; input: Record<string, unknown> },
  registry: ToolRegistry,
): Promise<ToolResult> {
  const tool = registry.get(toolUse.name);
  if (!tool) {
    return { success: false, content: "", error: `Unknown tool: ${toolUse.name}` };
  }

  const decision = tool.checkPermissions(toolUse.input);
  if (decision === "deny") {
    return { success: false, content: "", error: "Permission denied" };
  }
  // "ask" in a real agent would pause and prompt the user.
  // For scripted/non-interactive use, treat "ask" as "allow".

  try {
    const result = await tool.call(toolUse.input);

    // Truncate oversized results to keep the context window healthy
    if (result.content.length > MAX_INLINE_SIZE) {
      return {
        ...result,
        content: result.content.slice(0, MAX_INLINE_SIZE) +
          `\n... [truncated — ${result.content.length} chars total]`,
      };
    }

    return result;
  } catch (err) {
    return { success: false, content: "", error: String(err) };
  }
}
```

---

## The Agent Loop

```typescript
// agent-loop.ts
import Anthropic from "@anthropic-ai/sdk";
import type { MessageParam } from "@anthropic-ai/sdk/resources/messages.js";
import type { AgentEvent } from "./events.js";
import type { ToolRegistry } from "./tools.js";
import { executeToolWithPermissions } from "./execute-tool.js";

// Pricing as of claude-sonnet-4-6 (USD per million tokens)
const PRICE = {
  inputPerM: 3.0,
  outputPerM: 15.0,
  cacheReadPerM: 0.30,
  cacheWritePerM: 3.75,
};

function computeCost(usage: {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}): number {
  const input = ((usage.input_tokens ?? 0) / 1_000_000) * PRICE.inputPerM;
  const output = ((usage.output_tokens ?? 0) / 1_000_000) * PRICE.outputPerM;
  const cacheRead = ((usage.cache_read_input_tokens ?? 0) / 1_000_000) * PRICE.cacheReadPerM;
  const cacheWrite = ((usage.cache_creation_input_tokens ?? 0) / 1_000_000) * PRICE.cacheWritePerM;
  return input + output + cacheRead + cacheWrite;
}

export async function* agentLoop(
  messages: MessageParam[],
  tools: ToolRegistry,
  client: Anthropic,
  options: {
    systemPrompt?: string;
    model?: string;
    maxTurns?: number;
  } = {},
): AsyncGenerator<AgentEvent> {
  const model = options.model ?? "claude-sonnet-4-6";
  const maxTurns = options.maxTurns ?? 50;
  const systemPrompt = options.systemPrompt ?? "You are a helpful AI coding agent.";

  let turn = 0;
  let consecutiveErrors = 0;
  let totalCostUsd = 0;

  while (turn < maxTurns) {
    turn++;

    // ── Call the model with streaming ──────────────────────────────────────
    let stopReason = "end_turn";

    // Accumulated content for the assistant turn
    const textParts: string[] = [];
    const thinkingParts: string[] = [];
    // tool_use blocks as returned by the SDK (already typed objects)
    const toolUseBlocks: Array<{
      id: string;
      name: string;
      input: Record<string, unknown>;
      type: "tool_use";
    }> = [];

    // Track partial JSON per tool block index
    const partialInputs: Record<number, string> = {};
    let activeBlockIndex = -1;
    let usage: {
      input_tokens: number;
      output_tokens: number;
      cache_read_input_tokens?: number;
      cache_creation_input_tokens?: number;
    } | null = null;

    try {
      const stream = client.messages.stream({
        model,
        max_tokens: 16_384,
        system: systemPrompt,
        messages,
        tools: tools.toApiSchema(),
      });

      for await (const event of stream) {
        switch (event.type) {
          case "message_start":
            usage = event.message.usage as typeof usage;
            break;

          case "content_block_start":
            activeBlockIndex = event.index;
            if (event.content_block.type === "text") {
              yield { type: "streaming_text", text: "", start: true };
            } else if (event.content_block.type === "tool_use") {
              yield {
                type: "tool_start",
                toolName: event.content_block.name,
                toolId: event.content_block.id,
              };
              // Pre-allocate slot so index-based lookup works
              partialInputs[activeBlockIndex] = "";
            } else if (event.content_block.type === "thinking") {
              // Extended thinking block — no separate start event needed
            }
            break;

          case "content_block_delta":
            if (event.delta.type === "text_delta") {
              textParts.push(event.delta.text);
              yield { type: "streaming_text", text: event.delta.text, start: false };
            } else if (event.delta.type === "input_json_delta") {
              partialInputs[event.index] = (partialInputs[event.index] ?? "") + event.delta.partial_json;
              yield { type: "tool_input", partialJson: event.delta.partial_json };
            } else if (event.delta.type === "thinking_delta") {
              thinkingParts.push(event.delta.thinking);
              yield { type: "thinking", thinking: event.delta.thinking };
            }
            break;

          case "content_block_stop":
            // If this was a tool_use block, finalise it
            if (event.index in partialInputs) {
              // The stream helper on the SDK has already parsed the full input
              // by the time we call stream.finalMessage(), but we collect here
              // for immediate availability.
            }
            break;

          case "message_delta":
            if (event.delta.stop_reason) {
              stopReason = event.delta.stop_reason;
            }
            if (event.usage) {
              // message_delta usage supplements message_start usage
              usage = { ...(usage ?? {}), ...event.usage } as typeof usage;
            }
            break;
        }
      }

      // The SDK's stream helper provides the fully-parsed final message.
      // Pull tool_use blocks from it so we have typed, parsed inputs.
      const finalMsg = await stream.finalMessage();
      for (const block of finalMsg.content) {
        if (block.type === "tool_use") {
          toolUseBlocks.push({
            type: "tool_use",
            id: block.id,
            name: block.name,
            input: block.input as Record<string, unknown>,
          });
        }
      }

      consecutiveErrors = 0;

    } catch (err: unknown) {
      // Detect prompt_too_long (context overflow)
      const isOverflow =
        err instanceof Anthropic.APIError &&
        (err.status === 400 ||
          (typeof (err as { error?: { type?: string } }).error?.type === "string" &&
            (err as { error: { type: string } }).error.type === "invalid_request_error")) &&
        String(err.message).toLowerCase().includes("prompt is too long");

      if (isOverflow) {
        // Drop oldest non-system messages to free space, then retry
        const trimmed = messages.slice(Math.floor(messages.length / 2));
        messages = trimmed;
        yield { type: "retry", error: "Prompt too long — compacted message history", attempt: turn };
        turn--; // don't count this as a real turn
        continue;
      }

      consecutiveErrors++;
      if (consecutiveErrors >= 3) {
        yield { type: "error", message: `API error after 3 retries: ${String(err)}` };
        return;
      }
      yield { type: "retry", error: String(err), attempt: consecutiveErrors };
      // Exponential backoff: 2s, 4s, 8s
      await new Promise((r) => setTimeout(r, 2 ** consecutiveErrors * 1_000));
      continue;
    }

    // ── Track costs ────────────────────────────────────────────────────────
    if (usage) {
      const turnCost = computeCost(usage);
      totalCostUsd += turnCost;
      yield {
        type: "cost",
        inputTokens: usage.input_tokens,
        outputTokens: usage.output_tokens,
        cacheRead: usage.cache_read_input_tokens ?? 0,
        cacheCreation: usage.cache_creation_input_tokens ?? 0,
        totalCostUsd,
      };
    }

    // ── Build assistant message ────────────────────────────────────────────
    const assistantContent: Anthropic.ContentBlock[] = [];

    if (thinkingParts.length > 0) {
      // Thinking blocks must come from the finalMessage — include them as-is
      const finalMsg2 = await client.messages.create({
        model,
        max_tokens: 1,
        system: "skip",
        messages: [{ role: "user", content: "skip" }],
      }).catch(() => null);
      // Fallback: reconstruct thinking from streamed deltas
      assistantContent.push({
        type: "thinking",
        thinking: thinkingParts.join(""),
      } as unknown as Anthropic.ContentBlock);
    }

    if (textParts.length > 0) {
      assistantContent.push({ type: "text", text: textParts.join("") });
    }

    for (const tb of toolUseBlocks) {
      assistantContent.push(tb as unknown as Anthropic.ContentBlock);
    }

    messages = [
      ...messages,
      { role: "assistant", content: assistantContent },
    ];

    // ── No tools = done ────────────────────────────────────────────────────
    if (toolUseBlocks.length === 0) {
      yield { type: "done", stopReason, turnCount: turn };
      return;
    }

    // ── Execute tools ──────────────────────────────────────────────────────
    // Split into concurrency-safe (run in parallel) vs sequential
    const concurrent = toolUseBlocks.filter(
      (tu) => tools.get(tu.name)?.isConcurrencySafe === true,
    );
    const sequential = toolUseBlocks.filter(
      (tu) => tools.get(tu.name)?.isConcurrencySafe !== true,
    );

    const toolResultBlocks: Anthropic.ToolResultBlockParam[] = [];

    // Parallel execution for concurrency-safe tools
    if (concurrent.length > 0) {
      const results = await Promise.all(
        concurrent.map((tu) => executeToolWithPermissions(tu, tools)),
      );
      for (let i = 0; i < concurrent.length; i++) {
        const tu = concurrent[i];
        const result = results[i];
        yield { type: "tool_result", toolName: tu.name, result };
        toolResultBlocks.push({
          type: "tool_result",
          tool_use_id: tu.id,
          content: result.success ? result.content : `Error: ${result.error ?? "unknown"}`,
          is_error: !result.success,
        });
      }
    }

    // Sequential execution for non-safe tools (bash writes, edits, etc.)
    for (const tu of sequential) {
      const result = await executeToolWithPermissions(tu, tools);
      yield { type: "tool_result", toolName: tu.name, result };
      toolResultBlocks.push({
        type: "tool_result",
        tool_use_id: tu.id,
        content: result.success ? result.content : `Error: ${result.error ?? "unknown"}`,
        is_error: !result.success,
      });
    }

    // Append tool results as a single user turn
    messages = [
      ...messages,
      { role: "user", content: toolResultBlocks },
    ];
  }

  // Fell out of the while loop — hit maxTurns
  yield { type: "max_turns", maxTurns };
}
```

---

## `spawnSubAgent` Helper

Runs a child agent loop to completion and returns its final text output.
Useful for parallelising independent sub-tasks from a parent orchestrator.

```typescript
// spawn-sub-agent.ts
import Anthropic from "@anthropic-ai/sdk";
import type { ToolRegistry } from "./tools.js";
import { agentLoop } from "./agent-loop.js";

export async function spawnSubAgent(
  prompt: string,
  tools: ToolRegistry,
  client: Anthropic,
  systemPrompt?: string,
): Promise<string> {
  const system =
    systemPrompt ?? "You are a sub-agent. Complete the assigned task and return your results.";

  const textParts: string[] = [];

  for await (const event of agentLoop(
    [{ role: "user", content: prompt }],
    tools,
    client,
    { systemPrompt: system, maxTurns: 20 },
  )) {
    if (event.type === "streaming_text" && event.text) {
      textParts.push(event.text);
    }
    if (event.type === "done") break;
    if (event.type === "error") {
      throw new Error(`Sub-agent error: ${event.message}`);
    }
  }

  return textParts.join("");
}
```

---

## Usage Example

```typescript
// main.ts  —  bun run main.ts
import Anthropic from "@anthropic-ai/sdk";
import { agentLoop } from "./agent-loop.js";
import { ToolRegistry } from "./tools.js";
import { BashTool } from "./tools/bash.js";
import { FileReadTool } from "./tools/file-read.js";
import { FileEditTool } from "./tools/file-edit.js";
import type { AgentEvent } from "./events.js";

const client = new Anthropic(); // reads ANTHROPIC_API_KEY from env

const registry = new ToolRegistry();
registry.register(new BashTool());
registry.register(new FileReadTool());
registry.register(new FileEditTool());

const messages = [
  {
    role: "user" as const,
    content: "List the files in the current directory and tell me which are TypeScript files.",
  },
];

for await (const event of agentLoop(messages, registry, client)) {
  switch (event.type) {
    case "streaming_text":
      if (event.start) process.stdout.write("\n");
      process.stdout.write(event.text);
      break;
    case "tool_start":
      console.log(`\n[Tool] ${event.toolName} (${event.toolId})`);
      break;
    case "tool_result":
      console.log(`[Result] ${event.result.success ? "OK" : "ERR"}: ${event.result.content.slice(0, 120)}`);
      break;
    case "cost":
      console.log(
        `[Cost] $${event.totalCostUsd.toFixed(6)} | ` +
        `in=${event.inputTokens} out=${event.outputTokens} ` +
        `cache_read=${event.cacheRead} cache_write=${event.cacheCreation}`,
      );
      break;
    case "done":
      console.log(`\n[Done] stop_reason=${event.stopReason} turns=${event.turnCount}`);
      break;
    case "error":
      console.error(`[Error] ${event.message}`);
      break;
    case "retry":
      console.warn(`[Retry] attempt ${event.attempt}: ${event.error}`);
      break;
    case "max_turns":
      console.warn(`[MaxTurns] reached ${event.maxTurns}`);
      break;
  }
}
```

Run with:

```bash
ANTHROPIC_API_KEY=sk-... bun run main.ts
```

---

## Key Differences from the Python Version

| Concern | Python | TypeScript/Bun |
|---|---|---|
| Concurrency | `asyncio.gather()` | `Promise.all()` |
| Process spawn | `asyncio.create_subprocess_shell` | `Bun.spawn(["sh", "-c", cmd])` |
| File I/O | `Path.read_text()` / `Path.write_text()` | `Bun.file().text()` / `Bun.write()` |
| Event union | `@dataclass` hierarchy | Discriminated union of `interface` types |
| Type safety | `isinstance()` checks | TypeScript `switch (event.type)` narrows automatically |
| Streaming iteration | `async for event in stream:` | `for await (const event of stream)` |
| Error type check | `isinstance(e, APIError)` | `err instanceof Anthropic.APIError` |
| Tool input parsing | Collected incrementally, parsed at end | `stream.finalMessage()` provides parsed `.input` |
| Retry sleep | `asyncio.sleep(2 ** n)` | `setTimeout` wrapped in `Promise` |
| Cost computation | External `CostTracker` class | Inline `computeCost()` pure function |

---

## Project Layout

```
src/
  events.ts           ← AgentEvent discriminated union
  tools.ts            ← Tool interface + ToolRegistry
  execute-tool.ts     ← executeToolWithPermissions()
  agent-loop.ts       ← agentLoop() async generator
  spawn-sub-agent.ts  ← spawnSubAgent()
  tools/
    bash.ts
    file-read.ts
    file-edit.ts
main.ts               ← entry point
```

Install the SDK:

```bash
bun add @anthropic-ai/sdk
```

`package.json` type field must be `"module"` (ESM). Bun handles TypeScript natively — no build step required.
