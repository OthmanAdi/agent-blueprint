# Architecture Overview

The complete architecture of a production AI coding agent, reverse-engineered from Claude Code.

## System Diagram

```
┌─────────────────────────────────────────────────┐
│                   Entry Point                    │
│  CLI arg parse → Auth → Init → Launch REPL      │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Query Engine (Session)               │
│  Owns: messages, abort, usage, file cache        │
│  submitMessage() → yields SDKMessage stream      │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Agent Loop (while true)              │
│                                                   │
│  ┌─────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Context  │→│ LLM API  │→│ Tool Execution │  │
│  │ Manage   │  │ Stream   │  │ Pipeline       │  │
│  └─────────┘  └──────────┘  └────────────────┘  │
│       ↑                           │               │
│       └───── tool results ────────┘               │
│                                                   │
│  Exit when: no tool_use in response               │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Tool Execution Pipeline              │
│                                                   │
│  Schema Validate → Hooks → Permission → Execute  │
│  → Post Hooks → Persist Large Results             │
└──────────────────────────────────────────────────┘
```

## Core Modules

| Module | Responsibility | Key Pattern |
|--------|---------------|-------------|
| Agent Loop | Drive the conversation turns | Async generator yielding events |
| Tool Registry | Define and dispatch tools | Strategy pattern with Zod/Pydantic schemas |
| Permission System | Allow/deny/ask per tool call | Defense-in-depth with 7 layers |
| Context Manager | Keep conversations in token limits | Compact, snip, budget |
| System Prompt | Assemble dynamic prompts | String array with cache boundary |
| API Client | Stream from LLM provider | Server-sent events with tool_use |
| Cost Tracker | Track tokens and USD cost | Per-turn accumulation |

## Dependency Graph

```
main.py
  ├── cli.py (argument parsing)
  ├── auth.py (API key management)
  ├── repl.py (interactive UI)
  │     └── query_engine.py (session orchestrator)
  │           ├── agent_loop.py (while-true loop)
  │           │     ├── api_client.py (LLM streaming)
  │           │     ├── context.py (message management)
  │           │     │     └── compact.py (summarization)
  │           │     ├── tool_executor.py (dispatch + hooks)
  │           │     │     ├── tools.py (registry)
  │           │     │     └── permissions.py (allow/deny)
  │           │     └── cost_tracker.py (usage)
  │           └── system_prompt.py (prompt assembly)
  └── headless.py (non-interactive mode)
```

## Key Design Decisions

### 1. Async Generator Pattern
The agent loop is an `async generator` that yields events as they happen. This enables:
- Real-time streaming to any UI (terminal, web, API)
- Cancellation via abort signals
- Clean composition with sub-agents

### 2. Tool as Data
Every tool is a data structure (name, schema, handler), not a class hierarchy. This enables:
- Dynamic tool registration (add tools at runtime)
- MCP server integration (external tools look the same)
- Easy testing (mock any tool)

### 3. Prompt Cache Optimization
The system prompt is split into:
- **Static prefix** (never changes → gets cached by API → 10x cheaper)
- **Dynamic boundary marker** (signals cache break)
- **Dynamic suffix** (changes per turn → not cached)

### 4. Streaming Tool Execution
When the LLM streams multiple `tool_use` blocks:
- Concurrency-safe tools execute in parallel
- Non-safe tools wait for exclusivity
- Results yielded in original order

### 5. Defense-in-Depth Permissions
Seven independent layers check every tool call:
1. Hardcoded deny rules
2. Tool-specific permission logic
3. User-configured allow rules
4. Safety checks for sensitive paths
5. Interactive user prompts
6. Pre-tool hooks (can block/modify)
7. Post-tool hooks (can modify output)

## Technology Stack Choices

### For Python Agents
| Component | Recommended Library |
|-----------|-------------------|
| LLM Client | `anthropic` SDK (streaming) |
| Schema Validation | `pydantic` v2 |
| Terminal UI | `rich` or `textual` |
| Async Runtime | `asyncio` + `aiohttp` |
| CLI Framework | `click` or `typer` |
| Testing | `pytest` + `pytest-asyncio` |

### For TypeScript Agents
| Component | Recommended Library |
|-----------|-------------------|
| LLM Client | `@anthropic-ai/sdk` |
| Schema Validation | `zod` v4 |
| Terminal UI | `ink` (React for terminals) |
| Runtime | `bun` or `node` |
| CLI Framework | `commander` |
| Testing | `vitest` |

### For Other Languages
The patterns are language-agnostic. Implement the same interfaces using:
- Rust: `tokio` + `serde` + `clap`
- Go: `goroutines` + `struct tags` + `cobra`
- Java: `Virtual Threads` + `Jackson` + `picocli`
