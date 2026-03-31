---
name: agent-blueprint
description: Build production-grade AI agents from scratch using patterns inspired by production-grade agentic systems like Claude Code. Covers two audiences — (1) workflow builders who need planner+executor pipelines, human-in-the-loop checkpoints, and multi-agent orchestration, and (2) application embedders who need HTTP/SSE APIs, multi-user session management, and React frontend integration. Use when asked to build an AI agent, create a coding assistant, embed an agent in an app, build an API agent, clone Claude Code, build a CLI agent, design an agent architecture, scaffold an agentic system, or automate a multi-step workflow. Triggers on keywords like agent builder, agent blueprint, coding assistant, CLI agent, agentic tool, claude code clone, agent scaffold, agent architecture, workflow automation, multi-agent, planner executor, SSE streaming, agent API.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: agent-blueprint-community
---

# Agent Blueprint - Build Production-Grade AI Agents

Generate complete, working AI agents using battle-tested patterns studied from production-grade agentic systems. Every pattern is independently implemented and adapted for any framework, language, or purpose.

## When to Use

**Use this skill when:**
- Building any kind of AI agent or agentic tool
- Creating a coding assistant, research agent, or task automation agent
- Designing an agent architecture with tools, permissions, and context management
- Scaffolding a new agent project from scratch
- Cloning or recreating Claude Code-like functionality
- Building multi-step automated workflows with planner + executor architecture
- Embedding an agent inside a web application via HTTP/SSE API
- Adding human-in-the-loop approval checkpoints to an autonomous agent
- Making an agent production-ready (observability, cost control, long tasks)

**Do NOT use when:**
- Building a simple chatbot without tools (use prompt engineering instead)
- Creating a non-agentic API wrapper (use your SDK directly)

## The 5-Phase Build Process

### Phase 1: Define Your Agent

Before writing any code, answer these 4 questions. Write answers to a `AGENT.md` spec file in your project:

```
1. PURPOSE: What does this agent do in one sentence?
2. TOOLS: What actions can it take? (file ops, shell, web, APIs, custom)
3. TRIGGERS: When should it act autonomously vs wait for input?
4. OUTPUT: What does "done" look like? (code changes, reports, files, etc.)
```

Based on your answers, choose a template:

| Template | Best For | Complexity |
|----------|----------|------------|
| [minimal-agent](./templates/minimal-agent/) | Single-purpose agents, scripts, tools | Low (~200 lines, Python) |
| [coding-agent](./templates/coding-agent/) | Full coding assistants with file ops | Medium (~800 lines, Python) |
| [research-agent](./templates/research-agent/) | Research, analysis, web search | Medium (~600 lines, Python) |
| [task-agent](./templates/task-agent/) | Multi-step automation, orchestration | High (~1200 lines, Python) |

For TypeScript/Bun: use [references/typescript-agent-loop.md](./references/typescript-agent-loop.md) as your starting point and [examples/typescript-research-agent.md](./examples/typescript-research-agent.md) as a runnable example.
For Rust: use [references/rust-agent.md](./references/rust-agent.md).

**Not sure which template? Use this quick selector:**

| Your goal | Template | Key references to load |
|-----------|----------|------------------------|
| CLI tool that does one thing | minimal-agent | agent-loop.md |
| Coding assistant in terminal | coding-agent | tool-system.md, permission-system.md |
| Automated multi-step workflow | workflow-agent | planner-executor.md, human-in-the-loop.md |
| Agent inside a web/mobile app | api-agent | serving.md, long-tasks.md |
| Research + report generation | research-agent | context-management.md, memory.md |
| 100+ step autonomous task | task-agent | long-tasks.md, production-principles.md |

### Phase 2: Implement Core Components

Build these 5 components in order. Each is a separate module:

```
agent/
  core/
    agent_loop.py      # The main while-true loop (Phase 2.1)
    tool_registry.py   # Tool definitions and dispatch (Phase 2.2)
    permissions.py     # Allow/deny/ask permission system (Phase 2.3)
    context.py         # Message history and compaction (Phase 2.4)
    system_prompt.py   # Dynamic prompt assembly (Phase 2.5)
```

#### Phase 2.1: The Agent Loop

The heart of every agent. This is the `while(true)` pattern that drives all agentic behavior:

```python
async def agent_loop(messages, tools, system_prompt, api_client):
    while True:
        # 1. Manage context window
        messages = compact_if_needed(messages)

        # 2. Call the LLM with streaming
        response = await api_client.stream(
            system=system_prompt,
            messages=messages,
            tools=tools.to_api_schema(),
        )

        # 3. Process response blocks
        assistant_content = []
        for block in response:
            if block.type == "text":
                assistant_content.append(block)
            elif block.type == "tool_use":
                # 4. Check permissions
                decision = permissions.check(block.name, block.input)
                if decision == "deny":
                    assistant_content.append(tool_error(block.id, "Permission denied"))
                    continue

                # 5. Execute tool
                result = await tools.execute(block.name, block.input)
                messages.append(tool_result(block.id, result))
                assistant_content.append(block)

        messages.append(assistant_message(assistant_content))

        # 6. Continue if tools were used, stop if text-only
        if not any(b.type == "tool_use" for b in assistant_content):
            break

    return messages
```

**Critical patterns:**
- Stream responses for real-time feedback
- Execute concurrency-safe tools in parallel
- Handle `prompt_too_long` errors by compacting context
- Track token usage and cost per turn

Load [references/agent-loop.md](./references/agent-loop.md) for the complete loop with error recovery, streaming, and cost tracking.

#### Phase 2.2: The Tool System

Every tool implements this interface:

```python
class Tool:
    name: str                    # Unique identifier (e.g., "bash", "read_file")
    description: str             # What it does (shown to LLM)
    input_schema: dict           # JSON Schema for parameters
    is_read_only: bool           # True if no side effects
    is_concurrency_safe: bool    # True if parallel execution is safe

    async def call(self, input: dict, context: dict) -> ToolResult:
        """Execute the tool and return result."""

    def check_permissions(self, input: dict) -> PermissionDecision:
        """Return allow/deny/ask for this specific invocation."""

    def validate_input(self, input: dict) -> ValidationResult:
        """Validate parameters before execution."""
```

**Built-in tools to implement first:**

| Tool | Purpose | Priority |
|------|---------|----------|
| `bash` | Execute shell commands | Must-have |
| `read_file` | Read file contents | Must-have |
| `write_file` | Create/overwrite files | Must-have |
| `edit_file` | Find-and-replace in files | Must-have |
| `glob` | Find files by pattern | Must-have |
| `grep` | Search file contents | Must-have |
| `web_search` | Search the internet | Nice-to-have |
| `web_fetch` | Fetch URL content | Nice-to-have |
| `ask_user` | Ask user a question | Nice-to-have |

Load [references/tool-system.md](./references/tool-system.md) for complete tool implementations with Zod/Pydantic schemas.

#### Phase 2.3: The Permission System

Defense-in-depth permission checking:

```python
def check_permission(tool_name, tool_input, context):
    # Layer 1: Always-deny rules (hardcoded safety)
    if matches_deny_rule(tool_name, tool_input):
        return DENY

    # Layer 2: Tool-specific checks
    tool_check = tools[tool_name].check_permissions(tool_input)
    if tool_check in (ALLOW, DENY):
        return tool_check

    # Layer 3: Always-allow rules (user-configured patterns)
    if matches_allow_rule(tool_name, tool_input):
        return ALLOW

    # Layer 4: Safety checks for sensitive paths
    if is_sensitive_path(tool_input):
        return ASK

    # Layer 5: Ask the user
    return ASK
```

**Permission rule format:** `"ToolName(pattern)"` — e.g., `"Bash(git *)"`, `"Read"`, `"Edit(*.ts)"`

Load [references/permission-system.md](./references/permission-system.md) for the complete 7-layer permission pipeline.

#### Phase 2.4: Context Management

The #1 challenge in production agents. Implement these 3 strategies:

**Strategy 1: Auto-Compact** — When approaching context limits, summarize old messages:
```python
if token_count(messages) > context_window - buffer:
    messages = await summarize_old_messages(messages, keep_recent=5)
```

**Strategy 2: Tool Result Budget** — Persist large outputs to disk, keep summaries in context:
```python
if len(tool_result.content) > MAX_INLINE_SIZE:
    path = save_to_disk(tool_result.content)
    tool_result.content = f"[Result too large. Full output at: {path}]"
```

**Strategy 3: Micro-Compact** — Replace old tool results with `[Old tool result cleared]`:
```python
for msg in messages[:-RECENT_WINDOW:]:
    if msg.type == "tool_result" and msg.age > TURNOVER_THRESHOLD:
        msg.content = "[Old tool result content cleared]"
```

Load [references/context-management.md](./references/context-management.md) for the complete context lifecycle.

#### Phase 2.5: System Prompt Assembly

Build the system prompt as a string array, not a single blob:

```python
def build_system_prompt(tools, context):
    parts = []

    # Static prefix (cacheable)
    parts.append(identity_section())        # "You are..."
    parts.append(tool_instructions(tools))  # Per-tool usage rules
    parts.append(output_rules())            # Formatting, verbosity

    # Dynamic boundary (changes per turn)
    parts.append(current_datetime())        # Timestamp
    parts.append(git_context(context))      # Branch, status
    parts.append(user_context(context))     # CLAUDE.md or project config

    return parts
```

**Key pattern:** Split static/dynamic content with a cache boundary. Static prefix gets cached by the API (10x cheaper). Dynamic suffix changes every turn.

Load [references/system-prompts.md](./references/system-prompts.md) for the complete prompt template.

### Phase 3: Add the Interaction Layer

Choose your UI:

| Interface | Framework | Best For |
|-----------|-----------|----------|
| Terminal TUI | Ink (React) / Rich (Python) | Developer tools, CLIs |
| Web UI | React + SSE | User-facing products |
| API Server | FastAPI / Express | Headless agents, integrations |
| Headless CLI | argparse + streaming | CI/CD, automation |

**Terminal UI component tree (React/Ink pattern):**
```
App
  Messages          # Virtual scrollable message list
  Spinner            # "Thinking..." / "Running bash..." indicator
  PromptInput        # User input with history, autocomplete
  PermissionDialog   # "Allow Bash(git push)?" prompt
  CostDisplay        # Running token/cost counter
```

### Phase 4: Add Advanced Features

These make a good agent into a great one:

| Feature | Description | Reference |
|---------|-------------|-----------|
| Sub-agents | Spawn child agents for parallel work | [agent-loop.md](./references/agent-loop.md) |
| Hooks | Pre/post tool execution callbacks | [permission-system.md](./references/permission-system.md) |
| Skill system | Load dynamic capabilities at runtime | See templates |
| Slash commands | `/compact`, `/review`, `/cost` etc. | See templates |
| Streaming output | Show results as they arrive | [agent-loop.md](./references/agent-loop.md) |
| Cost tracking | Per-turn token and cost reporting | [architecture.md](./references/architecture.md) |
| Session persistence | Save/restore conversations | [memory.md](./references/memory.md) |
| Long-term memory | Facts that persist across sessions | [memory.md](./references/memory.md) |
| MCP integration | Plug in external tool servers | [mcp.md](./references/mcp.md) |

### Phase 5: Validate and Ship

Run the validation script to verify your agent has all critical components:

```bash
python scripts/validate_agent.py ./my-agent
```

This checks:
- Agent loop handles tool_use and text responses
- Tool registry with proper schemas
- Permission system with deny/allow/ask
- Context compaction for long conversations
- System prompt assembly
- Error handling for API failures
- Cost tracking

## Templates

Ready-to-use starting points. Copy and customize:

| Template | Files | Description |
|----------|-------|-------------|
| [minimal-agent](./templates/minimal-agent/) | 3 files | Single-purpose agent with 2 tools |
| [coding-agent](./templates/coding-agent/) | 7 files | Full coding assistant with 9 tools |
| [research-agent](./templates/research-agent/) | 5 files | Research and analysis agent |
| [task-agent](./templates/task-agent/) | 9 files | Multi-step orchestration agent |
| [workflow-agent](./templates/workflow-agent/) | 8 files | Planner + executor + HITL checkpoints |
| [api-agent](./templates/api-agent/) | 6 files | HTTP/SSE server for embedding in apps |

## Examples

Complete walkthroughs of real agent builds:

| Example | What It Builds | Stack |
|---------|---------------|-------|
| [python-coding-agent](./examples/python-coding-agent.md) | Full Claude Code clone | Python + Anthropic SDK + rich |
| [typescript-research-agent](./examples/typescript-research-agent.md) | Streaming research assistant | TypeScript + Bun + Anthropic SDK |
| [multi-agent-orchestrator](./examples/multi-agent-orchestrator.md) | Agent that spawns sub-agents | Python + asyncio |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/validate_agent.py` | Validate agent has all required components |
| `scripts/scaffold.py` | Generate agent project from template |

## Reference Files

Deep-dive documentation for each subsystem. Load when you need implementation details:

| File | Contents |
|------|----------|
| [references/architecture.md](./references/architecture.md) | Complete Claude Code architecture overview |
| [references/agent-loop.md](./references/agent-loop.md) | The streaming agent loop with error recovery (Python) |
| [references/typescript-agent-loop.md](./references/typescript-agent-loop.md) | Full TypeScript/Bun async generator agent loop |
| [references/tool-system.md](./references/tool-system.md) | Tool registry, schemas, and execution pipeline |
| [references/permission-system.md](./references/permission-system.md) | 7-layer permission pipeline with hooks |
| [references/context-management.md](./references/context-management.md) | Context window lifecycle and compaction |
| [references/system-prompts.md](./references/system-prompts.md) | Prompt assembly with cache optimization |
| [references/mcp.md](./references/mcp.md) | MCP server integration — plug in external tools |
| [references/memory.md](./references/memory.md) | Session persistence and long-term memory system |
| [references/rust-agent.md](./references/rust-agent.md) | Full Rust implementation with tokio + reqwest |
| [references/production-principles.md](./references/production-principles.md) | 85% compounding problem, Manus 6 principles, production checklist |
| [references/planner-executor.md](./references/planner-executor.md) | Three-agent pattern: planner + executor + verifier with dependency graph |
| [references/human-in-the-loop.md](./references/human-in-the-loop.md) | HITL approval matrix, tiered delegation, terminal + webhook notifications |
| [references/serving.md](./references/serving.md) | FastAPI + SSE server, multi-user sessions, React EventSource frontend |
| [references/long-tasks.md](./references/long-tasks.md) | File-based planning for 100+ step tasks, context handoff, resume protocol |
| [references/observability.md](./references/observability.md) | Trace system, OpenTelemetry, LangSmith, cost/latency/error dashboards |
| [references/cost-optimization.md](./references/cost-optimization.md) | Prompt caching, model routing, context discipline, batch API |
| [references/framework-guide.md](./references/framework-guide.md) | LangGraph vs CrewAI vs Mastra vs Vercel AI SDK — decision matrix + examples |
