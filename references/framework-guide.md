# Framework Guide — Choosing the Right Agent Framework

Most developers reach for a framework before understanding what they're building.
This guide helps you pick the right tool — or decide to build without one.

---

## Quick Decision Matrix

| If you need... | Framework | Why |
|---|---|---|
| Production Python workflow with checkpointing | LangGraph | State persistence, HITL, audit trails |
| Quick business workflow prototype | CrewAI | Easy role definitions, fast to start |
| TypeScript-native agent | Mastra | TypeScript-first, MCP native, active ecosystem |
| Full-stack Next.js app with agent | Vercel AI SDK 6 | Streaming UI + agent loop in one stack |
| OpenAI models, simplest setup | OpenAI Agents SDK | 10.3M monthly downloads, straightforward |
| Claude models, deepest MCP integration | Anthropic Agent SDK | Best MCP integration, Claude-optimized |
| Azure/Microsoft enterprise | Microsoft Agent Framework | Governance, Azure AI Foundry, GA Q1 2026 |
| Build from scratch (learning/custom) | agent-blueprint patterns | Full control, no framework magic |

---

## When to Use agent-blueprint Patterns INSTEAD of a Framework

Frameworks solve common problems. They also add layers you may not need or want.

Use agent-blueprint patterns directly when:

- **You need full control** over every line of code — no hidden retry logic, no opaque prompt injection
- **You're building a framework or platform** — don't build on top of another framework; you'll fight it
- **Your tool execution is unusual** — streaming tool results, parallel fan-out, hardware APIs, custom protocols
- **You're optimizing aggressively for cost/latency** — framework overhead adds tokens and roundtrips
- **You're learning** — understanding the raw agent loop makes you a better engineer regardless of what you eventually use
- **Edge or embedded deployment** — framework cold starts and dependencies are unacceptable at the edge

The agent loop is not complex. It is a while loop, a tool dispatcher, and state management.
Every framework is that loop plus opinions. Know the loop first.

---

## 2026 Framework Landscape

### LangGraph
- ~30k GitHub stars. Dominant for production Python agent workflows.
- Directed graph model: nodes are functions, edges define control flow
- Checkpointing to PostgreSQL or Redis — full state persistence and replay
- `interrupt_before` pauses the graph at a named node for human approval
- 15B+ traces processed via LangSmith
- **Choose this for:** any Python agent that needs reliability, HITL, or audit trails

### CrewAI
- Role-based multi-agent: Researcher, Analyst, Writer personas
- Claims 450M monthly workflows, 60% Fortune 500
- **BUT:** 3x more tokens than LangChain, 3x slower in benchmarks
- Best use case: rapid prototyping and demos. Migrate to LangGraph for production.

### Mastra
- TypeScript-native, Y Combinator-backed ($13M seed)
- PayPal, Adobe, Docker in production
- 40+ model providers, native MCP support, supervisor pattern, LSP diagnostics
- **Choose this for:** any TypeScript agent not tied to a Next.js stack

### Vercel AI SDK 6 (Feb 2026)
- `ToolLoopAgent` class, `needsApproval` for HITL, full MCP support
- Built-in DevTools, Fluid compute for long-running agents
- **Choose this for:** Next.js or full-stack TypeScript where you want streaming UI + agent in one place

### OpenAI Agents SDK
- 10.3M monthly downloads, 19k GitHub stars
- Claims 100+ LLM support. Simplest path from prototype to production.
- **Choose this for:** teams already in the OpenAI ecosystem who want minimal setup

### Anthropic Agent SDK
- 4.6k stars. Deepest MCP integration of any framework.
- Anthropic models only.
- **Choose this for:** Claude-committed teams who want best-in-class MCP tooling

### Microsoft Agent Framework
- AutoGen is retired (maintenance mode only). Do not start new projects on AutoGen.
- Microsoft Agent Framework went GA Q1 2026. Replacement for AutoGen.
- AG2 is a community fork at ag2.ai — lives on, but fragmented ecosystem.
- **Choose this for:** Azure shops with enterprise governance requirements

### Google ADK
- New as of late 2025. OpenTelemetry built in. Google Cloud + Gemini native.
- Still early — API surface is changing. Budget 20–30% extra time for breaking changes.
- **Choose this for:** Gemini-native teams on Google Cloud who can absorb churn

---

## LangGraph — Setup and Core Patterns

Graph = nodes + edges + state. Every run is reproducible and resumable.

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

def call_model(state: AgentState) -> AgentState:
    # Call LLM, return updated messages
    ...

def call_tools(state: AgentState) -> AgentState:
    # Execute tool calls from last message
    ...

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tools"
    return "end"

# Build graph
builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("call_tools", call_tools)
builder.set_entry_point("call_model")
builder.add_conditional_edges("call_model", should_continue)
builder.add_edge("call_tools", "call_model")

# Compile with persistence
graph = builder.compile(
    checkpointer=PostgresSaver.from_conn_string("postgresql://user:pass@localhost/agentdb"),
    interrupt_before=["call_tools"]  # pause for human approval before any tool call
)

# Run
config = {"configurable": {"thread_id": "session-abc-123"}}
result = graph.invoke({"messages": [user_message]}, config=config)
```

**Key strength:** `interrupt_before=["call_tools"]` pauses execution and persists state.
Resume with the same `thread_id` after human review — the graph continues exactly where it stopped.

---

## CrewAI — Setup and Core Patterns

Agents have roles. Tasks have expected outputs. The crew executes sequentially or in parallel.

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Research Analyst",
    goal="Find accurate, current information on the given topic",
    backstory="You are a meticulous researcher with 10 years of experience in competitive intelligence.",
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Write clear, structured reports from research findings",
    backstory="You turn dense research into readable executive summaries.",
    verbose=True
)

research_task = Task(
    description="Research the current state of {topic}. Find key players, recent developments, and market size.",
    agent=researcher,
    expected_output="Markdown report with sources, minimum 500 words"
)

write_task = Task(
    description="Write an executive summary based on the research provided.",
    agent=writer,
    expected_output="500-word executive summary with 3 key takeaways",
    context=[research_task]
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff(inputs={"topic": "AI agent frameworks 2026"})
print(result.raw)
```

**Warning:** CrewAI uses 3x more tokens than an equivalent LangChain implementation and runs 3x slower
in benchmarks. Use it to validate business logic and role definitions. Migrate to LangGraph for production.

---

## Mastra — Setup and Core Patterns

TypeScript-native. MCP built in. 40+ providers from one API.

```typescript
import { Mastra, createAgent, createTool } from "@mastra/core";
import { z } from "zod";

// Define a tool
const webSearch = createTool({
  id: "web-search",
  description: "Search the web for current information",
  inputSchema: z.object({ query: z.string() }),
  outputSchema: z.object({ results: z.array(z.string()) }),
  execute: async ({ context }) => {
    // your search implementation
    return { results: [] };
  }
});

// Define an agent
const researchAgent = createAgent({
  name: "researcher",
  model: {
    provider: "ANTHROPIC",
    name: "claude-sonnet-4-6"
  },
  tools: { webSearch },
  instructions: `You are a research specialist.
  Search for current, accurate information and cite your sources.
  Always verify facts before including them in your response.`
});

// Register with Mastra
const mastra = new Mastra({
  agents: { researcher: researchAgent }
});

// Run
const result = await mastra.getAgent("researcher").generate(
  "Research the current state of LangGraph adoption in production"
);

console.log(result.text);
```

**Supervisor pattern** (multi-agent):
```typescript
const supervisorAgent = createAgent({
  name: "supervisor",
  model: { provider: "ANTHROPIC", name: "claude-sonnet-4-6" },
  agents: { researcher: researchAgent, writer: writerAgent },
  instructions: "Coordinate researcher and writer agents to produce reports."
});
```

---

## Vercel AI SDK 6 — Setup and Core Patterns

Best for Next.js. Streaming UI + agent loop coexist in one stack.

```typescript
import { ToolLoopAgent } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { tool } from "ai";
import { z } from "zod";

// Define tools
const bashTool = tool({
  description: "Execute a shell command",
  parameters: z.object({ command: z.string() }),
  needsApproval: true, // HITL — prompts user before execution
  execute: async ({ command }) => {
    // your bash execution
    return { output: "" };
  }
});

const readFileTool = tool({
  description: "Read a file from the filesystem",
  parameters: z.object({ path: z.string() }),
  execute: async ({ path }) => {
    // your file reading
    return { content: "" };
  }
});

// Create agent
const agent = new ToolLoopAgent({
  model: anthropic("claude-sonnet-4-6"),
  tools: { bash: bashTool, readFile: readFileTool },
  maxSteps: 20,
  system: "You are a helpful coding assistant."
});

// In a Next.js API route (app/api/agent/route.ts):
export async function POST(req: Request) {
  const { prompt } = await req.json();

  const { stream } = await agent.streamText({ prompt });
  return stream.toDataStreamResponse();
}
```

**On the client** (React):
```typescript
import { useAgent } from "ai/react";

export default function AgentChat() {
  const { messages, sendMessage, isLoading } = useAgent({ api: "/api/agent" });

  return (
    <div>
      {messages.map(m => <div key={m.id}>{m.content}</div>)}
      <input onKeyDown={e => e.key === "Enter" && sendMessage(e.currentTarget.value)} />
    </div>
  );
}
```

---

## Migration Path

The common journey from learning to production:

```
Start with agent-blueprint patterns (understand the raw agent loop)
    ↓
Prototype with CrewAI or agent-blueprint minimal template (fast iteration, validate logic)
    ↓
Production with LangGraph (Python) or Mastra (TypeScript)
    (checkpointing, HITL, observability)
    ↓
Scale with framework + agent-blueprint production principles
    (cost optimization, reliability, graceful degradation)
```

Skip steps only if you understand what you're skipping. Going directly from prototype to production
without addressing checkpointing and HITL is the most common cause of agent reliability failures.

---

## The "Don't Make Everything Agentic" Rule

Most real applications: **30–40% genuinely needs LLM reasoning.** The rest should be:

- Deterministic Python or TypeScript code
- Standard API calls
- Database queries
- Simple rule-based logic

**The mistake:** wrapping everything in an agent loop when it's actually just `if/else`.

```python
# Wrong — using an agent for something deterministic
result = agent.run("Check if the user's subscription is active and return True or False")

# Right — just check the database
is_active = db.query("SELECT active FROM subscriptions WHERE user_id = ?", user_id)
```

If you can write the logic as a function without calling an LLM, write it as a function.
Agents are for tasks that require reasoning over ambiguous inputs, multi-step planning, or
dynamic tool selection based on context. Not for tasks that have a deterministic answer.

**Identifying what needs an agent:**
- Input is ambiguous or variable in structure → agent
- Steps depend on previous results in unpredictable ways → agent
- Tool selection depends on reasoning about the task → agent
- Logic is a known sequence of deterministic operations → function, not agent

Keep the LLM calls minimal. Every unnecessary agent call is latency, cost, and a failure surface.

---

## Summary

| Framework | Language | Best For | Watch Out For |
|---|---|---|---|
| LangGraph | Python | Production, checkpointing, HITL | Steeper learning curve |
| CrewAI | Python | Prototyping, role-based demos | 3x token cost, 3x slower |
| Mastra | TypeScript | TS agents, MCP, multi-provider | Newer, fewer Stack Overflow answers |
| Vercel AI SDK 6 | TypeScript | Next.js, streaming UI + agent | Vercel ecosystem lock-in |
| OpenAI Agents SDK | Python/TS | Simplicity, OpenAI-first teams | Less flexible outside OpenAI |
| Anthropic Agent SDK | Python | Claude + MCP, deepest integration | Anthropic models only |
| Microsoft Agent Framework | Python | Azure enterprise, governance | Azure dependency |
| Google ADK | Python | Gemini, Google Cloud native | Breaking changes expected |
| agent-blueprint patterns | Any | Control, learning, custom needs | You write more code |
