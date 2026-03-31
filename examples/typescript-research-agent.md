# Example: TypeScript Research Agent

Build a streaming research agent in TypeScript using Bun and the Anthropic SDK.

## What We're Building

A research assistant that:
- Searches the web for information (plug in Brave/Tavily/SerpAPI)
- Runs bash commands with async, non-blocking execution via `Bun.spawn`
- Streams response tokens to the terminal as they arrive
- Loops until the model stops calling tools

## Setup

```bash
mkdir my-research-agent && cd my-research-agent
bun init -y
bun add @anthropic-ai/sdk
```

## agent.ts

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// --- Tool schemas (inline JSON — no zod-to-json-schema needed) ---

const tools: Anthropic.Tool[] = [
  {
    name: "bash",
    description: "Execute a shell command and return stdout/stderr.",
    input_schema: {
      type: "object" as const,
      properties: {
        command: {
          type: "string",
          description: "Shell command to execute",
        },
      },
      required: ["command"],
    },
  },
  {
    name: "web_search",
    description: "Search the web for current information.",
    input_schema: {
      type: "object" as const,
      properties: {
        query: {
          type: "string",
          description: "Search query",
        },
      },
      required: ["query"],
    },
  },
];

// --- Tool execution ---

async function executeTool(name: string, input: Record<string, string>): Promise<string> {
  switch (name) {
    case "bash": {
      try {
        const proc = Bun.spawn(["bash", "-c", input.command], {
          stdout: "pipe",
          stderr: "pipe",
        });
        const stdout = await new Response(proc.stdout).text();
        const stderr = await new Response(proc.stderr).text();
        await proc.exited;
        return stdout + (stderr ? `\nSTDERR: ${stderr}` : "");
      } catch (e: unknown) {
        return `Error: ${e instanceof Error ? e.message : String(e)}`;
      }
    }

    case "web_search": {
      // Replace this stub with a real search API.
      // Brave Search: set BRAVE_API_KEY and call https://api.search.brave.com/res/v1/web/search
      // Tavily:       set TAVILY_API_KEY and call https://api.tavily.com/search
      // SerpAPI:      set SERPAPI_KEY and call https://serpapi.com/search
      const apiKey = process.env.BRAVE_API_KEY;
      if (!apiKey) {
        return `[web_search stub] Query: "${input.query}" — set BRAVE_API_KEY to enable real search.`;
      }
      const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(input.query)}&count=5`;
      const res = await fetch(url, {
        headers: { "X-Subscription-Token": apiKey, Accept: "application/json" },
      });
      const data = (await res.json()) as { web?: { results?: Array<{ title: string; url: string; description: string }> } };
      const results = data.web?.results ?? [];
      return results
        .map((r) => `[${r.title}](${r.url})\n${r.description}`)
        .join("\n\n") || "No results found.";
    }

    default:
      return `Unknown tool: ${name}`;
  }
}

// --- Agent loop ---

async function agent(prompt: string): Promise<void> {
  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: prompt },
  ];

  while (true) {
    // Collect streamed content blocks
    const contentBlocks: Anthropic.ContentBlock[] = [];
    let currentText = "";
    let currentToolUse: Partial<Anthropic.ToolUseBlock> & { partial_json?: string } = {};

    const stream = client.messages.stream({
      model: "claude-sonnet-4-6",
      max_tokens: 4096,
      system: `You are a research assistant. Use web_search to find current information and bash for
local tasks. Provide well-sourced answers and cite URLs when available.`,
      tools,
      messages,
    });

    for await (const event of stream) {
      if (event.type === "content_block_start") {
        if (event.content_block.type === "text") {
          currentText = "";
        } else if (event.content_block.type === "tool_use") {
          currentToolUse = {
            type: "tool_use",
            id: event.content_block.id,
            name: event.content_block.name,
            partial_json: "",
          };
        }
      } else if (event.type === "content_block_delta") {
        if (event.delta.type === "text_delta") {
          process.stdout.write(event.delta.text);
          currentText += event.delta.text;
        } else if (event.delta.type === "input_json_delta") {
          currentToolUse.partial_json = (currentToolUse.partial_json ?? "") + event.delta.partial_json;
        }
      } else if (event.type === "content_block_stop") {
        if (currentText) {
          contentBlocks.push({ type: "text", text: currentText });
          currentText = "";
        } else if (currentToolUse.id) {
          contentBlocks.push({
            type: "tool_use",
            id: currentToolUse.id!,
            name: currentToolUse.name!,
            input: JSON.parse(currentToolUse.partial_json || "{}"),
          } as Anthropic.ToolUseBlock);
          currentToolUse = {};
        }
      }
    }

    messages.push({ role: "assistant", content: contentBlocks });

    // Collect tool calls
    const toolUses = contentBlocks.filter(
      (b): b is Anthropic.ToolUseBlock => b.type === "tool_use"
    );

    if (toolUses.length === 0) {
      process.stdout.write("\n");
      break;
    }

    // Execute tools and build result message
    const toolResults: Anthropic.ToolResultBlockParam[] = [];
    for (const toolUse of toolUses) {
      const result = await executeTool(toolUse.name, toolUse.input as Record<string, string>);
      process.stdout.write(`\n[${toolUse.name}] OK\n`);
      toolResults.push({
        type: "tool_result",
        tool_use_id: toolUse.id,
        content: result,
      });
    }

    messages.push({ role: "user", content: toolResults });
  }
}

// --- Entry point ---

const prompt = process.argv.slice(2).join(" ") || "What are the best practices for building AI agents in 2026?";
agent(prompt).catch(console.error);
```

## Usage

```bash
# Run directly with Bun
bun run agent.ts "Research the latest developments in AI agent frameworks"

# With a real search API (Brave)
BRAVE_API_KEY=your-key bun run agent.ts "What is the current state of LLM context windows?"
```

## Expected Output

```
I'll research AI agent frameworks for you.

[web_search] OK

Based on my research, here are the key developments in AI agent frameworks in 2026:

1. **LangGraph** remains dominant for stateful multi-agent systems...
2. **Anthropic's Agent SDK** introduced native streaming tool execution...

Sources:
- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- [Anthropic Blog](https://anthropic.com/research)
```
