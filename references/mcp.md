# MCP Integration

Connect external tools to your agent via the Model Context Protocol.

## What MCP Is

MCP servers expose tools over a standard JSON-RPC protocol via stdio (subprocess) or SSE (HTTP).
Your agent connects to them at startup, lists their available tools, and registers them alongside
built-in tools. From the agent loop's perspective, an MCP tool is indistinguishable from any
other `Tool` subclass — it goes through the same permission checks and execution pipeline.

## Connecting to an MCP Server (Python)

```python
import asyncio
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

@asynccontextmanager
async def connect_stdio(command: str, args: list[str], env: dict[str, str] | None = None):
    """Connect to an MCP server running as a subprocess."""
    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

@asynccontextmanager
async def connect_sse(url: str):
    """Connect to an MCP server over HTTP/SSE."""
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

async def list_server_tools(session: ClientSession) -> list[dict]:
    """Return raw MCP tool descriptors from the server."""
    response = await session.list_tools()
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema or {"properties": {}, "required": []},
        }
        for tool in response.tools
    ]
```

## MCPTool Adapter (Python)

Wraps a single MCP server tool into the agent's `Tool` interface so it slots into the registry.

```python
from typing import Any
from mcp import ClientSession

# Tool and ToolResult are defined in tool-system.md
class MCPTool(Tool):
    """Wraps an MCP server tool into the agent's Tool interface."""

    def __init__(self, session: ClientSession, server_name: str, descriptor: dict):
        self._session = session
        self._server_name = server_name
        self.name = f"mcp__{server_name}__{descriptor['name']}"
        self.description = descriptor["description"]
        self.input_schema = descriptor["input_schema"]
        self.is_read_only = False
        self.is_concurrency_safe = False

    async def call(self, input: dict, context: dict) -> ToolResult:
        # Strip the mcp__servername__ prefix to get the bare tool name
        bare_name = self.name.split("__", 2)[2]
        try:
            result = await self._session.call_tool(bare_name, arguments=input)
            parts = [
                item.text if hasattr(item, "text") else str(item)
                for item in result.content
            ]
            output = "\n".join(parts)
            if result.isError:
                return ToolResult(success=False, error=output)
            return ToolResult(success=True, content=output)
        except Exception as e:
            return ToolResult(success=False, error=f"MCP call failed: {e}")
```

## MCPTool Adapter (TypeScript)

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";

// Tool and ToolResult mirror the Python interface
interface ToolResult {
  success: boolean;
  content: string;
  error: string;
}

interface Tool {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  call(input: Record<string, unknown>): Promise<ToolResult>;
}

export async function createStdioClient(
  command: string,
  args: string[],
  env?: Record<string, string>
): Promise<Client> {
  const transport = new StdioClientTransport({ command, args, env });
  const client = new Client({ name: "agent-blueprint", version: "1.0.0" });
  await client.connect(transport);
  return client;
}

export async function createSSEClient(url: string): Promise<Client> {
  const transport = new SSEClientTransport(new URL(url));
  const client = new Client({ name: "agent-blueprint", version: "1.0.0" });
  await client.connect(transport);
  return client;
}

export class MCPTool implements Tool {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  private client: Client;
  private bareName: string;

  constructor(client: Client, serverName: string, descriptor: {
    name: string;
    description: string;
    inputSchema: Record<string, unknown>;
  }) {
    this.client = client;
    this.bareName = descriptor.name;
    this.name = `mcp__${serverName}__${descriptor.name}`;
    this.description = descriptor.description;
    this.inputSchema = descriptor.inputSchema;
  }

  async call(input: Record<string, unknown>): Promise<ToolResult> {
    try {
      const result = await this.client.callTool({
        name: this.bareName,
        arguments: input,
      });
      const parts = (result.content as Array<{ type: string; text?: string }>)
        .map((item) => (item.type === "text" ? (item.text ?? "") : JSON.stringify(item)));
      const output = parts.join("\n");
      return result.isError
        ? { success: false, content: "", error: output }
        : { success: true, content: output, error: "" };
    } catch (err) {
      return { success: false, content: "", error: `MCP call failed: ${err}` };
    }
  }
}

export async function loadMCPTools(
  client: Client,
  serverName: string
): Promise<MCPTool[]> {
  const { tools } = await client.listTools();
  return tools.map(
    (t) =>
      new MCPTool(client, serverName, {
        name: t.name,
        description: t.description ?? "",
        inputSchema: (t.inputSchema as Record<string, unknown>) ?? {},
      })
  );
}
```

## MCP Server Registry (Python)

Manages multiple MCP server connections and flattens their tools into one list.

```python
import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPServerRegistry:
    def __init__(self):
        self._stack = AsyncExitStack()
        self._tools: list[MCPTool] = []
        self._sessions: dict[str, ClientSession] = {}

    async def __aenter__(self):
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self._stack.__aexit__(*exc)

    async def connect_server(
        self,
        name: str,
        command: list[str],
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> list[MCPTool]:
        """Connect to an MCP server and register its tools. Returns the new tools."""
        params = StdioServerParameters(
            command=command[0],
            args=(command[1:] + (args or [])),
            env=env,
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        self._sessions[name] = session
        descriptors = await list_server_tools(session)
        new_tools = [MCPTool(session, name, d) for d in descriptors]
        self._tools.extend(new_tools)
        return new_tools

    def get_all_tools(self) -> list[MCPTool]:
        return list(self._tools)

    def get_tools_for_server(self, name: str) -> list[MCPTool]:
        prefix = f"mcp__{name}__"
        return [t for t in self._tools if t.name.startswith(prefix)]

    async def disconnect_all(self):
        await self._stack.aclose()
        self._tools.clear()
        self._sessions.clear()
```

## Integration with ToolRegistry

Drop MCP tools into the existing registry alongside built-in tools.

```python
from tool_system import create_default_tool_registry

async def build_registry_with_mcp() -> tuple[ToolRegistry, MCPServerRegistry]:
    registry = create_default_tool_registry()

    mcp = MCPServerRegistry()
    await mcp.__aenter__()

    await mcp.connect_server(
        "filesystem",
        ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."],
    )
    await mcp.connect_server(
        "github",
        ["npx", "-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ["GITHUB_TOKEN"]},
    )

    for tool in mcp.get_all_tools():
        registry.register(tool)

    # mcp must stay alive for the duration of the agent session
    return registry, mcp


# Usage in your main agent loop:
async def main():
    registry, mcp = await build_registry_with_mcp()
    try:
        await run_agent(registry)
    finally:
        await mcp.disconnect_all()
```

## Common MCP Servers

| Server | Install command | What it provides |
|--------|----------------|-----------------|
| filesystem | `npx -y @modelcontextprotocol/server-filesystem <path>` | File read/write/list inside a directory |
| sqlite | `npx -y @modelcontextprotocol/server-sqlite <db-file>` | SQLite queries and schema inspection |
| github | `npx -y @modelcontextprotocol/server-github` | Issues, PRs, file contents, search |
| brave-search | `npx -y @modelcontextprotocol/server-brave-search` | Web search via Brave API |
| postgres | `npx -y @modelcontextprotocol/server-postgres <conn-string>` | PostgreSQL queries |
| memory | `npx -y @modelcontextprotocol/server-memory` | Key/value store for agent memory |
| fetch | `npx -y @modelcontextprotocol/server-fetch` | HTTP GET for scraping/reading URLs |

Set required environment variables before spawning the server process. For example, `brave-search`
requires `BRAVE_API_KEY` and `github` requires `GITHUB_PERSONAL_ACCESS_TOKEN`.

## MCP Tools in the Permission System

MCP tools follow the naming convention `mcp__<servername>__<toolname>`. The permission system
treats them identically to built-in tools — the same `allow`/`deny`/`ask` rules apply.

### Allow read-only filesystem access only

```json
{
  "permissions": {
    "allow": [
      "mcp__filesystem__read_file",
      "mcp__filesystem__list_directory",
      "mcp__filesystem__get_file_info"
    ],
    "deny": [
      "mcp__filesystem__write_file",
      "mcp__filesystem__create_directory",
      "mcp__filesystem__move_file",
      "mcp__filesystem__delete_file"
    ]
  }
}
```

### Allow all tools from a server with a glob prefix

The `matches_rule` function in `permission-system.md` supports prefix matching. Add wildcard
support for MCP server names:

```python
def matches_mcp_rule(tool_name: str, rule: str) -> bool:
    """Match rules of the form 'mcp__server' (all tools) or 'mcp__server__tool'."""
    if not rule.startswith("mcp__"):
        return False
    # "mcp__github" matches "mcp__github__create_issue", "mcp__github__list_repos", etc.
    if tool_name.startswith(rule + "__") or tool_name == rule:
        return True
    return fnmatch(tool_name, rule)
```

Example rule set for a code-review agent:

```json
{
  "permissions": {
    "allow": [
      "mcp__filesystem__read_file",
      "mcp__filesystem__list_directory",
      "mcp__github__get_pull_request",
      "mcp__github__list_pull_request_files",
      "mcp__brave-search__search"
    ],
    "deny": [
      "mcp__github__merge_pull_request",
      "mcp__github__delete_branch"
    ],
    "ask": [
      "mcp__github__create_issue",
      "mcp__github__create_pull_request_review"
    ]
  }
}
```

### Deny all MCP tools globally (and selectively re-enable)

```json
{
  "permissions": {
    "deny": ["mcp__*"],
    "allow": ["mcp__filesystem__read_file", "mcp__filesystem__list_directory"]
  }
}
```

Note: process `allow` before `deny` in your rule matching, or invert the order depending on
which should take precedence in your permission system implementation.
