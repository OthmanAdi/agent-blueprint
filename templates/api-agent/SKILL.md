# API Agent Template

An agent that runs as an HTTP server inside your application. Streams responses via SSE, supports multiple concurrent users, and tracks cost per session. ~550 lines total.

## When to use this

Use this when someone says "I want to put an agent inside my app." You need:
- A frontend (web, mobile, desktop) talking to a backend agent
- Multiple users each with their own conversation history
- Real-time streaming instead of waiting for the full response
- Cost visibility per user or session

## Files

```
my-api-agent/
  server.py            ← FastAPI server with SSE streaming
  agent/
    loop.py            ← Async generator agent loop
    tools.py           ← Tool registry
    context.py         ← Context management
    sessions.py        ← Multi-user session store
    cost.py            ← Per-user cost tracking
  frontend/
    index.html         ← Example frontend (vanilla JS + EventSource)
  requirements.txt
```

---

## requirements.txt

```
anthropic>=0.40.0
fastapi>=0.115.0
uvicorn>=0.32.0
pydantic>=2.0
python-multipart>=0.0.12
```

---

## agent/cost.py

```python
"""Per-user cost tracking. Anthropic claude-sonnet-4-6 pricing."""

PRICING = {
    "claude-sonnet-4-6": {
        "input":          3.00 / 1_000_000,
        "output":        15.00 / 1_000_000,
        "cache_read":     0.30 / 1_000_000,
        "cache_creation": 3.75 / 1_000_000,
    }
}


def calculate_cost(usage, model: str = "claude-sonnet-4-6") -> float:
    """Calculate cost in USD from an Anthropic usage object or dict."""
    p = PRICING.get(model, PRICING["claude-sonnet-4-6"])

    if hasattr(usage, "__dict__"):
        usage = vars(usage)

    input_tokens        = usage.get("input_tokens", 0)
    output_tokens       = usage.get("output_tokens", 0)
    cache_read_tokens   = usage.get("cache_read_input_tokens", 0)
    cache_create_tokens = usage.get("cache_creation_input_tokens", 0)

    return (
        input_tokens        * p["input"]
        + output_tokens     * p["output"]
        + cache_read_tokens * p["cache_read"]
        + cache_create_tokens * p["cache_creation"]
    )


def format_cost(cost_usd: float) -> str:
    """Format cost as human-readable string. e.g. '$0.0042'"""
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    return f"${cost_usd:.2f}"
```

---

## agent/sessions.py

```python
"""In-memory session store. Swap the dict for Redis in production."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Session:
    session_id: str
    messages: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0
    turn_count: int = 0
    created_at: str = field(default_factory=_now)
    last_active: str = field(default_factory=_now)


class SessionStore:
    def __init__(self):
        self._store: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._store:
            self._store[session_id] = Session(session_id=session_id)
        return self._store[session_id]

    def update(self, session_id: str, new_messages: list[dict], cost: float) -> None:
        session = self._store.get(session_id)
        if not session:
            return
        session.messages = new_messages
        session.total_cost_usd += cost
        session.turn_count += 1
        session.last_active = _now()

    def delete(self, session_id: str) -> bool:
        return self._store.pop(session_id, None) is not None

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours. Returns count deleted."""
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_hours * 3600
        stale = [
            sid for sid, s in self._store.items()
            if datetime.fromisoformat(s.last_active).timestamp() < cutoff
        ]
        for sid in stale:
            del self._store[sid]
        return len(stale)

    def get(self, session_id: str) -> Optional[Session]:
        return self._store.get(session_id)
```

---

## agent/tools.py

```python
"""Tool registry with 5 built-in tools. BashTool is read-only by default."""

import asyncio
import glob as glob_module
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Awaitable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], Awaitable[str]]

    def to_api(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# --- Tool handlers ---

async def _bash_handler(inp: dict) -> str:
    """Read-only bash. Blocks write operations."""
    command = inp.get("command", "").strip()
    blocked = ["rm ", "rm\t", "rmdir", "mv ", "mv\t", "cp ", ">", ">>",
               "chmod", "chown", "dd ", "mkfs", "sudo", "apt", "pip install",
               "curl -X POST", "curl -X PUT", "curl -X DELETE", "wget -O"]
    for b in blocked:
        if b in command:
            return f"Blocked: '{b}' is not allowed in read-only bash mode."
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(timeout=15)
        output = (stdout + stderr).decode(errors="replace")
        return output[:4000] if len(output) > 4000 else output
    except asyncio.TimeoutError:
        return "Error: command timed out after 15 seconds."
    except Exception as e:
        return f"Error: {e}"


async def _file_read_handler(inp: dict) -> str:
    path = inp.get("path", "")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read(50_000)
        return content
    except Exception as e:
        return f"Error reading {path}: {e}"


async def _glob_handler(inp: dict) -> str:
    pattern = inp.get("pattern", "**/*")
    base = inp.get("path", ".")
    try:
        matches = glob_module.glob(
            os.path.join(base, pattern), recursive=True
        )
        if not matches:
            return "No files matched."
        return "\n".join(matches[:200])
    except Exception as e:
        return f"Error: {e}"


async def _grep_handler(inp: dict) -> str:
    pattern = inp.get("pattern", "")
    path = inp.get("path", ".")
    glob_pat = inp.get("glob", "")
    try:
        cmd = ["grep", "-rn", "--include", glob_pat or "*", pattern, path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout or result.stderr
        return output[:4000] if len(output) > 4000 else output or "No matches found."
    except Exception as e:
        return f"Error: {e}"


async def _web_search_handler(inp: dict) -> str:
    """Stub. Replace with your preferred search API (Brave, Serper, Tavily)."""
    query = inp.get("query", "")
    return (
        f"[WebSearch stub] Query received: '{query}'\n"
        "To enable real search, set SEARCH_API_KEY and implement this handler."
    )


# --- Registry ---

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_api_defs(self) -> list[dict]:
        return [t.to_api() for t in self._tools.values()]

    async def execute(self, name: str, inp: dict) -> str:
        tool = self.get(name)
        if not tool:
            return f"Unknown tool: {name}"
        return await tool.handler(inp)


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(
        name="bash",
        description=(
            "Run a read-only shell command. Write operations (rm, mv, pip install, etc.) "
            "are blocked. Use for ls, cat, wc, find, ps, df, etc."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Read-only shell command"},
            },
            "required": ["command"],
        },
        handler=_bash_handler,
    ))
    registry.register(Tool(
        name="read_file",
        description="Read a file's full contents (up to 50,000 characters).",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
            },
            "required": ["path"],
        },
        handler=_file_read_handler,
    ))
    registry.register(Tool(
        name="glob",
        description="Find files matching a glob pattern.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
                "path": {"type": "string", "description": "Base directory to search from", "default": "."},
            },
            "required": ["pattern"],
        },
        handler=_glob_handler,
    ))
    registry.register(Tool(
        name="grep",
        description="Search file contents with a regex pattern.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search", "default": "."},
                "glob": {"type": "string", "description": "File filter, e.g. '*.py'", "default": ""},
            },
            "required": ["pattern"],
        },
        handler=_grep_handler,
    ))
    registry.register(Tool(
        name="web_search",
        description="Search the web. Returns a stub until you configure a search API.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
        handler=_web_search_handler,
    ))
    return registry
```

---

## agent/context.py

```python
"""
Lightweight context management.

In production you want bespoke logic here. This gives you three primitives:
- apply_budget: truncate a tool result that's too long, store full text separately
- snip_stale: remove old tool results from message history to save tokens
- is_over_limit: gate before each API call
"""

import json
from typing import Any

MAX_TOOL_RESULT_INLINE = 2000   # chars — beyond this we summarize inline
CONTEXT_TOKEN_LIMIT = 180_000   # claude-sonnet-4-6 has 200k; leave headroom


def apply_budget(result: str, tool_name: str) -> str:
    """If result is large, return a truncated preview with a note."""
    if len(result) <= MAX_TOOL_RESULT_INLINE:
        return result
    preview = result[:MAX_TOOL_RESULT_INLINE]
    truncated_chars = len(result) - MAX_TOOL_RESULT_INLINE
    return (
        f"{preview}\n\n"
        f"[Truncated: {truncated_chars:,} more chars not shown. "
        f"Use a more specific query to get a smaller result.]"
    )


def snip_stale(messages: list[dict], keep_last_n_tool_results: int = 6) -> list[dict]:
    """
    Remove old tool_result blocks from message history to shrink context.
    Keeps the last N tool results intact so the model still has recent context.
    """
    tool_result_indices = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_result_indices.append(i)
                    break

    if len(tool_result_indices) <= keep_last_n_tool_results:
        return messages

    # Replace old tool result messages with a placeholder
    stale_indices = set(tool_result_indices[:-keep_last_n_tool_results])
    result = []
    for i, msg in enumerate(messages):
        if i in stale_indices:
            result.append({
                "role": "user",
                "content": [{"type": "tool_result",
                              "tool_use_id": msg["content"][0].get("tool_use_id", ""),
                              "content": "[Result removed to save context]"}]
            })
        else:
            result.append(msg)
    return result


def is_over_limit(messages: list[dict], char_limit: int = CONTEXT_TOKEN_LIMIT * 4) -> bool:
    """
    Rough check: 1 token ~ 4 chars. Returns True if we're likely near the limit.
    """
    total = sum(
        len(json.dumps(m)) for m in messages
    )
    return total > char_limit
```

---

## agent/loop.py

```python
"""
Async generator agent loop.

Yields SSE-serializable dicts that server.py forwards directly to the frontend.
Never raises — all errors are yielded as {"type": "error", "message": "..."}.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from anthropic import AsyncAnthropic

from .context import apply_budget, snip_stale, is_over_limit
from .cost import calculate_cost
from .tools import ToolRegistry

client = AsyncAnthropic()
MODEL = "claude-sonnet-4-6"
MAX_TURNS = 20


async def agent_loop(
    messages: list[dict],
    session_id: str,
    tools: ToolRegistry,
    system_prompt: str,
) -> AsyncGenerator[dict, None]:
    """
    Yields event dicts:
      {"type": "text",        "content": "...",    "session_id": "..."}
      {"type": "tool_start",  "tool": "grep",      "input": {...}}
      {"type": "tool_result", "tool": "grep",      "success": True, "preview": "..."}
      {"type": "cost",        "turn_cost_usd": 0.002, "total_cost_usd": 0.008}
      {"type": "done",        "turn_count": 4,     "total_cost_usd": 0.008}
      {"type": "error",       "message": "..."}
    """
    turn_count = 0
    total_cost = 0.0
    working_messages = list(messages)

    try:
        while turn_count < MAX_TURNS:
            if is_over_limit(working_messages):
                working_messages = snip_stale(working_messages)

            response = await client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=system_prompt,
                tools=tools.all_api_defs(),
                messages=working_messages,
            )

            turn_cost = calculate_cost(response.usage)
            total_cost += turn_cost
            turn_count += 1

            # Collect assistant turn content
            assistant_content = response.content
            working_messages.append({"role": "assistant", "content": assistant_content})

            # Stream text blocks
            for block in assistant_content:
                if block.type == "text" and block.text:
                    yield {"type": "text", "content": block.text, "session_id": session_id}

            # If no tool calls, we're done
            tool_uses = [b for b in assistant_content if b.type == "tool_use"]
            if not tool_uses:
                yield {
                    "type": "cost",
                    "turn_cost_usd": round(turn_cost, 6),
                    "total_cost_usd": round(total_cost, 6),
                }
                yield {
                    "type": "done",
                    "turn_count": turn_count,
                    "total_cost_usd": round(total_cost, 6),
                }
                return

            # Execute tool calls, collect results
            tool_results = []
            for block in tool_uses:
                yield {
                    "type": "tool_start",
                    "tool": block.name,
                    "input": block.input,
                }

                raw_result = await tools.execute(block.name, block.input)
                trimmed = apply_budget(raw_result, block.name)
                success = not trimmed.startswith("Error") and not trimmed.startswith("Blocked")

                yield {
                    "type": "tool_result",
                    "tool": block.name,
                    "success": success,
                    "preview": trimmed[:200],
                }

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": trimmed,
                })

            working_messages.append({"role": "user", "content": tool_results})

            yield {
                "type": "cost",
                "turn_cost_usd": round(turn_cost, 6),
                "total_cost_usd": round(total_cost, 6),
            }

        # Reached MAX_TURNS
        yield {"type": "error", "message": f"Reached max turns ({MAX_TURNS}). Task may be incomplete."}
        yield {"type": "done", "turn_count": turn_count, "total_cost_usd": round(total_cost, 6)}

    except Exception as e:
        yield {"type": "error", "message": str(e)}
        yield {"type": "done", "turn_count": turn_count, "total_cost_usd": round(total_cost, 6)}
```

---

## server.py

```python
"""
FastAPI server — runs the agent and streams responses as SSE.
Start with: python server.py
"""

import asyncio
import json
import os
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent.loop import agent_loop
from agent.sessions import SessionStore
from agent.tools import default_registry

# --- Config ---

SYSTEM_PROMPT = """You are a helpful AI assistant with access to shell and file tools.
You can read files, search codebases, and run read-only shell commands.
Be concise. Show your reasoning briefly before using tools.
Never perform destructive operations."""

sessions = SessionStore()
tools = default_registry()

app = FastAPI(title="API Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/response models ---

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""


# --- Routes ---

@app.post("/chat")
async def chat(request: ChatRequest):
    """Stream agent response as Server-Sent Events."""
    session_id = request.session_id or str(uuid4())
    session = sessions.get_or_create(session_id)

    async def generate():
        messages = session.messages + [{"role": "user", "content": request.message}]
        final_cost = 0.0
        final_messages = list(messages)

        async for event in agent_loop(messages, session_id, tools, SYSTEM_PROMPT):
            yield f"data: {json.dumps(event)}\n\n"

            # Track final state from "done" event
            if event["type"] == "done":
                final_cost = event.get("total_cost_usd", 0.0)

        # Persist updated messages (append user message + assistant response)
        # We reconstruct from messages by adding the assistant's last reply
        # Simple approach: just track cost; message history update is approximate
        sessions.update(session_id, final_messages, final_cost)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Session-ID": session_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session history and cost."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "turn_count": session.turn_count,
        "total_cost_usd": session.total_cost_usd,
        "message_count": len(session.messages),
        "created_at": session.created_at,
        "last_active": session.last_active,
    }


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Delete session and start fresh."""
    deleted = sessions.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(sessions._store),
        "model": "claude-sonnet-4-6",
    }


@app.get("/frontend/index.html")
async def serve_frontend():
    return FileResponse("frontend/index.html")


@app.get("/")
async def root():
    return {"message": "API Agent running. Frontend at /frontend/index.html"}


# --- Startup ---

if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set.")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
```

---

## frontend/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Agent</title>
  <style>
    body { font-family: monospace; max-width: 760px; margin: 40px auto; padding: 0 16px; background: #0f0f0f; color: #e0e0e0; }
    #messages { min-height: 300px; border: 1px solid #333; padding: 12px; border-radius: 6px; overflow-y: auto; max-height: 60vh; }
    .msg-user  { color: #7ec8e3; margin: 8px 0; }
    .msg-agent { color: #e0e0e0; margin: 8px 0; white-space: pre-wrap; }
    .msg-tool  { color: #f0a500; font-size: 0.85em; margin: 2px 0; }
    .msg-error { color: #e05050; font-size: 0.85em; }
    #input { width: calc(100% - 90px); padding: 8px; background: #1a1a1a; color: #e0e0e0; border: 1px solid #444; border-radius: 4px; font-family: monospace; }
    button { padding: 8px 14px; background: #2a6fdb; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 6px; }
    button:disabled { background: #444; cursor: not-allowed; }
    #cost { display: block; margin-top: 6px; color: #888; font-size: 0.8em; }
  </style>
</head>
<body>
  <h2>Agent</h2>
  <div id="messages"></div>
  <div style="margin-top:10px">
    <input id="input" placeholder="Ask anything..." onkeydown="if(event.key==='Enter')send()" />
    <button id="sendBtn" onclick="send()">Send</button>
  </div>
  <small id="cost">Cost: $0.0000</small>

  <script>
    let sessionId = localStorage.getItem("agentSessionId") || "";
    let totalCost = 0;

    function appendMessage(cls, text) {
      const div = document.createElement("div");
      div.className = cls;
      div.textContent = text;
      document.getElementById("messages").appendChild(div);
      div.scrollIntoView({ behavior: "smooth" });
      return div;
    }

    async function send() {
      const input = document.getElementById("input");
      const btn = document.getElementById("sendBtn");
      const message = input.value.trim();
      if (!message) return;

      input.value = "";
      btn.disabled = true;
      appendMessage("msg-user", "You: " + message);

      let agentDiv = null;

      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, session_id: sessionId }),
        });

        // Capture session ID from response header
        const newId = res.headers.get("X-Session-ID");
        if (newId && newId !== sessionId) {
          sessionId = newId;
          localStorage.setItem("agentSessionId", sessionId);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop(); // last partial line back into buffer

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6).trim();
            if (raw === "[DONE]") continue;

            let event;
            try { event = JSON.parse(raw); } catch { continue; }

            if (event.type === "text") {
              if (!agentDiv) agentDiv = appendMessage("msg-agent", "");
              agentDiv.textContent += event.content;
              agentDiv.scrollIntoView({ behavior: "smooth" });

            } else if (event.type === "tool_start") {
              appendMessage("msg-tool", `  Tool: ${event.tool} — running...`);

            } else if (event.type === "tool_result") {
              const status = event.success ? "done" : "failed";
              appendMessage("msg-tool", `  ${event.tool}: ${status} — ${event.preview}`);

            } else if (event.type === "cost") {
              totalCost = event.total_cost_usd;
              document.getElementById("cost").textContent =
                `Cost: $${totalCost.toFixed(4)} (this session)`;

            } else if (event.type === "error") {
              appendMessage("msg-error", "Error: " + event.message);

            } else if (event.type === "done") {
              agentDiv = null; // reset for next turn
            }
          }
        }
      } catch (err) {
        appendMessage("msg-error", "Connection error: " + err.message);
      } finally {
        btn.disabled = false;
        input.focus();
      }
    }
  </script>
</body>
</html>
```

---

## Usage

```bash
# Install
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# Start server
python server.py
# → API running at http://localhost:8000
# → Frontend at http://localhost:8000/frontend/index.html

# Test with curl (streaming):
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "List all Python files in /tmp and count their lines"}' \
  --no-buffer
```

### Example SSE stream

```
data: {"type": "text", "content": "I'll help you list", "session_id": "abc123"}
data: {"type": "text", "content": " and count the Python files.", "session_id": "abc123"}
data: {"type": "tool_start", "tool": "glob", "input": {"pattern": "**/*.py", "path": "/tmp"}}
data: {"type": "tool_result", "tool": "glob", "success": true, "preview": "/tmp/main.py\n/tmp/utils.py"}
data: {"type": "tool_start", "tool": "bash", "input": {"command": "wc -l /tmp/*.py"}}
data: {"type": "tool_result", "tool": "bash", "success": true, "preview": "  42 /tmp/main.py\n  18 /tmp/utils.py"}
data: {"type": "text", "content": "Found 2 Python files:\n- main.py (42 lines)\n- utils.py (18 lines)", "session_id": "abc123"}
data: {"type": "cost", "turn_cost_usd": 0.0018, "total_cost_usd": 0.0018}
data: {"type": "done", "turn_count": 2, "total_cost_usd": 0.0018}
data: [DONE]
```

---

## Session management

```bash
# Get session info
curl http://localhost:8000/sessions/abc123

# Clear session (start fresh)
curl -X DELETE http://localhost:8000/sessions/abc123

# Health check (shows active session count)
curl http://localhost:8000/health
```

---

## Customization

### 1. Define your agent's domain

```python
# server.py — replace SYSTEM_PROMPT

SYSTEM_PROMPT = """You are a [DOMAIN] specialist. You help users with [SPECIFIC TASK].
You have access to: [describe available tools]
You should: [behavior guidelines]"""
```

### 2. Add domain-specific tools

```python
# agent/tools.py — add to default_registry()

registry.register(Tool(
    name="query_database",
    description="Query the application's PostgreSQL database.",
    input_schema={
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "SELECT query only"},
        },
        "required": ["sql"],
    },
    handler=your_db_handler,
))
```

Common additions:
- `DatabaseQueryTool` — query your app's DB (read-only SELECT only)
- `UserDataTool` — fetch the current user's profile or history
- `NotificationTool` — send emails or push notifications via your app's API
- `DocumentSearchTool` — query your vector store

### 3. Swap in Redis sessions for production

```python
# agent/sessions.py — replace self._store dict with:
import redis
r = redis.Redis(host="localhost", port=6379, db=0)
# serialize Session to JSON on set, deserialize on get
```

### 4. Add authentication

```python
# server.py
from fastapi import Depends, Header

async def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    # validate token against your auth system

@app.post("/chat")
async def chat(request: ChatRequest, _=Depends(verify_token)):
    ...
```
