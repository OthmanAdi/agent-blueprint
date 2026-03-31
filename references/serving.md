# Serving an Agent as an HTTP API with SSE Streaming

The "agent in an application" pattern: your agent loop becomes an HTTP endpoint.
Clients connect, stream events in real-time, sessions are isolated per user.

---

## 1. Architecture Overview

```
Client (React / any frontend)
        ↕  SSE stream  (text/event-stream)
FastAPI / Hono server
        ↕  async generator
Agent Loop  (agent_loop.py / agentLoop.ts)
        ↕  tool calls, API requests
Claude API + Tools
```

Key constraint: SSE is one-directional (server → client). The client POSTs a
message once, then reads the stream. For bidirectional real-time use WebSockets —
but for most LLM apps, SSE is the right choice. Simpler, HTTP/1.1 compatible,
auto-reconnect built in.

---

## 2. FastAPI + SSE Implementation (Python)

### Install

```bash
pip install fastapi uvicorn anthropic python-dotenv
```

### Full server

```python
# server.py
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# --- your existing agent_loop import ---
# from agent_loop import agent_loop, AgentEvent

# ── Inline stub so this file is self-contained ──────────────────────────────
import anthropic

MODEL = "claude-sonnet-4-6"
client = anthropic.Anthropic()


@dataclass
class AgentEvent:
    type: str          # "text" | "tool_use" | "tool_result" | "done" | "error"
    content: str = ""
    tool_name: str = ""
    tool_input: dict = None
    cost_usd: float = 0.0

    def __post_init__(self):
        if self.tool_input is None:
            self.tool_input = {}


async def agent_loop(
    messages: list[dict],
    system: str,
    tools: list[dict],
) -> AsyncGenerator[AgentEvent, None]:
    """
    Minimal async generator that wraps Anthropic streaming.
    Replace with your real agent_loop from agent-loop.md.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system,
        messages=messages,
        tools=tools,
        stream=True,
    )
    full_text = ""
    for event in response:
        if event.type == "content_block_delta":
            delta = event.delta
            if hasattr(delta, "text"):
                full_text += delta.text
                yield AgentEvent(type="text", content=delta.text)
        elif event.type == "message_stop":
            usage = getattr(response, "usage", None)
            cost = 0.0
            if usage:
                cost = (usage.input_tokens * 3 + usage.output_tokens * 15) / 1_000_000
            yield AgentEvent(type="done", cost_usd=cost)
# ─────────────────────────────────────────────────────────────────────────────


# ── Session store ─────────────────────────────────────────────────────────────

@dataclass
class Session:
    session_id: str
    messages: list[dict]
    total_cost_usd: float = 0.0


class SessionStore:
    """
    In-memory for development.
    Swap the dict for Redis calls in production (see note at bottom).
    """

    def __init__(self):
        self._store: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._store:
            self._store[session_id] = Session(session_id=session_id, messages=[])
        return self._store[session_id]

    def append_messages(self, session_id: str, messages: list[dict]) -> None:
        session = self.get_or_create(session_id)
        session.messages.extend(messages)

    def get_messages(self, session_id: str) -> list[dict]:
        if session_id not in self._store:
            return []
        return self._store[session_id].messages

    def add_cost(self, session_id: str, cost: float) -> None:
        session = self.get_or_create(session_id)
        session.total_cost_usd += cost

    def get_cost(self, session_id: str) -> float:
        if session_id not in self._store:
            return 0.0
        return self._store[session_id].total_cost_usd

    def delete(self, session_id: str) -> bool:
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False


sessions = SessionStore()


# ── Rate limiter ──────────────────────────────────────────────────────────────

import time
from collections import defaultdict


class RateLimiter:
    def __init__(
        self,
        max_requests_per_minute: int = 20,
        max_cost_per_day_usd: float = 2.0,
    ):
        self.max_rpm = max_requests_per_minute
        self.max_daily_cost = max_cost_per_day_usd
        self._request_times: dict[str, list[float]] = defaultdict(list)
        self._daily_cost_reset: dict[str, float] = {}

    def check(self, session_id: str, current_cost: float) -> tuple[bool, str]:
        now = time.time()

        # sliding window: keep only last 60s
        window = [t for t in self._request_times[session_id] if now - t < 60]
        self._request_times[session_id] = window

        if len(window) >= self.max_rpm:
            return False, f"Rate limit: {self.max_rpm} requests/minute exceeded"

        # daily cost reset
        reset_at = self._daily_cost_reset.get(session_id, 0)
        if now - reset_at > 86400:
            self._daily_cost_reset[session_id] = now
            sessions.get_or_create(session_id).total_cost_usd = 0.0

        if current_cost >= self.max_daily_cost:
            return False, f"Daily cost limit ${self.max_daily_cost:.2f} reached"

        self._request_times[session_id].append(now)
        return True, "ok"


limiter = RateLimiter()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = "You are a helpful assistant."
TOOLS: list[dict] = []   # add your tool schemas here


@app.post("/chat")
async def chat(request: Request) -> StreamingResponse:
    # ── parse body ────────────────────────────────────────────────────────────
    body = await request.json()
    user_message: str = body.get("message", "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message is required")

    # ── session ───────────────────────────────────────────────────────────────
    session_id = request.headers.get("X-Session-ID") or str(uuid.uuid4())
    session = sessions.get_or_create(session_id)

    # ── rate limit ────────────────────────────────────────────────────────────
    allowed, reason = limiter.check(session_id, session.total_cost_usd)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    # ── build message history ─────────────────────────────────────────────────
    sessions.append_messages(session_id, [{"role": "user", "content": user_message}])
    messages = sessions.get_messages(session_id)

    # ── SSE generator ─────────────────────────────────────────────────────────
    async def generate() -> AsyncGenerator[str, None]:
        assistant_text = ""
        try:
            async for event in agent_loop(messages, SYSTEM_PROMPT, TOOLS):
                if event.type == "text":
                    assistant_text += event.content
                if event.type == "done":
                    sessions.add_cost(session_id, event.cost_usd)

                payload = asdict(event)
                payload["session_id"] = session_id
                yield f"data: {json.dumps(payload)}\n\n"

        except Exception as e:
            error_event = {"type": "error", "content": str(e), "session_id": session_id}
            yield f"data: {json.dumps(error_event)}\n\n"

        finally:
            if assistant_text:
                sessions.append_messages(
                    session_id, [{"role": "assistant", "content": assistant_text}]
                )
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Session-ID": session_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    msgs = sessions.get_messages(session_id)
    if not msgs:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "messages": msgs,
        "cost_usd": sessions.get_cost(session_id),
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = sessions.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}
```

### SSE event format

Every `yield` writes one SSE event. The format is strict:

```
data: {"type": "text", "content": "Hello"}\n\n
data: {"type": "tool_use", "tool_name": "read_file", "tool_input": {"path": "/tmp/x.txt"}}\n\n
data: {"type": "tool_result", "content": "file contents here"}\n\n
data: {"type": "done", "cost_usd": 0.0012}\n\n
data: [DONE]\n\n
```

The double newline `\n\n` is mandatory — it signals the end of one event.
The `[DONE]` sentinel matches the OpenAI convention; clients can look for it.

---

## 3. Bun/TypeScript + Hono Implementation

### Install

```bash
bun add hono @anthropic-ai/sdk
```

### Full server

```typescript
// server.ts
import { Hono } from "hono";
import { cors } from "hono/cors";
import { stream } from "hono/streaming";
import Anthropic from "@anthropic-ai/sdk";

const MODEL = "claude-sonnet-4-6";
const client = new Anthropic();

// ── Types ─────────────────────────────────────────────────────────────────────

interface AgentEvent {
  type: "text" | "tool_use" | "tool_result" | "done" | "error";
  content?: string;
  tool_name?: string;
  tool_input?: Record<string, unknown>;
  cost_usd?: number;
  session_id?: string;
}

interface Session {
  messages: Anthropic.MessageParam[];
  totalCostUsd: number;
  lastActiveAt: number;
}

// ── Session store ─────────────────────────────────────────────────────────────

const sessionStore = new Map<string, Session>();

function getOrCreateSession(sessionId: string): Session {
  if (!sessionStore.has(sessionId)) {
    sessionStore.set(sessionId, {
      messages: [],
      totalCostUsd: 0,
      lastActiveAt: Date.now(),
    });
  }
  const session = sessionStore.get(sessionId)!;
  session.lastActiveAt = Date.now();
  return session;
}

// ── Agent loop ────────────────────────────────────────────────────────────────

async function* agentLoop(
  messages: Anthropic.MessageParam[],
  system: string
): AsyncGenerator<AgentEvent> {
  const stream = await client.messages.stream({
    model: MODEL,
    max_tokens: 4096,
    system,
    messages,
  });

  for await (const event of stream) {
    if (
      event.type === "content_block_delta" &&
      event.delta.type === "text_delta"
    ) {
      yield { type: "text", content: event.delta.text };
    }
  }

  const final = await stream.finalMessage();
  const cost =
    (final.usage.input_tokens * 3 + final.usage.output_tokens * 15) /
    1_000_000;
  yield { type: "done", cost_usd: cost };
}

// ── Hono app ──────────────────────────────────────────────────────────────────

const app = new Hono();

app.use(
  "/*",
  cors({
    origin: ["http://localhost:3000", "http://localhost:5173"],
    allowHeaders: ["Content-Type", "X-Session-ID"],
    exposeHeaders: ["X-Session-ID"],
  })
);

app.post("/chat", async (c) => {
  const body = await c.req.json<{ message: string }>();
  if (!body.message?.trim()) {
    return c.json({ error: "message is required" }, 400);
  }

  const sessionId = c.req.header("X-Session-ID") ?? crypto.randomUUID();
  const session = getOrCreateSession(sessionId);

  session.messages.push({ role: "user", content: body.message });

  return stream(c, async (s) => {
    let assistantText = "";
    try {
      for await (const event of agentLoop(
        session.messages,
        "You are a helpful assistant."
      )) {
        if (event.type === "text") assistantText += event.content ?? "";
        if (event.type === "done") session.totalCostUsd += event.cost_usd ?? 0;

        const payload: AgentEvent = { ...event, session_id: sessionId };
        await s.write(`data: ${JSON.stringify(payload)}\n\n`);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      await s.write(
        `data: ${JSON.stringify({ type: "error", content: msg })}\n\n`
      );
    } finally {
      if (assistantText) {
        session.messages.push({ role: "assistant", content: assistantText });
      }
      await s.write("data: [DONE]\n\n");
    }
  });
});

app.get("/sessions/:sessionId", (c) => {
  const { sessionId } = c.req.param();
  const session = sessionStore.get(sessionId);
  if (!session) return c.json({ error: "Session not found" }, 404);
  return c.json({
    session_id: sessionId,
    messages: session.messages,
    cost_usd: session.totalCostUsd,
  });
});

app.delete("/sessions/:sessionId", (c) => {
  const { sessionId } = c.req.param();
  if (!sessionStore.delete(sessionId))
    return c.json({ error: "Session not found" }, 404);
  return c.json({ deleted: sessionId });
});

export default { port: 8000, fetch: app.fetch };
```

### Run

```bash
bun run server.ts
```

---

## 4. Multi-User Session Management

For production, swap the in-memory dict for Redis:

```python
# session_store_redis.py
import json
import redis.asyncio as redis

class RedisSessionStore:
    def __init__(self, url: str = "redis://localhost:6379"):
        self.r = redis.from_url(url)
        self.ttl = 60 * 60 * 24  # 24h session TTL

    async def get_or_create(self, session_id: str) -> dict:
        raw = await self.r.get(f"session:{session_id}")
        if raw:
            return json.loads(raw)
        session = {"session_id": session_id, "messages": [], "total_cost_usd": 0.0}
        await self.r.setex(f"session:{session_id}", self.ttl, json.dumps(session))
        return session

    async def append_messages(self, session_id: str, messages: list[dict]) -> None:
        session = await self.get_or_create(session_id)
        session["messages"].extend(messages)
        await self.r.setex(f"session:{session_id}", self.ttl, json.dumps(session))

    async def get_messages(self, session_id: str) -> list[dict]:
        raw = await self.r.get(f"session:{session_id}")
        if not raw:
            return []
        return json.loads(raw)["messages"]

    async def add_cost(self, session_id: str, cost: float) -> None:
        session = await self.get_or_create(session_id)
        session["total_cost_usd"] += cost
        await self.r.setex(f"session:{session_id}", self.ttl, json.dumps(session))

    async def get_cost(self, session_id: str) -> float:
        raw = await self.r.get(f"session:{session_id}")
        return json.loads(raw)["total_cost_usd"] if raw else 0.0

    async def delete(self, session_id: str) -> bool:
        return bool(await self.r.delete(f"session:{session_id}"))
```

---

## 5. React Frontend (SSE Consumption)

### With EventSource (simple, no request body)

```typescript
// useAgentStream.ts
import { useState, useCallback } from "react";

export function useAgentStream(baseUrl: string) {
  const [output, setOutput] = useState("");
  const [streaming, setStreaming] = useState(false);

  const send = useCallback(
    async (message: string, sessionId: string) => {
      setStreaming(true);
      setOutput("");

      // SSE via fetch + ReadableStream (supports POST + headers)
      const res = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Session-ID": sessionId,
        },
        body: JSON.stringify({ message }),
      });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") { setStreaming(false); return; }

          const event = JSON.parse(raw);
          if (event.type === "text") {
            setOutput((prev) => prev + event.content);
          }
        }
      }
      setStreaming(false);
    },
    [baseUrl]
  );

  return { output, streaming, send };
}
```

### Usage in a component

```tsx
// Chat.tsx
import { useState } from "react";
import { useAgentStream } from "./useAgentStream";

const SESSION_ID = crypto.randomUUID();  // persist in localStorage for real apps

export function Chat() {
  const [input, setInput] = useState("");
  const { output, streaming, send } = useAgentStream("http://localhost:8000");

  return (
    <div>
      <pre>{output}</pre>
      <input value={input} onChange={(e) => setInput(e.target.value)} />
      <button disabled={streaming} onClick={() => send(input, SESSION_ID)}>
        {streaming ? "Thinking..." : "Send"}
      </button>
    </div>
  );
}
```

---

## 6. Rate Limiting and Cost Controls

The `RateLimiter` class shown in section 2 covers the basics. For production:

```python
# rate_limiter.py
from dataclasses import dataclass, field
import time
from collections import defaultdict

@dataclass
class RateLimiter:
    max_requests_per_minute: int = 20
    max_cost_per_day_usd: float = 2.0
    _request_times: dict = field(default_factory=lambda: defaultdict(list))
    _day_start: dict = field(default_factory=dict)
    _day_cost: dict = field(default_factory=lambda: defaultdict(float))

    def check(self, session_id: str) -> tuple[bool, str]:
        now = time.time()

        # sliding window
        self._request_times[session_id] = [
            t for t in self._request_times[session_id] if now - t < 60
        ]
        if len(self._request_times[session_id]) >= self.max_requests_per_minute:
            retry_after = 60 - (now - self._request_times[session_id][0])
            return False, f"Rate limited. Retry after {retry_after:.0f}s"

        # daily cost reset
        if now - self._day_start.get(session_id, 0) > 86400:
            self._day_start[session_id] = now
            self._day_cost[session_id] = 0.0

        if self._day_cost[session_id] >= self.max_cost_per_day_usd:
            return False, f"Daily cost cap ${self.max_cost_per_day_usd:.2f} reached"

        self._request_times[session_id].append(now)
        return True, "ok"

    def record_cost(self, session_id: str, cost: float) -> None:
        self._day_cost[session_id] += cost
```

---

## 7. CORS + API Key Auth

```python
# Add to server.py
from fastapi import Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_KEYS = {"dev-key-abc123", "prod-key-xyz789"}  # load from env in prod

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

# Add to your route:
@app.post("/chat")
async def chat(request: Request, key: str = Depends(verify_api_key)):
    ...
```

Client sends: `X-API-Key: dev-key-abc123` in every request.

For user-facing apps, issue JWTs instead. For internal services, a static key
loaded from an env variable is fine.

```python
import os
VALID_KEYS = set(os.environ.get("API_KEYS", "").split(","))
```

---

## 8. Deployment Quick-Start

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

`--workers 1` is intentional: uvicorn workers don't share in-memory session
stores. Use Redis + multiple workers, or a single worker for low-traffic deploys.

### requirements.txt

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
anthropic>=0.28.0
python-dotenv>=1.0.0
redis>=5.0.0
```

### Run locally

```bash
ANTHROPIC_API_KEY=sk-... uvicorn server:app --reload --port 8000
```

### Run in production

```bash
docker build -t agent-api .
docker run -d \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e API_KEYS=prod-key-xyz789 \
  --name agent-api \
  agent-api
```

### nginx config snippet (SSE-safe)

```nginx
location /chat {
    proxy_pass         http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header   Connection "";
    proxy_buffering    off;          # critical for SSE
    proxy_cache        off;
    proxy_read_timeout 300s;
}
```

`proxy_buffering off` is mandatory. Without it, nginx holds the entire response
body before forwarding — your SSE events arrive all at once at the end.
