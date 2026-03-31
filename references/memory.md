# Memory & Session Persistence

Two mechanisms that make an agent feel "smart across sessions":

- **Session persistence** — save the full conversation to disk so the user can resume exactly where they left off
- **Long-term memory** — structured facts about the user, project, and preferences that survive indefinitely and get injected into every system prompt

---

## Session Persistence

A session is a JSON file stored in `~/.agent/sessions/<session_id>.json`.

### SessionManager

```python
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone


class SessionManager:
    def __init__(self, sessions_dir: str = "~/.agent/sessions"):
        self.dir = Path(sessions_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def save(self, session_id: str, messages: list[dict], metadata: dict) -> None:
        data = {
            "session_id": session_id,
            "created_at": metadata.get("created_at", datetime.now(timezone.utc).isoformat()),
            "last_message_at": datetime.now(timezone.utc).isoformat(),
            "turn_count": metadata.get("turn_count", len([m for m in messages if m["role"] == "user"])),
            "total_cost_usd": metadata.get("total_cost_usd", 0.0),
            "messages": messages,
        }
        self._path(session_id).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, session_id: str) -> tuple[list[dict], dict]:
        raw = json.loads(self._path(session_id).read_text(encoding="utf-8"))
        messages = raw.pop("messages")
        return messages, raw  # (messages, metadata)

    def list_sessions(self) -> list[dict]:
        sessions = []
        for f in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            raw = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": raw["session_id"],
                "created_at": raw["created_at"],
                "last_message_at": raw["last_message_at"],
                "turn_count": raw["turn_count"],
                "total_cost_usd": raw["total_cost_usd"],
                "preview": next(
                    (m["content"][:80] for m in raw["messages"] if m["role"] == "user"),
                    "(empty)"
                ),
            })
        return sessions

    def delete(self, session_id: str) -> None:
        p = self._path(session_id)
        if p.exists():
            p.unlink()

    def new_id(self) -> str:
        return str(uuid.uuid4())[:8]
```

### Saved session format

```json
{
  "session_id": "a3f8c1b2",
  "created_at": "2026-03-31T09:00:00+00:00",
  "last_message_at": "2026-03-31T09:47:12+00:00",
  "turn_count": 12,
  "total_cost_usd": 0.42,
  "messages": [
    {"role": "user", "content": "Refactor the auth module"},
    {"role": "assistant", "content": [{"type": "text", "text": "Starting with..."}]}
  ]
}
```

### `/resume` command

```python
async def handle_resume_command(session_manager: SessionManager, agent_state: dict) -> str:
    sessions = session_manager.list_sessions()
    if not sessions:
        return "No saved sessions found."

    lines = ["Saved sessions:\n"]
    for i, s in enumerate(sessions):
        cost = f"${s['total_cost_usd']:.2f}"
        lines.append(f"  [{i+1}] {s['session_id']}  {s['turn_count']} turns  {cost}  — {s['preview']}")
    lines.append("\nEnter number to resume (or 'cancel'):")
    print("\n".join(lines))

    choice = input("> ").strip()
    if choice.lower() == "cancel" or not choice.isdigit():
        return "Resume cancelled."

    idx = int(choice) - 1
    if idx < 0 or idx >= len(sessions):
        return "Invalid selection."

    session_id = sessions[idx]["session_id"]
    messages, metadata = session_manager.load(session_id)
    agent_state["messages"] = messages
    agent_state["session_id"] = session_id
    agent_state["metadata"] = metadata
    return f"Resumed session {session_id} ({metadata['turn_count']} turns, ${metadata['total_cost_usd']:.2f})."
```

---

## Long-Term Memory (memdir pattern)

Each memory is a `.md` file with YAML frontmatter. The directory lives at `~/.agent/memory/`.

### Memory file format

```markdown
---
name: user-role
type: user
description: User is an AI instructor who teaches TypeScript and Python
---
Ahmad is an AI/ML instructor at a training company.
He uses Bun (not Node.js), prefers concise answers,
and teaches with German technical terminology.
```

### Memory types

| Type | What it stores |
|------|---------------|
| `user` | Identity, expertise, communication preferences |
| `feedback` | Corrections the user gave, validated approaches |
| `project` | Active work, goals, file locations, deadlines |
| `reference` | Pointers to external systems, APIs, credentials paths |

### MemoryManager

```python
import re
from pathlib import Path


class MemoryManager:
    def __init__(self, memory_dir: str = "~/.agent/memory"):
        self.dir = Path(memory_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)

    def save_memory(self, name: str, type: str, description: str, content: str) -> None:
        safe_name = re.sub(r"[^\w\-]", "-", name.lower())
        text = f"---\nname: {safe_name}\ntype: {type}\ndescription: {description}\n---\n{content.strip()}\n"
        (self.dir / f"{safe_name}.md").write_text(text, encoding="utf-8")

    def load_all(self) -> list[dict]:
        memories = []
        for f in self.dir.glob("*.md"):
            raw = f.read_text(encoding="utf-8")
            parts = raw.split("---", 2)
            if len(parts) < 3:
                continue
            frontmatter = {}
            for line in parts[1].strip().splitlines():
                if ": " in line:
                    k, v = line.split(": ", 1)
                    frontmatter[k.strip()] = v.strip()
            memories.append({**frontmatter, "content": parts[2].strip()})
        return memories

    def find_relevant(self, query: str, messages: list[dict], top_n: int = 5) -> list[dict]:
        all_memories = self.load_all()
        recent_text = " ".join(
            m["content"] if isinstance(m["content"], str) else ""
            for m in messages[-6:]
        ).lower()
        combined = (query + " " + recent_text).lower()

        def score(mem: dict) -> int:
            keywords = re.findall(r"\w+", mem.get("description", "").lower())
            return sum(1 for kw in keywords if len(kw) > 3 and kw in combined)

        ranked = sorted(all_memories, key=score, reverse=True)
        return [m for m in ranked[:top_n] if score(m) > 0]

    def build_memory_prompt(self, query: str, messages: list[dict]) -> str:
        relevant = self.find_relevant(query, messages)
        if not relevant:
            return ""
        lines = ["## Memories"]
        for m in relevant:
            lines.append(f"- [{m.get('type', 'general')}] {m['content']}")
        return "\n".join(lines)

    def delete_memory(self, name: str) -> bool:
        safe_name = re.sub(r"[^\w\-]", "-", name.lower())
        p = self.dir / f"{safe_name}.md"
        if p.exists():
            p.unlink()
            return True
        return False

    def clear_all(self) -> int:
        count = 0
        for f in self.dir.glob("*.md"):
            f.unlink()
            count += 1
        return count
```

### TypeScript equivalent (key classes)

```typescript
import fs from "fs";
import path from "path";
import os from "os";

interface Memory {
  name: string;
  type: string;
  description: string;
  content: string;
}

class MemoryManager {
  private dir: string;

  constructor(memoryDir = path.join(os.homedir(), ".agent", "memory")) {
    this.dir = memoryDir;
    fs.mkdirSync(this.dir, { recursive: true });
  }

  saveMemory(name: string, type: string, description: string, content: string): void {
    const safeName = name.toLowerCase().replace(/[^\w-]/g, "-");
    const text = `---\nname: ${safeName}\ntype: ${type}\ndescription: ${description}\n---\n${content.trim()}\n`;
    fs.writeFileSync(path.join(this.dir, `${safeName}.md`), text, "utf-8");
  }

  loadAll(): Memory[] {
    return fs.readdirSync(this.dir)
      .filter(f => f.endsWith(".md"))
      .map(f => {
        const raw = fs.readFileSync(path.join(this.dir, f), "utf-8");
        const parts = raw.split("---");
        if (parts.length < 3) return null;
        const fm: Record<string, string> = {};
        for (const line of parts[1].trim().split("\n")) {
          const [k, ...v] = line.split(": ");
          if (k && v.length) fm[k.trim()] = v.join(": ").trim();
        }
        return { ...fm, content: parts[2].trim() } as Memory;
      })
      .filter(Boolean) as Memory[];
  }

  buildMemoryPrompt(query: string): string {
    const memories = this.loadAll();
    if (!memories.length) return "";
    const lines = memories.map(m => `- [${m.type}] ${m.content}`);
    return `## Memories\n${lines.join("\n")}`;
  }
}
```

---

## Auto-Memory Extraction

At the end of a session, ask the LLM to pull memorable facts out of the conversation:

```python
async def extract_memories(messages: list[dict], api_client) -> list[dict]:
    extraction_prompt = """Review this conversation and extract facts worth remembering long-term.

Return a JSON array. Each item must have:
- name: short slug (e.g. "user-prefers-bun")
- type: one of user | feedback | project | reference
- description: one sentence (used for retrieval matching)
- content: the actual fact in 1-3 sentences

Only extract facts that will change how you respond in future sessions.
Skip anything transient or session-specific.

Return valid JSON only. No explanation text."""

    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content'] if isinstance(m['content'], str) else '[tool use]'}"
        for m in messages[-20:]
    )

    response = await api_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=extraction_prompt,
        messages=[{"role": "user", "content": conversation_text}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
```

---

## Integration with the Agent Loop

Inject memory into the system prompt before each API call:

```python
# In your agent loop, before api_client.messages.create(...)
system_prompt = build_system_prompt(tools)           # returns list[str]
memory_section = memory_manager.build_memory_prompt(user_message, messages)
if memory_section:
    system_prompt.append(memory_section)             # appended as dynamic suffix

response = await api_client.messages.create(
    model="claude-opus-4-5",
    system="\n\n".join(system_prompt),
    messages=messages,
    tools=tools.to_api_format(),
    max_tokens=8096,
)
```

Auto-extract after the loop completes:

```python
# After agent_loop() returns
new_memories = await extract_memories(messages, api_client)
for m in new_memories:
    memory_manager.save_memory(
        name=m["name"],
        type=m["type"],
        description=m["description"],
        content=m["content"],
    )
if new_memories:
    print(f"Saved {len(new_memories)} new memories.")
```

---

## `/memory` Slash Command

```python
async def handle_memory_command(args: str, memory_manager: MemoryManager) -> str:
    parts = args.strip().split(maxsplit=2)
    subcmd = parts[0].lower() if parts else "list"

    if subcmd == "list":
        memories = memory_manager.load_all()
        if not memories:
            return "No memories stored."
        lines = [f"[{m.get('type', '?')}] {m.get('name', '?')}: {m.get('description', '')}"
                 for m in memories]
        return "\n".join(lines)

    if subcmd == "add":
        if len(parts) < 3:
            return "Usage: /memory add <name> <content>"
        name, content = parts[1], parts[2]
        memory_manager.save_memory(
            name=name,
            type="user",
            description=f"Manually added memory: {name}",
            content=content,
        )
        return f"Memory '{name}' saved."

    if subcmd == "delete":
        if len(parts) < 2:
            return "Usage: /memory delete <name>"
        deleted = memory_manager.delete_memory(parts[1])
        return f"Memory '{parts[1]}' deleted." if deleted else f"Memory '{parts[1]}' not found."

    if subcmd == "clear":
        count = memory_manager.clear_all()
        return f"Cleared {count} memories."

    return f"Unknown subcommand '{subcmd}'. Use: list | add | delete | clear"
```

---

## Wiring It All Together

```python
# Startup
session_manager = SessionManager()
memory_manager = MemoryManager()

session_id = session_manager.new_id()
messages: list[dict] = []
metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "total_cost_usd": 0.0}

# Main REPL
while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.startswith("/resume"):
        print(await handle_resume_command(session_manager, agent_state))
        continue
    if user_input.startswith("/memory"):
        print(await handle_memory_command(user_input[7:].strip(), memory_manager))
        continue

    messages.append({"role": "user", "content": user_input})
    async for event in agent_loop(messages, tools, permissions, context_manager,
                                   build_system_prompt(tools, memory_manager, user_input),
                                   api_client):
        # render events...
        pass

    session_manager.save(session_id, messages, metadata)

# On exit
new_memories = await extract_memories(messages, api_client)
for m in new_memories:
    memory_manager.save_memory(**m)
```
