#!/usr/bin/env python3
"""Validate an agent project has all required components."""

import sys
import os
from pathlib import Path

REQUIRED_FILES = {
    "agent_loop": ["loop.py", "agent.py", "main.py", "core.py"],
    "tools": ["tools.py", "tool_registry.py", "agent.py"],
    "permissions": ["permissions.py", "agent.py"],
    "context": ["context.py", "agent.py"],
    "system_prompt": ["prompt.py", "system_prompt.py", "agent.py"],
}

CHECKS = {
    "agent_loop": [
        ("while True", "Main loop (while True)"),
        ("tool_use", "Tool use detection"),
        ("tool_result", "Tool result handling"),
        ("async", "Async support"),
    ],
    "tools": [
        ("class Tool", "Tool base class/interface"),
        ("async def call", "Tool call method"),
        ("input_schema", "Input schema definition"),
        ("bash", "Bash tool implementation"),
        ("read_file", "File read tool"),
    ],
    "permissions": [
        ("deny", "Deny rule handling"),
        ("allow", "Allow rule handling"),
        ("ask", "Ask/user prompt handling"),
    ],
    "context": [
        ("compact", "Context compaction"),
        ("token", "Token counting/awareness"),
    ],
    "system_prompt": [
        ("system", "System prompt construction"),
        ("tool", "Tool descriptions in prompt"),
    ],
}

def find_agent_files(project_dir: Path) -> list[Path]:
    files = []
    for ext in ("*.py", "*.ts", "*.js"):
        files.extend(project_dir.rglob(ext))
    return [f for f in files if not any(p in str(f) for p in [
        "node_modules", "__pycache__", ".git", "venv"
    ])]

def check_component(files: list[Path], keywords: list[tuple[str, str]]) -> list[tuple[str, bool]]:
    results = []
    for keyword, description in keywords:
        found = False
        for f in files:
            try:
                if keyword.lower() in f.read_text(encoding="utf-8", errors="replace").lower():
                    found = True
                    break
            except Exception:
                continue
        results.append((description, found))
    return results

def validate(project_dir: str) -> bool:
    root = Path(project_dir).resolve()
    if not root.exists():
        print(f"ERROR: Directory not found: {root}")
        return False

    print(f"=== Agent Validation: {root.name} ===\n")

    files = find_agent_files(root)
    if not files:
        print("ERROR: No source files found (.py, .ts, .js)")
        return False

    print(f"Found {len(files)} source files\n")

    all_pass = True
    total_found = 0
    total_checks = 0

    for component, keywords in CHECKS.items():
        results = check_component(files, keywords)
        found = sum(1 for _, f in results if f)
        total_found += found
        total_checks += len(results)

        status = "PASS" if found == len(results) else "WARN"
        if status == "WARN":
            all_pass = False

        print(f"[{status}] {component}: {found}/{len(results)} checks")
        for desc, ok in results:
            icon = "+" if ok else "-"
            print(f"  [{icon}] {desc}")
        print()

    print(f"=== Summary: {total_found}/{total_checks} checks passed ===")

    if all_pass:
        print("All critical components found. Agent is production-ready.")
    else:
        print("Some components missing. See warnings above.")

    return all_pass

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    ok = validate(target)
    sys.exit(0 if ok else 1)
