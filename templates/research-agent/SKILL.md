# Research Agent Template

Research and analysis agent with web search, web fetch, and file tools.

## Project Structure

```
my-research-agent/
  main.py                # Entry point
  agent/
    __init__.py
    loop.py              # Agent loop
    tools.py             # Web search + web fetch + file tools
    permissions.py       # Permission system
    context.py           # Context management
    prompt.py            # Research-focused system prompt
    report.py            # Report generation
  requirements.txt
```

## System Prompt (Research-Focused)

```python
RESEARCH_SYSTEM_PROMPT = """You are a research assistant specializing in thorough analysis.

## Your Process
1. Understand the research question
2. Search for relevant information using web_search
3. Fetch and read promising URLs using web_fetch
4. Synthesize findings into a structured report
5. Cite sources with URLs

## Report Format
# [Research Topic]

## Summary
[2-3 sentence executive summary]

## Key Findings
1. [Finding with citation](URL)
2. [Finding with citation](URL)

## Detailed Analysis
[In-depth analysis with subsections]

## Sources
- [Source 1](URL)
- [Source 2](URL)

## Research Tools
- web_search: Search the internet for information
- web_fetch: Fetch and read web page content
- read_file: Read local files for context
- write_file: Save research reports"""
```

## Custom Tools

### WebSearchTool

```python
class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the internet for information."
    input_schema = {
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    }
    is_read_only = True
    is_concurrency_safe = True

    async def call(self, input: dict, context: dict) -> ToolResult:
        # Implement using your preferred search API
        # Options: SerpAPI, Brave Search, Tavily, or Anthropic's built-in
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": input["query"], "count": 10},
                headers={"X-Subscription-Token": os.environ.get("BRAVE_API_KEY", "")},
            )
            data = resp.json()
            results = []
            for r in data.get("web", {}).get("results", [])[:5]:
                results.append(f"- {r['title']}: {r.get('description', '')}\n  URL: {r['url']}")
            return ToolResult(success=True, content="\n".join(results))
```

### WebFetchTool

```python
class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch and extract text content from a URL."
    input_schema = {
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
        },
        "required": ["url"],
    }
    is_read_only = True
    is_concurrency_safe = True

    async def call(self, input: dict, context: dict) -> ToolResult:
        import httpx
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(input["url"], timeout=30)
            # Basic HTML to text conversion
            text = resp.text
            # Strip HTML tags (simple approach)
            import re
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return ToolResult(success=True, content=text[:10000])
```

## Usage

```bash
python main.py "Research the current state of quantum computing in 2026"
```
