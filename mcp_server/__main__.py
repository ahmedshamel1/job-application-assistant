import os
import re

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# This is the MCP server — it exposes tools that agents can call.
# Think of it exactly like your .NET MCP server: you define tools here,
# and any MCP client (agent) connects, gets the tool list, and calls them.
# The difference from .NET: transport here is SSE (HTTP streaming) on port 3000.

load_dotenv()

mcp = FastMCP("job-search-tools", host="0.0.0.0", port=3000)

_TAVILY_URL = "https://api.tavily.com/search"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web for a query. Returns top 5 results with titles, URLs, and snippets."""
    api_key = os.getenv("TAVILY_API_KEY")
    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.post(
            _TAVILY_URL,
            json={"api_key": api_key, "query": query, "max_results": 5},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   URL: {r['url']}")
        lines.append(f"   {r['content']}")
    return "\n\n".join(lines)


@mcp.tool()
async def fetch_page(url: str) -> str:
    """Fetch a web page and return its text content (HTML tags stripped). Capped at 5000 chars."""
    # verify=False: Windows Python often fails SSL cert checks on corporate/dev machines
    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True, verify=False) as client:
        resp = await client.get(url, timeout=15)
        resp.raise_for_status()

        html = resp.text
        # Remove script and style blocks entirely — they're noise
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL)
        # Strip remaining tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > 5000:
            return text[:5000] + "\n\n[truncated — page continues]"
        return text


def main():
    # transport="sse" means the server speaks MCP over HTTP/SSE.
    # This is exactly the protocol your sibling repo's mcp.py connects to via sse_client().
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()
