import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Union

import aiohttp
from qwen_agent.tools.base import BaseTool, register_tool


SERPER_KEY = os.environ.get("SERPER_KEY_ID", "")


def contains_chinese_basic(text: str) -> bool:
    return any("\u4E00" <= ch <= "\u9FFF" for ch in text)


@register_tool("google_scholar", allow_overwrite=True)
class ScholarSearch(BaseTool):
    name = "google_scholar"
    description = "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries."
    parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string", "description": "The search query."},
                    "minItems": 1,
                    "description": "The list of search queries for Google Scholar."
                },
            },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.max_concurrency = 16

    def _build_payload(self, query: str) -> Dict[str, Any]:
        if contains_chinese_basic(query):
            return {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
        return {"q": query, "location": "United States", "gl": "us", "hl": "en"}

    def _format_results(self, query: str, results: Dict[str, Any]) -> str:
        try:
            if "organic" not in results:
                raise Exception

            web_snippets: List[str] = []
            idx = 0
            for page in results.get("organic", []):
                idx += 1

                date_published = f"\nDate published: {page['date']}" if "date" in page else ""
                source = f"\nSource: {page['source']}" if "source" in page else ""
                snippet = f"\n{page['snippet']}" if "snippet" in page else ""

                title = page.get("title", "")
                link = page.get("link", "")
                line = f"{idx}. [{title}]({link}){date_published}{source}\n{snippet}"
                line = line.replace("Your browser can't play this video.", "")
                web_snippets.append(line)

            content = (
                f"A Google scholar search for '{query}' found {len(web_snippets)} results:\n\n"
                "## Scholar Results\n" + "\n\n".join(web_snippets)
            )
            return content
        except Exception:
            return f"No results found for '{query}'. Try with a more general query."

    async def _post_with_retries(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        retries: int = 5,
        base_sleep: float = 0.5,
    ) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for i in range(retries):
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                    return json.loads(text)
            except Exception as e:
                last_err = e
                if i == retries - 1:
                    break
                await asyncio.sleep(base_sleep * (2 ** i))
        raise RuntimeError(f"Serper request failed after {retries} retries: {last_err}")

    async def _search_one(self, session: aiohttp.ClientSession, query: str, sem: asyncio.Semaphore) -> str:
        if not SERPER_KEY:
            return "[scholar_search] SERPER_KEY_ID is not set in environment variables."

        url = "https://google.serper.dev/scholar"
        headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
        payload = self._build_payload(query)

        async with sem:
            try:
                results = await self._post_with_retries(session, url, payload, headers)
                return self._format_results(query, results)
            except Exception:
                return "Google scholar search Timeout, return None, Please try again later."

    async def _run_queries(self, queries: List[str]) -> str:
        timeout = aiohttp.ClientTimeout(total=60)
        sem = asyncio.Semaphore(self.max_concurrency)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [self._search_one(session, q, sem) for q in queries]
            outputs = await asyncio.gather(*tasks, return_exceptions=False)

        return "\n=======\n".join(outputs)

    async def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except Exception:
            return "[scholar_search] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            queries = [query]
        else:
            if not isinstance(query, list) or not all(isinstance(x, str) for x in query):
                return "[scholar_search] Invalid 'query': must be a string or a list of strings"
            queries = query

        return await self._run_queries(queries)