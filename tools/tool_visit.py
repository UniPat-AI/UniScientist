import os
import json
import time
import asyncio
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
import tiktoken
from openai import AsyncOpenAI
from qwen_agent.tools.base import BaseTool, register_tool


VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 3600))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 95000))

JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "")
SUMMARY_MODEL_NAME = os.getenv("SUMMARY_MODEL_NAME", "")

MAX_CONCURRENCY = int(os.getenv("VISIT_MAX_CONCURRENCY", 16))
PER_URL_DEADLINE_SEC = int(os.getenv("VISIT_PER_URL_DEADLINE_SEC", 3600))


SYSTEM_PROMPT = """# Response Formats
## useful_information
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""

USER_PROMPT_TEMPLATE = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{raw_response}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
""".strip()


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def _pick_jina_key() -> Optional[str]:
    keys = [k.strip() for k in (JINA_API_KEYS or "").split(",") if k.strip()]
    if not keys:
        return None
    return random.choice(keys)


def _json_loads_loose(s: str) -> Any:
    s = (s or "").strip()
    s = s.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(s)
    except Exception:
        left = s.find("{")
        right = s.rfind("}")
        if left != -1 and right != -1 and left <= right:
            return json.loads(s[left : right + 1])
        raise


@register_tool("visit", allow_overwrite=True)
class Visit(BaseTool):
    name = "visit"
    description = "Visit webpage(s) and return the summary of the content."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
            },
            "goal": {"type": "string", "description": "The goal of the visit for webpage(s)."},
        },
        "required": ["url", "goal"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.max_concurrency = MAX_CONCURRENCY

    async def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except Exception:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        if isinstance(url, str):
            return (await self.readpage_jina(url, goal)).strip()

        if not isinstance(url, list) or not all(isinstance(u, str) for u in url):
            return "[Visit] Invalid 'url': must be a string or a list of strings"

        sem = asyncio.Semaphore(self.max_concurrency)

        async def run_one(u: str) -> str:
            async with sem:
                try:
                    return await asyncio.wait_for(self.readpage_jina(u, goal), timeout=PER_URL_DEADLINE_SEC)
                except asyncio.TimeoutError:
                    return self._fallback(u, goal)
                except Exception as e:
                    return f"Error fetching {u}: {str(e)}"

        results = await asyncio.gather(*[run_one(u) for u in url], return_exceptions=False)
        response = "\n=======\n".join(results)

        return response.strip()

    def _fallback(self, url: str, goal: str) -> str:
        useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
        useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
        useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
        return useful_information

    async def jina_readpage(self, session: aiohttp.ClientSession, url: str) -> str:
        max_retries = 3
        timeout = aiohttp.ClientTimeout(total=50)

        for attempt in range(max_retries):
            key = _pick_jina_key()
            headers: Dict[str, str] = {}
            if key:
                headers["Authorization"] = f"Bearer {key}"

            try:
                async with session.get(f"https://r.jina.ai/{url}", headers=headers, timeout=timeout) as resp:
                    text = await resp.text()
                    if resp.status == 200 and text:
                        return text
                    await asyncio.sleep(0.5)
            except Exception:
                await asyncio.sleep(0.5)

        return "[visit] Failed to read page."

    async def html_readpage_jina(self, session: aiohttp.ClientSession, url: str) -> str:
        max_attempts = 8
        for _ in range(max_attempts):
            content = await self.jina_readpage(session, url)
            if (
                content
                and not content.startswith("[visit] Failed to read page.")
                and content != "[visit] Empty content."
                and not content.startswith("[document_parser]")
            ):
                return content
            
        return "[visit] Failed to read page."

    async def call_server(self, raw_response: str, goal: str, max_retries: int = 2) -> str:
        if not OPENROUTER_API_KEY:
            return ""
        
        client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(raw_response=raw_response, goal=goal)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=SUMMARY_MODEL_NAME,
                    messages=messages,
                    temperature=1.0,
                    top_p=0.95,
                    timeout=VISIT_SERVER_TIMEOUT,
                    extra_body={
                        "reasoning": {"enabled": True}
                    }
                )
                content = resp.choices[0].message.content or ""
                content = content.strip()
                if content:
                    return content
            except Exception:
                if attempt == max_retries - 1:
                    return ""
                await asyncio.sleep(0.3 * (2 ** attempt))

        return ""

    async def readpage_jina(self, url: str, goal: str) -> str:
        max_retries = int(os.getenv("VISIT_SERVER_MAX_RETRIES", 1))

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            content = await self.html_readpage_jina(session, url)

        if not content or content.startswith("[visit] Failed to read page.") or content == "[visit] Empty content." or content.startswith("[document_parser]"):
            return self._fallback(url, goal)

        content = truncate_to_tokens(content, max_tokens=WEBCONTENT_MAXLENGTH)

        raw = await self.call_server(raw_response=content, goal=goal, max_retries=max_retries)

        summary_retries = 3
        while len(raw) < 10 and summary_retries >= 0:
            truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
            if summary_retries > 0:
                print(
                    f"[visit] Summary url[{url}] attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, truncating to {truncate_length} chars"
                )
            else:
                print(
                    f"[visit] Summary url[{url}] failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )

            content = content[:truncate_length]
            raw = await self.call_server(raw_response=content, goal=goal, max_retries=max_retries)
            summary_retries -= 1

        parse_retry_times = 0
        parsed: Any = None
        while parse_retry_times < 3:
            try:
                parsed = _json_loads_loose(raw)
                break
            except Exception:
                raw = await self.call_server(raw_response=content, goal=goal, max_retries=max_retries)
                parse_retry_times += 1

        if parsed is None or not isinstance(parsed, dict):
            return self._fallback(url, goal)

        evidence = str(parsed.get("evidence", ""))
        summary = str(parsed.get("summary", ""))

        useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
        useful_information += "Evidence in page: \n" + evidence + "\n\n"
        useful_information += "Summary: \n" + summary + "\n\n"

        if len(useful_information) < 10:
            return "[visit] Failed to read page"

        return useful_information