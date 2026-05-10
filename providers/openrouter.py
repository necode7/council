"""
OpenRouter Provider — handles API calls to OpenRouter.
"""

import asyncio
import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 120.0


async def call_openrouter(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    label: str,
    max_retries: int = 2,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/llm-council",
        "X-Title": "LLM Council",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = f"[ERROR from {label} ({model}) — exhausted retries]"
    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=timeout
            )
            if resp.status_code != 200:
                last_error = f"[ERROR from {label} ({model}) — HTTP {resp.status_code}: {resp.text[:300]}]"
                if resp.status_code in (408, 425, 429, 500, 502, 503, 504) and attempt < max_retries:
                    await asyncio.sleep(1.5 ** attempt)
                    continue
                return last_error
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                last_error = f"[ERROR from {label} ({model}) — empty choices: {str(data)[:300]}]"
                if attempt < max_retries:
                    await asyncio.sleep(0.5)
                    continue
                return last_error
            content = choices[0].get("message", {}).get("content", "")
            if not content or not content.strip():
                last_error = f"[ERROR from {label} ({model}) — empty content]"
                if attempt < max_retries:
                    # Bump max_tokens on retry — empty content often means
                    # reasoning models burned all tokens internally.
                    payload["max_tokens"] = int(max_tokens * 1.5)
                    await asyncio.sleep(0.5)
                    continue
                return last_error
            return content.strip()
        except httpx.TimeoutException:
            last_error = f"[ERROR from {label} ({model}) — timed out after {timeout}s]"
            if attempt < max_retries:
                continue
            return last_error
        except Exception as e:
            last_error = f"[ERROR from {label} ({model}) — {type(e).__name__}: {e}]"
            return last_error
    return last_error
