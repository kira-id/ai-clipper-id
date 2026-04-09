"""
LLM backends: OpenRouter, Anthropic, OpenAI, Ollama.
"""

import json
import os
import re
import sys
import time
from typing import Any, Callable

from ..config import get_openrouter_settings
from ..utils import log, BOLD, RESET, DEFAULT_OPENROUTER_MODEL, DEFAULT_OPENROUTER_BASE


def _retry_on_rate_limit(
    api_call_fn: Callable,
    max_retries: int = 5,
    initial_wait: float = 2.0,
) -> str:
    """
    Retry API call on rate limits, transient provider errors, or empty responses.
    
    Args:
        api_call_fn: Callable that makes the API call (no args)
        max_retries: Maximum number of retries
        initial_wait: Initial wait time in seconds before first retry
    
    Returns:
        Response text from successful API call
        
    Notes:
        Non-retriable errors are raised immediately. Persistent retriable
        failures return an empty string so higher-level JSON handling can
        decide whether to retry with prompt constraints or continue.
    """
    for attempt in range(max_retries):
        try:
            response = api_call_fn()
            # Treat empty response as retriable error
            if response:
                return response
            
            # Empty response - retry unless this is the last attempt
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                log("WARN", f"Empty response. Retry {attempt + 1}/{max_retries} in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                log("ERROR", f"Empty response after {max_retries} retries")
                return ""
        except Exception as e:
            err_text = str(e)
            err_lower = err_text.lower()
            err_type = type(e).__name__

            # 1) Explicit rate-limit detection (429 / SDK RateLimitError).
            is_rate_limit = (
                "429" in err_text
                or "RateLimitError" in err_type
                or "rate limit" in err_lower
            )

            # 2) Transient upstream/transport parse failures.
            # OpenRouter (via OpenAI SDK) may occasionally return a non-JSON body
            # during provider hiccups, which surfaces as JSONDecodeError.
            is_transient = isinstance(e, json.JSONDecodeError) or any(
                marker in err_lower
                for marker in (
                    "expecting value",
                    "line 1 column 1",
                    "bad gateway",
                    "service unavailable",
                    "gateway timeout",
                    "502",
                    "503",
                    "504",
                    "api connection error",
                    "connection reset",
                    "timed out",
                )
            )

            if not (is_rate_limit or is_transient):
                # Non-retriable error; preserve existing behavior.
                raise

            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                reason = "Rate limited (429)" if is_rate_limit else "Transient API error"
                log("WARN", f"{reason}. Retry {attempt + 1}/{max_retries} in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                # Do not hard-crash the whole pipeline on persistent transient provider failures.
                # Returning empty string lets JSON retry logic log raw failure and continue.
                reason = "Rate limit" if is_rate_limit else "Transient API error"
                log("ERROR", f"{reason} persisted after {max_retries} retries")
                return ""


def _parse_llm_json(raw: str) -> tuple[list[dict[str, Any]], bool]:
    """
    Robustly extract a JSON array from LLM text.
    
    Returns:
        (parsed_data, success) — success=False if parsing failed
    """
    # Remove any thinking block that some reasoning models might output in content.
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    # Strip markdown code fences.
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip().rstrip("`").strip()

    def _extract_balanced_array(text: str) -> str | None:
        """
        Return first syntactically balanced JSON array substring if present.

        Handles nested braces/brackets and quoted strings so bracket matching
        does not break on user text like "]" inside JSON strings.
        """
        start = text.find("[")
        if start < 0:
            return None

        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _parse_partial_array(text: str) -> list[dict[str, Any]]:
        """
        Parse as many dict items as possible from an array-like payload.

        Useful when the model output is truncated near the end but still has
        valid leading items. Returns [] when no object can be decoded.
        """
        arr = text.strip()
        if not arr.startswith("["):
            return []

        # Work on the content after '['; allow missing closing ']'.
        body = arr[1:]
        decoder = json.JSONDecoder()
        idx = 0
        out: list[dict[str, Any]] = []

        while idx < len(body):
            # Skip delimiters and whitespace
            while idx < len(body) and body[idx] in " \t\r\n,":
                idx += 1
            if idx >= len(body) or body[idx] == "]":
                break

            try:
                item, next_idx = decoder.raw_decode(body, idx)
            except json.JSONDecodeError:
                # Cannot decode next element; keep already-decoded objects.
                break

            if isinstance(item, dict):
                out.append(item)
            idx = next_idx

        return out

    # Try direct parse first.
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data, True
        # Wrapped in an object — find the array
        for v in data.values():
            if isinstance(v, list):
                return v, True
        # Dict with no array value — parsing failed (triggers retry)
        return [], False
    except json.JSONDecodeError:
        pass
    # Last resort 1: find first balanced [ ... ] block and parse it.
    block = _extract_balanced_array(cleaned)
    if block:
        try:
            data = json.loads(block)
            if isinstance(data, list):
                return data, True
        except json.JSONDecodeError:
            pass

    # Last resort 2: salvage leading valid objects from a possibly truncated array.
    partial = _parse_partial_array(cleaned)
    if partial:
        log("WARN", f"Recovered {len(partial)} clip(s) from partial JSON array output")
        return partial, True

    return [], False


def _retry_on_json_failure(
    api_call_fn,
    system: str,
    user: str,
    max_attempts: int = 2,
) -> list[dict[str, Any]]:
    """
    Retry LLM call with modified prompts if JSON parsing fails.
    
    Args:
        api_call_fn: Callable that takes (system, user) and returns raw response text
        system: System prompt
        user: User prompt
        max_attempts: Total number of attempts (1 = single attempt, no retry)
    
    Returns:
        Parsed clips, or empty list if all attempts fail
    """
    retry_prompts = [
        "",  # First attempt: original prompt
        "\n\n[RETRY: Please respond with ONLY a valid JSON array, no other text. Start with [ and end with ].]",
        "\n\n[RETRY 2: Output ONLY valid JSON array. Example format: [{\"start\": 0, \"end\": 10, \"reason\": \"...\"}, ...].]",
    ]

    # Initialize raw so it is always bound even if the loop is skipped (max_attempts=0).
    raw = ""

    for attempt in range(min(max_attempts, len(retry_prompts))):
        modified_user = user + retry_prompts[attempt]
        raw = api_call_fn(system, modified_user)
        clips, success = _parse_llm_json(raw)
        
        if success:
            return clips
        
        if attempt < max_attempts - 1:
            log("WARN", f"JSON parse failed (attempt {attempt + 1}), retrying with modified prompt...")
            log("DEBUG", f"Raw output that failed to parse: {raw[:500]}...")
    
    log("ERROR", "Could not parse JSON from LLM response after retries")
    log("ERROR", f"Final raw output: {raw}")
    return []


def _is_reasoning_unsupported(error: Exception) -> bool:
    """
    Detect whether an API error indicates the model does not support reasoning.
    OpenRouter surfaces this as a 400 with a message about the parameter being
    invalid, unsupported, or not allowed for the chosen model.
    """
    msg = str(error).lower()
    markers = (
        "reasoning",
        "unsupported parameter",
        "invalid parameter",
        "not supported",
        "unknown field",
        "extra inputs are not permitted",
        "400",
    )
    return any(m in msg for m in markers)


def openrouter(
    system: str,
    user: str,
    api_key: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    base_url: str = DEFAULT_OPENROUTER_BASE,
    enable_reasoning: bool = False,
) -> list[dict[str, Any]]:
    """
    Call OpenRouter (OpenAI-compatible API).

    When *enable_reasoning* is True the call is made with
    ``reasoning={"effort": "high"}`` and temperature=1 (required by most
    reasoning-capable models).  If the model does not support reasoning the
    error is caught and the call is transparently retried without it at the
    normal temperature.

    The reasoning trace (``message.reasoning``) is logged at DEBUG level but
    is never mixed into the text that gets JSON-parsed, so downstream parsing
    is unaffected regardless of whether reasoning fired.
    """
    try:
        from openai import OpenAI
    except ImportError:
        log("ERROR", "openai SDK not installed.  pip install openai")
        sys.exit(1)

    log("LLM", f"OpenRouter → {BOLD}{model}{RESET}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    common_headers = {
        "HTTP-Referer": "https://github.com/ai-video-clipper",
        "X-Title": "AI Video Clipper",
    }

    def _call_once(sys_prompt: str, user_prompt: str, reasoning: bool) -> str:
        """
        Single attempt: call the API with or without reasoning and return the
        text content of the reply.  Rate-limit retries are handled inside here.
        """
        def api_call() -> str:
            kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=16000 if reasoning else 8192,
                # Reasoning models require temperature=1; use 0.3 otherwise.
                temperature=1 if reasoning else 0.3,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_headers=common_headers,
            )
            if reasoning:
                # OpenRouter forwards this to the underlying provider.
                # effort can be "low" | "medium" | "high".
                kwargs["extra_body"] = {"reasoning": {"effort": "high"}}

            resp = client.chat.completions.create(**kwargs)
            message = resp.choices[0].message

            reasoning_text: str = getattr(message, "reasoning", None) or ""
            content: str = message.content or ""

            if reasoning_text:
                preview = reasoning_text[:300].replace("\n", " ")
                log("DEBUG", f"Reasoning trace ({len(reasoning_text)} chars): {preview}…")

            # Some reasoning models place their final JSON
            # answer inside the reasoning field and leave content empty, or
            # vice-versa.  We pick whichever field contains parseable JSON;
            # if both are non-empty we prefer content (the intended answer).
            if content:
                _, content_ok = _parse_llm_json(content)
            else:
                content_ok = False

            if content_ok:
                return content

            # content missing or un-parseable — try reasoning field as fallback
            if reasoning_text:
                _, reasoning_ok = _parse_llm_json(reasoning_text)
                if reasoning_ok:
                    log("DEBUG", "JSON found in reasoning field; using that as answer.")
                    return reasoning_text

            # Never return non-JSON reasoning text as answer payload.
            # If content is empty and reasoning has no JSON, return empty so
            # retry logic can issue a fresh request.
            if content:
                return content
            return ""

        return _retry_on_rate_limit(api_call, max_retries=5, initial_wait=2.0)

    def _call_openrouter(sys_prompt: str, user_prompt: str) -> str:
        """
        Try with reasoning first; fall back to a plain call if the model
        signals it does not support the reasoning parameter.
        """
        if not enable_reasoning:
            return _call_once(sys_prompt, user_prompt, reasoning=False)

        try:
            result = _call_once(sys_prompt, user_prompt, reasoning=True)
            log("DEBUG", "Reasoning mode: ON")
            return result
        except Exception as e:
            if _is_reasoning_unsupported(e):
                log("WARN", f"Model does not support reasoning ({e}); retrying without.")
                return _call_once(sys_prompt, user_prompt, reasoning=False)
            raise  # unrelated error — propagate as usual

    def _call_openrouter_no_reasoning(sys_prompt: str, user_prompt: str) -> str:
        """Plain call with reasoning disabled — used as last-resort fallback."""
        return _call_once(sys_prompt, user_prompt, reasoning=False)

    # First attempt: with reasoning (if enabled).
    clips = _retry_on_json_failure(_call_openrouter, system, user, max_attempts=2)

    # Last-resort fallback: if reasoning produced nothing, try without it.
    if not clips and enable_reasoning:
        log("WARN", "Reasoning mode yielded no clips — retrying with reasoning OFF.")
        clips = _retry_on_json_failure(_call_openrouter_no_reasoning, system, user, max_attempts=2)
        if clips:
            log("OK", f"OpenRouter returned {len(clips)} clips (reasoning OFF fallback)")
        else:
            log("WARN", "OpenRouter returned 0 clips even without reasoning. Raw response logged above.")
        return clips

    if clips:
        log("OK", f"OpenRouter returned {len(clips)} clips")
    else:
        log("WARN", "OpenRouter returned 0 clips — model found nothing to clip in this chunk, or JSON parsing failed. Raw response logged above.")
    return clips


def anthropic(system: str, user: str, api_key: str) -> list[dict[str, Any]]:
    """Call Anthropic Claude. With retry on JSON parse failure and rate limits."""
    try:
        import anthropic
    except ImportError:
        log("ERROR", "anthropic SDK not installed.  pip install anthropic")
        sys.exit(1)

    # FIX: updated from stale snapshot ID "claude-sonnet-4-20250514" to the
    # current model string "claude-sonnet-4-6".
    model = "claude-sonnet-4-6"
    log("LLM", f"Anthropic → {BOLD}{model}{RESET}")
    
    def _call_anthropic(sys_prompt: str, user_prompt: str) -> str:
        """Internal function that makes the actual API call and returns raw response."""
        def api_call():
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=8192,
                system=sys_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text
        
        return _retry_on_rate_limit(api_call, max_retries=5, initial_wait=2.0)
    
    clips = _retry_on_json_failure(_call_anthropic, system, user, max_attempts=2)
    log("OK", f"Anthropic returned {len(clips)} clips")
    return clips


def openai(system: str, user: str, api_key: str) -> list[dict[str, Any]]:
    """Call OpenAI GPT-4o. With retry on JSON parse failure and rate limits."""
    try:
        from openai import OpenAI
    except ImportError:
        log("ERROR", "openai SDK not installed.  pip install openai")
        sys.exit(1)

    log("LLM", f"OpenAI → {BOLD}gpt-4o{RESET}")
    
    def _call_openai(sys_prompt: str, user_prompt: str) -> str:
        """Internal function that makes the actual API call and returns raw response."""
        def api_call():
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        
        return _retry_on_rate_limit(api_call, max_retries=5, initial_wait=2.0)
    
    clips = _retry_on_json_failure(_call_openai, system, user, max_attempts=2)
    log("OK", f"OpenAI returned {len(clips)} clips")
    return clips


def ollama(system: str, user: str) -> list[dict[str, Any]]:
    """Call local Ollama instance. With retry on JSON parse failure and rate limits."""
    try:
        import requests
    except ImportError:
        log("ERROR", "requests not installed.  pip install requests")
        sys.exit(1)

    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    log("LLM", f"Ollama (local) → {BOLD}{model}{RESET}")
    
    def _call_ollama(sys_prompt: str, user_prompt: str) -> str:
        """Internal function that makes the actual API call with rate limit retry."""
        def api_call():
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        
        return _retry_on_rate_limit(api_call, max_retries=5, initial_wait=2.0)
    
    clips = _retry_on_json_failure(_call_ollama, system, user, max_attempts=2)
    log("OK", f"Ollama returned {len(clips)} clips")
    return clips


def call_llm(
    system: str,
    user: str,
    api_key: str | None = None,
    llm_model: str | None = None,
    enable_reasoning: bool = False,
) -> list[dict[str, Any]]:
    """
    Call the best available LLM backend and return raw parsed clips.

    Backend priority:
      1. OpenRouter  (OPENROUTER_API_KEY) — default free model
      2. Anthropic   (ANTHROPIC_API_KEY)
      3. OpenAI      (OPENAI_API_KEY)
      4. Ollama      (local, no key needed)

    Args:
        enable_reasoning: When True, OpenRouter calls will request
            extended reasoning from the model.  If the model does not support
            it the call is transparently retried without reasoning.  Set to
            False (default) to skip reasoning (better consistency, lower cost/latency).
    """
    or_key = api_key or os.getenv("OPENROUTER_API_KEY")
    ant_key = os.getenv("ANTHROPIC_API_KEY")
    oai_key = os.getenv("OPENAI_API_KEY")

    if or_key:
        # Priority: explicit param → env var → config → default
        or_settings = get_openrouter_settings()
        model = llm_model or os.getenv("OPENROUTER_MODEL") or or_settings.get("model", DEFAULT_OPENROUTER_MODEL)
        base_url = or_settings.get("base_url", DEFAULT_OPENROUTER_BASE)
        return openrouter(system, user, or_key, model=model, base_url=base_url, enable_reasoning=enable_reasoning)
    elif ant_key:
        return anthropic(system, user, ant_key)
    elif oai_key:
        return openai(system, user, oai_key)
    else:
        log("WARN", "No API key found. Trying local Ollama …")
        return ollama(system, user)