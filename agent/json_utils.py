"""Shared JSON extraction utility for agent nodes."""

import json
import re


def extract_json(raw: str) -> dict:
    """Extract the FIRST complete JSON object, then fall back to truncation repair."""
    clean = raw.strip()
    clean = re.sub(r"```[a-zA-Z]*\s*\n?", "", clean)
    clean = re.sub(r"\n?\s*```", "", clean)
    clean = clean.strip()

    start = clean.find('{')
    if start == -1:
        raise ValueError(f"No JSON object found: {clean[:120]}")

    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(clean[start:], start):
        if esc:
            esc = False
            continue
        if ch == '\\' and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(clean[start:i + 1])
                except json.JSONDecodeError:
                    break

    fragment = clean[start:]
    if fragment.count('"') % 2 == 1:
        fragment += '"'
    fragment = fragment.rstrip()
    if fragment.count('[') > fragment.count(']'):
        fragment = re.sub(r',\s*$', '', fragment) + ']'
    fragment = re.sub(r',?\s*"[^"]*"\s*:\s*$', '', fragment.rstrip())
    fragment = re.sub(r',\s*$', '', fragment.rstrip()) + '}'
    try:
        return json.loads(fragment)
    except json.JSONDecodeError as e:
        raise ValueError(f"No JSON object found: {clean[:120]}") from e


def extract_json_with_gemini_retry(raw: str, provider: str, gemini_call, prompt: str, system: str) -> dict:
    """extract_json(raw); on failure AND provider != gemini, retry once against
    Gemini via gemini_call(prompt, system) then extract_json that. For Gemini,
    behave exactly like extract_json (raise on failure)."""
    try:
        return extract_json(raw)
    except ValueError:
        if provider == "gemini":
            raise
        retry_raw = gemini_call(prompt, system)
        return extract_json(retry_raw)
