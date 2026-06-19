import re

with open("pipeline/runner.py", "r", encoding="utf-8") as f:
    code = f.read()

new_functions = """
def _get_client():
    global _client
    if _client is not None:
        return _client
    
    provider = os.environ.get("VLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key: raise ValueError("GOOGLE_API_KEY not set")
        _client = genai.Client(api_key=key)
    elif provider == "azure":
        from openai import AzureOpenAI
        _client = AzureOpenAI(
            api_key=os.environ.get("AZURE_API_KEY"),
            azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
            api_version="2024-02-01"
        )
    else:
        from openai import OpenAI
        import os
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("QWEN_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1" if provider == "qwen" else os.environ.get("OPENAI_BASE_URL")
        _client = OpenAI(api_key=key, base_url=base_url)
    return _client

def _call_gemini(parts: list, max_tokens: int = 8192, retries: int = 5) -> str:
    import random
    import base64
    global _RETRY_COUNT

    provider = os.environ.get("VLM_PROVIDER", "gemini").lower()
    client = _get_client()

    for attempt in range(retries):
        try:
            if provider == "gemini":
                resp = client.models.generate_content(
                    model=MODEL,
                    contents=types.Content(role="user", parts=parts),
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=max_tokens,
                    ),
                )
                return resp.text or ""
            else:
                messages = [{"role": "user", "content": []}]
                for p in parts:
                    if isinstance(p, str):
                        messages[0]["content"].append({"type": "text", "text": p})
                    else:
                        # Assuming it's a types.Part with inline_data
                        b64 = base64.b64encode(p.inline_data.data).decode('utf-8')
                        mime = p.inline_data.mime_type or "image/png"
                        messages[0]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}
                        })
                
                resp = client.chat.completions.create(
                    model=os.environ.get("VLM_MODEL", "qwen-max"),
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens
                )
                return resp.choices[0].message.content or ""
        except Exception as exc:
            exc_str = str(exc).lower()
            is_rate_limit = "429" in exc_str or "quota" in exc_str or "rate" in exc_str
            if attempt >= retries - 1:
                raise
            _RETRY_COUNT += 1
            wait = min(60.0, 15 * (2 ** attempt)) + random.uniform(0, 5) if is_rate_limit else (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Gemini error (attempt {attempt+1}/{retries}): {exc} - retrying in {wait:.1f}s")
            time.sleep(wait)
    return ""
"""

pattern = re.compile(r"def _get_client\(\) -> genai\.Client:.*?return \"\"  # unreachable but satisfies type checker", re.DOTALL)

if not pattern.search(code):
    print("Pattern not found!")
else:
    new_code = pattern.sub(new_functions.strip(), code)
    with open("pipeline/runner.py", "w", encoding="utf-8") as f:
        f.write(new_code)
    print("Successfully refactored runner.py")
