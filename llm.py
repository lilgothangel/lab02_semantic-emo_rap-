import json
import time
from typing import Type, TypeVar

from pydantic import BaseModel

import config

T = TypeVar("T", bound=BaseModel)
_last_call = 0.0
_gemini_client = None


def _throttle():
    global _last_call
    wait = config.LLM_MIN_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        if not config.GEMINI_API_KEY:
            raise RuntimeError("Немає GEMINI_API_KEY у .env (ключ: aistudio.google.com)")
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def _retry(fn, attempts=8):
    for i in range(attempts):
        try:
            _throttle()
            return fn()
        except Exception as e:  # 429/503 - чекаємо
            msg = str(e)
            if i < attempts - 1 and any(c in msg for c in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                delay = 20 * (i + 1)
                print(f"  [llm] {msg[:80]}... повтор через {delay} с")
                time.sleep(delay)
                continue
            raise


def _gen_config(**kw):
    from google.genai import types
    t = config.GEMINI_THINKING.strip()
    if t:
        kw["thinking_config"] = (types.ThinkingConfig(thinking_budget=int(t)) if t.isdigit()
                                 else types.ThinkingConfig(thinking_level=t))
    return types.GenerateContentConfig(**kw)


def _call_gemini(contents, **cfg):
    try:
        return _gemini().models.generate_content(model=config.GEMINI_MODEL, contents=contents,
                                                 config=_gen_config(**cfg))
    except Exception as e:
        # модель не підтримує такий thinking_config - пробуємо без нього
        if config.GEMINI_THINKING and "think" in str(e).lower():
            print(f"  [llm] thinking '{config.GEMINI_THINKING}' не підтримується, вимикаю")
            config.GEMINI_THINKING = ""
            return _gemini().models.generate_content(model=config.GEMINI_MODEL, contents=contents,
                                                     config=_gen_config(**cfg))
        raise


def generate_structured(prompt: str, schema: Type[T]) -> T:
    def call():
        r = _call_gemini(prompt, response_mime_type="application/json",
                         response_schema=schema, temperature=0.0)
        if r.parsed is not None:
            return r.parsed
        return schema.model_validate_json(r.text)
    return _retry(call)


def generate_text(prompt: str, temperature: float = 0.2) -> str:
    return _retry(lambda: _call_gemini(prompt, temperature=temperature).text.strip())


def embed(texts: list[str], is_query: bool = False) -> list[list[float]]:
    from google.genai import types
    out = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]

        def call():
            return _gemini().models.embed_content(
                model=config.GEMINI_EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
                    output_dimensionality=config.EMBED_DIM,
                ),
            )
        out.extend(e.values for e in _retry(call).embeddings)
    return out


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
