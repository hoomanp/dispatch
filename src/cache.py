import hashlib
import json
from pathlib import Path
from typing import Callable, Any, Dict

# Resolve repo root relative to this file
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache" / "responses"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def cached_call(model: str, prompt: str, call_fn: Callable[[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """Caches LLM API responses to disk based on sha256(model + prompt)."""
    payload = f"{model}:{prompt}".encode("utf-8")
    cache_key = hashlib.sha256(payload).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["cached"] = True
            return data

    result = call_fn(model, prompt)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
