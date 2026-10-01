import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "dispatcher"))
from adapters import (anthropic_to_internal, internal_to_anthropic,
                      gemini_to_internal, internal_to_gemini,
                      map_model_to_group)

OPENAI_RESP = {"choices": [{"message": {"content": "hello"},
                            "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 7,
                         "total_tokens": 12}}


def test_anthropic_request_translation():
    r = anthropic_to_internal({
        "model": "claude-sonnet-4-6", "max_tokens": 99,
        "system": "sys", "messages": [{"role": "user", "content": "hi"}]})
    assert r["model"] == "premium-balanced"
    assert r["messages"][0] == {"role": "system", "content": "sys"}
    assert r["max_tokens"] == 99 and r["stream"] is False

def test_anthropic_block_content():
    r = anthropic_to_internal({"model": "x", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "a"},
                                     {"type": "text", "text": "b"}]}]})
    assert r["messages"][0]["content"] == "a\nb"

def test_anthropic_response_translation():
    out = internal_to_anthropic(OPENAI_RESP, "claude-sonnet-4-6")
    assert out["type"] == "message"
    assert out["content"][0]["text"] == "hello"
    assert out["usage"] == {"input_tokens": 5, "output_tokens": 7}

def test_gemini_request_translation():
    r = gemini_to_internal({
        "systemInstruction": {"parts": [{"text": "sys"}]},
        "contents": [{"role": "user", "parts": [{"text": "q"}]},
                     {"role": "model", "parts": [{"text": "a"}]}],
        "generationConfig": {"maxOutputTokens": 33}}, "gemini-2.5-pro")
    assert r["model"] == "premium-gemini"
    assert r["messages"][2]["role"] == "assistant"
    assert r["max_tokens"] == 33

def test_gemini_response_translation():
    out = internal_to_gemini(OPENAI_RESP)
    assert out["candidates"][0]["content"]["parts"][0]["text"] == "hello"
    assert out["usageMetadata"]["totalTokenCount"] == 12

def test_model_mapping():
    assert map_model_to_group("claude-opus-4-6") == "premium-best"
    assert map_model_to_group("claude-haiku-4-5") == "budget-claude"
    assert map_model_to_group("gemini-2.0-flash") == "budget-fast"
    assert map_model_to_group("gpt-4o") == "premium-openai"
    assert map_model_to_group("local-fast") == "local-fast"
    assert map_model_to_group("mystery-9000") == "auto"
