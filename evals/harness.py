#!/usr/bin/env python3
"""
DISPATCH eval harness — observability stack #3 (continuous evaluation).

Runs a golden set (evals/golden.jsonl) against a live DISPATCH instance
and reports a pass rate. Exit code 1 when below --threshold, so a cron
job or CI step becomes a regression gate for model swaps, config edits,
and provider changes.

Two check types:
  route             hits /dispatch/classify only — validates task
                    classification + tier selection. Needs NO providers
                    online, so it is safe in CI.
  response_contains full inference via /v1/chat/completions — response
                    must contain all listed substrings (case-insensitive).
                    Needs a live stack; skipped under --routing-only.

Usage:
  python evals/harness.py                          # full run, localhost
  python evals/harness.py --routing-only           # CI-safe subset
  python evals/harness.py --base-url http://mini:8080 --threshold 0.9

Golden case schema (one JSON object per line):
  {"id": "...", "check": "route", "prompt": "...",
   "expect_task": "coding", "expect_group_prefix": "local"}
  {"id": "...", "check": "response_contains", "prompt": "...",
   "contains": ["def ", "return"], "classification": "internal"}

Results append to evals/results/history.jsonl for drift tracking.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
GOLDEN = HERE / "golden.jsonl"
RESULTS_DIR = HERE / "results"


def load_golden() -> list[dict]:
    cases = []
    for line_no, line in enumerate(GOLDEN.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        case = json.loads(line)
        assert "id" in case and "check" in case and "prompt" in case, \
            f"golden.jsonl line {line_no}: id/check/prompt required"
        cases.append(case)
    return cases


def grade_route(case: dict, result: dict) -> tuple[bool, str]:
    """Grade a /dispatch/classify result against expectations."""
    task = result.get("task", "")
    group = result.get("model_group", "")
    if "expect_task" in case and task != case["expect_task"]:
        return False, f"task={task!r}, expected {case['expect_task']!r}"
    if "expect_group_prefix" in case and not group.startswith(
            case["expect_group_prefix"]):
        return False, (f"group={group!r}, expected prefix "
                       f"{case['expect_group_prefix']!r}")
    return True, f"task={task} group={group}"


def grade_contains(case: dict, text: str) -> tuple[bool, str]:
    """Grade an inference response for required substrings."""
    lower = text.lower()
    missing = [s for s in case.get("contains", [])
               if s.lower() not in lower]
    if missing:
        return False, f"missing substrings: {missing}"
    return True, f"all {len(case.get('contains', []))} substrings present"


def run(base_url: str, api_key: str, routing_only: bool,
        timeout: float) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    results = []
    with httpx.Client(base_url=base_url, headers=headers,
                      timeout=timeout) as client:
        for case in load_golden():
            started = time.time()
            record = {"id": case["id"], "check": case["check"],
                      "timestamp": started}
            try:
                if case["check"] == "route":
                    r = client.post("/dispatch/classify",
                                    json={"prompt": case["prompt"]})
                    r.raise_for_status()
                    ok, detail = grade_route(case, r.json())
                elif case["check"] == "response_contains":
                    if routing_only:
                        record.update(status="skipped",
                                      detail="--routing-only")
                        results.append(record)
                        continue
                    body = {"model": "auto", "max_tokens": 512,
                            "messages": [{"role": "user",
                                          "content": case["prompt"]}]}
                    if case.get("classification"):
                        body["classification"] = case["classification"]
                    r = client.post("/v1/chat/completions", json=body)
                    r.raise_for_status()
                    data = r.json()
                    text = data["choices"][0]["message"]["content"] or ""
                    ok, detail = grade_contains(case, text)
                    record["model_group"] = data.get(
                        "_dispatch", {}).get("model_group")
                else:
                    ok, detail = False, f"unknown check {case['check']!r}"
            except Exception as e:
                ok, detail = False, f"error: {e}"
            record.update(status="pass" if ok else "fail", detail=detail,
                          latency_ms=round((time.time() - started) * 1000))
            results.append(record)
            mark = {"pass": "PASS", "fail": "FAIL",
                    "skipped": "SKIP"}[record["status"]]
            print(f"  [{mark}] {case['id']:<28} {detail}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url",
                    default=os.getenv("DISPATCH_URL",
                                      "http://localhost:8080"))
    ap.add_argument("--api-key", default=os.getenv("DISPATCH_API_KEY", ""))
    ap.add_argument("--routing-only", action="store_true",
                    help="skip inference checks (CI-safe)")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="minimum pass rate (graded cases only)")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    print(f"eval run: {args.base_url} "
          f"({'routing-only' if args.routing_only else 'full'})")
    results = run(args.base_url, args.api_key, args.routing_only,
                  args.timeout)

    graded = [r for r in results if r["status"] != "skipped"]
    passed = sum(1 for r in graded if r["status"] == "pass")
    rate = passed / len(graded) if graded else 0.0

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {"timestamp": time.time(), "base_url": args.base_url,
               "routing_only": args.routing_only, "graded": len(graded),
               "passed": passed, "pass_rate": round(rate, 3),
               "results": results}
    with open(RESULTS_DIR / "history.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"\npass rate: {passed}/{len(graded)} = {rate:.0%} "
          f"(threshold {args.threshold:.0%})")
    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
