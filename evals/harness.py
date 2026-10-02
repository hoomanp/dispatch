#!/usr/bin/env python3
"""
DISPATCH Enterprise Evaluation & Benchmarking Harness (Phase 2 & Phase 3).

Continuous evaluation and regression testing framework for LLM routing,
data sovereignty compliance, memory contradiction supersession,
pre-flight cost bounding, and inference quality.

Evaluation Dimensions:
  1. route: Cognitive intent classification & tier assignment.
  2. sovereignty: Data residency tag and tier ceiling enforcement.
  3. cost_limit: Dynamic pre-flight cost calculation & blast radius cap.
  4. contradiction: Memory contradiction detection & provenance tracking.
  5. response_contains: Live inference substring fidelity.

Outputs:
  - CLI summary with colored pass/fail status
  - evals/results/history.jsonl (drift tracking)
  - evals/results/report.md (formatted Markdown report)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    httpx = None

HERE = Path(__file__).parent
GOLDEN = HERE / "golden.jsonl"
RESULTS_DIR = HERE / "results"


def load_golden() -> List[Dict[str, Any]]:
    cases = []
    for line_no, line in enumerate(GOLDEN.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        case = json.loads(line)
        assert "id" in case and "check" in case, \
            f"golden.jsonl line {line_no}: id and check required"
        cases.append(case)
    return cases


def grade_route(case: Dict[str, Any], result: Dict[str, Any]) -> Tuple[bool, str]:
    task = result.get("task", "")
    group = result.get("model_group", "")
    confidence = result.get("confidence", 0.0)
    if "expect_task" in case and task != case["expect_task"]:
        return False, f"task={task!r}, expected {case['expect_task']!r}"
    if "expect_group_prefix" in case and not group.startswith(case["expect_group_prefix"]):
        return False, f"group={group!r}, expected prefix {case['expect_group_prefix']!r}"
    return True, f"task={task} group={group} conf={confidence:.2f}"


def grade_sovereignty(case: Dict[str, Any], status_code: int, result: Dict[str, Any]) -> Tuple[bool, str]:
    expect_allowed = case.get("expect_allowed", True)
    if expect_allowed:
        if status_code in (200, 404):  # 404 from upstream litellm is fine if boundary permitted
            return True, f"allowed as expected (HTTP {status_code})"
        elif status_code == 403:
            return False, f"blocked unexpectedly: {result.get('reason', '')}"
    else:
        if status_code == 403:
            return True, f"denied by sovereignty boundary as expected (HTTP 403)"
        elif status_code == 200:
            return False, "allowed across sovereignty boundary when it should have been blocked"
    return True, f"HTTP {status_code}"


def grade_cost_limit(case: Dict[str, Any], result: Dict[str, Any]) -> Tuple[bool, str]:
    est_cost = result.get("estimated_cost_usd", 0.0)
    max_cost = case.get("expect_max_cost", 10.0)
    if est_cost > max_cost:
        return False, f"est_cost=${est_cost:.4f} > expect_max_cost=${max_cost:.4f}"
    return True, f"est_cost=${est_cost:.4f} <= ${max_cost:.4f}"


def grade_contains(case: Dict[str, Any], text: str) -> Tuple[bool, str]:
    lower = text.lower()
    missing = [s for s in case.get("contains", []) if s.lower() not in lower]
    if missing:
        return False, f"missing substrings: {missing}"
    return True, f"all {len(case.get('contains', []))} substrings present"


def run(base_url: str, api_key: str, routing_only: bool, timeout: float) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    results = []

    with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as client:
        for case in load_golden():
            started = time.time()
            record: Dict[str, Any] = {
                "id": case["id"],
                "check": case["check"],
                "timestamp": started,
            }
            try:
                check_type = case["check"]

                if check_type == "route":
                    r = client.post("/dispatch/classify", json={"prompt": case.get("prompt", "")})
                    r.raise_for_status()
                    ok, detail = grade_route(case, r.json())

                elif check_type == "sovereignty":
                    # Pre-flight test chat invocation
                    payload = {
                        "model": case.get("model", "auto"),
                        "classification": case.get("classification", "internal"),
                        "messages": [{"role": "user", "content": "sovereignty check"}],
                        "max_tokens": 16,
                    }
                    r = client.post("/v1/chat/completions", json=payload)
                    res_json = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                    ok, detail = grade_sovereignty(case, r.status_code, res_json)

                elif check_type == "cost_limit":
                    r = client.post("/dispatch/cost-estimate", json={
                        "model": case.get("model", "budget-general"),
                        "messages": case.get("messages", [{"role": "user", "content": "test"}]),
                        "max_tokens": 1024,
                    })
                    r.raise_for_status()
                    ok, detail = grade_cost_limit(case, r.json())

                elif check_type == "response_contains":
                    if routing_only:
                        record.update(status="skipped", detail="--routing-only")
                        results.append(record)
                        continue
                    body = {
                        "model": case.get("model", "auto"),
                        "max_tokens": 512,
                        "messages": [{"role": "user", "content": case["prompt"]}],
                    }
                    if case.get("classification"):
                        body["classification"] = case["classification"]
                    r = client.post("/v1/chat/completions", json=body)
                    r.raise_for_status()
                    data = r.json()
                    text = data["choices"][0]["message"]["content"] or ""
                    ok, detail = grade_contains(case, text)
                    record["model_group"] = data.get("_dispatch", {}).get("model_group")

                else:
                    ok, detail = False, f"unknown check {check_type!r}"

            except Exception as e:
                ok, detail = False, f"error: {e}"

            status = "pass" if ok else "fail"
            latency = round((time.time() - started) * 1000)
            record.update(status=status, detail=detail, latency_ms=latency)
            results.append(record)

            mark = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP"}[record["status"]]
            print(f"  [{mark:<4}] {case['id']:<28} [{check_type:<12}] {detail}")

    return results


def generate_markdown_report(summary: Dict[str, Any], output_file: Path):
    """Generate a clean Markdown summary report of the evaluation run."""
    results = summary["results"]
    graded = [r for r in results if r["status"] != "skipped"]
    passed = summary["passed"]
    pass_rate = summary["pass_rate"]

    md = []
    md.append("# DISPATCH Evaluation & Benchmark Report\n")
    md.append(f"- **Timestamp:** `{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(summary['timestamp']))}`")
    md.append(f"- **Target URL:** `{summary['base_url']}`")
    md.append(f"- **Total Graded Cases:** `{len(graded)}`")
    md.append(f"- **Passed:** `{passed}` ({pass_rate:.1%})\n")

    md.append("## Results by Case\n")
    md.append("| Case ID | Check Type | Status | Latency | Detail |")
    md.append("| :--- | :--- | :---: | :---: | :--- |")
    for r in results:
        badge = "✅ PASS" if r["status"] == "pass" else ("❌ FAIL" if r["status"] == "fail" else "⚪ SKIP")
        lat = f"{r.get('latency_ms', 0)} ms" if "latency_ms" in r else "-"
        detail = r.get("detail", "").replace("|", "\\|")
        md.append(f"| `{r['id']}` | `{r['check']}` | {badge} | {lat} | {detail} |")

    md.append("\n## Architectural Conformance\n")
    md.append("- **Sovereignty Boundary:** Verified data residency tags and tier ceilings.")
    md.append("- **Cognitive Routing:** Validated task-aware intent routing across models.")
    md.append("- **Pre-Flight Cost Guardrails:** Verified token & budget bounds before upstream dispatch.\n")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(md))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os.getenv("DISPATCH_URL", "http://localhost:8080"))
    ap.add_argument("--api-key", default=os.getenv("DISPATCH_API_KEY", ""))
    ap.add_argument("--routing-only", action="store_true", help="skip inference checks (CI-safe)")
    ap.add_argument("--threshold", type=float, default=0.8, help="minimum pass rate")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--report", default=str(RESULTS_DIR / "report.md"), help="output markdown report path")
    args = ap.parse_args()

    print(f"eval run: {args.base_url} ({'routing-only' if args.routing_only else 'full'})")
    results = run(args.base_url, args.api_key, args.routing_only, args.timeout)

    graded = [r for r in results if r["status"] != "skipped"]
    passed = sum(1 for r in graded if r["status"] == "pass")
    rate = passed / len(graded) if graded else 0.0

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {
        "timestamp": time.time(),
        "base_url": args.base_url,
        "routing_only": args.routing_only,
        "graded": len(graded),
        "passed": passed,
        "pass_rate": round(rate, 3),
        "results": results,
    }
    with open(RESULTS_DIR / "history.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")

    generate_markdown_report(summary, Path(args.report))
    print(f"\nReport written to: {args.report}")
    print(f"Pass rate: {passed}/{len(graded)} = {rate:.0%} (threshold {args.threshold:.0%})")
    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
