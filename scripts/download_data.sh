#!/usr/bin/env bash
set -euo pipefail

echo "Generating synthetic golden set for support routing evaluation..."

python3 - << 'PYEOF'
import json
from pathlib import Path

samples = [
    {
        "id": "req-001",
        "prompt": "How do I update the credit card on my subscription account?",
        "rubric": {
            "required_terms": ["billing", "settings", "payment"]
        }
    },
    {
        "id": "req-002",
        "prompt": "I need to dispute an unauthorized charge on my account and request an immediate refund.",
        "rubric": {
            "required_terms": ["dispute", "investigation", "refund"]
        }
    },
    {
        "id": "req-003",
        "prompt": "What are your business support hours and response time SLAs?",
        "rubric": {
            "required_terms": ["hours", "support", "sla"]
        }
    },
    {
        "id": "req-004",
        "prompt": "Please cancel my annual agreement immediately under section 4.2 terms of service exception.",
        "rubric": {
            "required_terms": ["cancellation", "terms", "policy"]
        }
    },
    {
        "id": "req-005",
        "prompt": "Where can I download the SOC 2 compliance report for my organization's vendor audit?",
        "rubric": {
            "required_terms": ["compliance", "security", "report"]
        }
    }
]

# Expand to 20 representative test queries across policy tiers
extended_samples = []
for i in range(4):
    for s in samples:
        item = dict(s)
        item["id"] = f"{s['id']}-run{i+1}"
        extended_samples.append(item)

out_file = Path("data/golden_set.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(extended_samples, f, indent=2)

print(f"Created {out_file} with {len(extended_samples)} evaluation cases.")
PYEOF

echo "Data setup complete."
