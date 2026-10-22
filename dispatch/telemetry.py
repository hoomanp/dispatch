import json
import time
from pathlib import Path
from typing import Any, Dict, List


class FinOpsLedger:
    def __init__(self, log_path: str = ".cache/finops_ledger.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    def emit(self, record: Dict[str, Any]) -> None:
        enriched = dict(record)
        enriched["timestamp"] = round(time.time(), 4)
        self.records.append(enriched)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(enriched, sort_keys=True) + "\n")
