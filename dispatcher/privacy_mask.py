"""
DISPATCH Edge PII/PHI Redaction & Synthetic Token Masking (Phase 4).

Performs deterministic, high-speed on-premise token redaction before prompts
are dispatched to cloud models, and reverses the synthetic tokens on arrival
so the client receives seamless responses without exposing sensitive credentials
or personal identifiable information to third-party providers.

Supported Entity Recognizers:
  - API_KEY: OpenAI, Anthropic, AWS, GitHub tokens, Generic Bearer keys
  - EMAIL: Email addresses
  - PHONE: US & International phone numbers
  - SSN: Social Security Numbers
  - CREDIT_CARD: Visa, Mastercard, Amex, Discover
  - IP_ADDRESS: IPv4 network addresses
  - JWT: JSON Web Tokens
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dispatch.privacy_mask")

# Compiled High-Precision Entity Pattern Recognizers (ordered from most specific to general)
ENTITY_PATTERNS = {
    "JWT": re.compile(r"\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b"),
    "API_KEY": re.compile(
        r"\b(?:sk-(?:ant-|proj-|live-|test-|[a-zA-Z0-9]{20,})[a-zA-Z0-9_-]*|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36}|"
        r"AIza[0-9A-Za-z-_]{35}|bearer\s+[a-zA-Z0-9_\-\.]{24,})\b", re.I),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(
        r"\b(?:4[0-9]{3}(?:[- ]?[0-9]{4}){3}|5[1-5][0-9]{2}(?:[- ]?[0-9]{4}){3}|"
        r"3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
    ),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"),
    "IP_ADDRESS": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ),
}

_MASK_TOKEN_RE = re.compile(r"^<MASK:([A-Z_]+)_(\d+)>$")


class PrivacyMask:
    """Zero-dependency edge privacy masking and unmasking engine."""

    def __init__(self, patterns: Optional[Dict[str, re.Pattern]] = None):
        self.patterns = patterns or ENTITY_PATTERNS

    @staticmethod
    def _init_counters_from_map(token_map: Dict[str, str]) -> Dict[str, int]:
        """Derive current max index per entity_type from existing token_map
        so multi-turn message masking never overwrites earlier synthetic tokens."""
        counters: Dict[str, int] = {}
        for synthetic_token in token_map.keys():
            m = _MASK_TOKEN_RE.match(synthetic_token)
            if m:
                etype, idx_str = m.group(1), m.group(2)
                counters[etype] = max(counters.get(etype, 0), int(idx_str))
        return counters

    def mask_text(self, text: str, mapping: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, str], List[str]]:
        """Scan and mask sensitive tokens with synthetic placeholders using single-pass
        regex span replacement (immune to prefix/substring clobbering and multi-turn collisions).

        Returns:
            Tuple of (masked_text, token_mapping, list_of_detected_entity_types)
        """
        if not text:
            return text, mapping if mapping is not None else {}, []

        token_map = mapping if mapping is not None else {}
        rev_map = {v: k for k, v in token_map.items()}
        counters = self._init_counters_from_map(token_map)
        detected_types: List[str] = []

        masked = text
        for entity_type, pattern in self.patterns.items():
            found_for_type = False

            def _replacer(match_obj: re.Match) -> str:
                nonlocal found_for_type
                raw_val = match_obj.group(0)
                found_for_type = True
                if raw_val in rev_map:
                    return rev_map[raw_val]
                counters[entity_type] = counters.get(entity_type, 0) + 1
                idx = counters[entity_type]
                synthetic_token = f"<MASK:{entity_type}_{idx}>"
                token_map[synthetic_token] = raw_val
                rev_map[raw_val] = synthetic_token
                return synthetic_token

            masked = pattern.sub(_replacer, masked)
            if found_for_type:
                detected_types.append(entity_type)

        return masked, token_map, detected_types

    def mask_messages(self, messages: List[Dict[str, Any]],
                      mapping: Optional[Dict[str, str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
        """Mask all message contents in an OpenAI-format messages list while preserving
        a consistent bijection across all turns."""
        token_map = mapping if mapping is not None else {}
        all_detected: List[str] = []
        masked_messages: List[Dict[str, Any]] = []

        for msg in messages:
            new_msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, str):
                masked_content, token_map, detected = self.mask_text(content, token_map)
                new_msg["content"] = masked_content
                for d in detected:
                    if d not in all_detected:
                        all_detected.append(d)
            elif isinstance(content, list):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                        b_text, token_map, detected = self.mask_text(block["text"], token_map)
                        for d in detected:
                            if d not in all_detected:
                                all_detected.append(d)
                        new_blocks.append({**block, "text": b_text})
                    else:
                        new_blocks.append(block)
                new_msg["content"] = new_blocks

            masked_messages.append(new_msg)

        return masked_messages, token_map, all_detected

    def unmask_text(self, text: str, mapping: Dict[str, str]) -> str:
        """Substitute synthetic tokens in generated output back with original values."""
        if not text or not mapping:
            return text
        unmasked = text
        # Sort keys by length descending for deterministic replacement
        for synthetic_token in sorted(mapping.keys(), key=len, reverse=True):
            unmasked = unmasked.replace(synthetic_token, mapping[synthetic_token])
        return unmasked


privacy_mask = PrivacyMask()
