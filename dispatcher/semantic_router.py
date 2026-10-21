"""
DISPATCH Cognitive & Semantic Task Router (Phase 3).

Provides zero-shot semantic intent classification and task routing.
Combines TF-IDF weighted semantic exemplar similarity with regex feature priors
to reliably route prompts to optimal model tiers (local, free, budget, premium)
with explainable confidence scores.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dispatch.semantic_router")

# Canonical task routes mapped to LiteLLM model groups
TASK_ROUTES: Dict[str, List[str]] = {
    "coding":       ["local-fast", "free-fast", "budget-general", "premium-balanced"],
    "reasoning":    ["local-fast", "budget-reasoning", "free-fast", "premium-balanced"],
    "fast_chat":    ["local-always-on", "local-mobile", "free-fast", "budget-fast"],
    "long_context": ["free-longcontext", "budget-fast", "budget-multilingual", "premium-gemini"],
    "multilingual": ["budget-multilingual", "free-longcontext", "budget-fast", "premium-gemini"],
    "math":         ["budget-reasoning", "free-fast", "premium-balanced", "premium-openai"],
    "quality":      ["premium-balanced", "premium-openai", "premium-gemini", "premium-best"],
    "governance":   ["local-always-on", "budget-general", "premium-balanced"],
    "retrieval":    ["local-embed", "local-fast", "budget-fast"],
    "general":      ["local-always-on", "free-general", "free-fast", "budget-general"],
}

# Regex feature patterns for rapid deterministic prior calculation
PATTERNS = {
    "coding": re.compile(
        r"\b(code|function|class|def |implement|refactor|debug|bug|script|"
        r"python|typescript|javascript|rust|golang|sql|api|endpoint|"
        r"algorithm|regex|dockerfile|git|json|yaml|html|css|react|vue)\b", re.I),
    "reasoning": re.compile(
        r"\b(analyze|reason|explain why|compare|evaluate|pros.?cons|"
        r"trade.?off|architecture|design|strategy|step.?by.?step|decision|root cause)\b", re.I),
    "math": re.compile(
        r"\b(calculate|compute|solve|equation|derivative|integral|proof|"
        r"theorem|probability|statistics|matrix|eigenvalue|bayes|calculus)\b", re.I),
    "long_context": re.compile(
        r"\b(entire|whole|full|complete|all of|summarize this|"
        r"analyze this|the following file|in the document above|attached)\b", re.I),
    "multilingual": re.compile(
        r"\b(translate|in (spanish|french|german|chinese|japanese|arabic|"
        r"persian|korean|portuguese|italian|russian|hindi)|translation)\b", re.I),
    "quality": re.compile(
        r"\b(best possible|highest quality|most accurate|critical|"
        r"production.?ready|thorough|rigorous|publication)\b", re.I),
    "governance": re.compile(
        r"\b(compliance|policy|gdpr|hipaa|sovereignty|audit|residency|"
        r"pii|redact|clearance|classified|confidential)\b", re.I),
}

# Exemplar corpus for cognitive term & semantic matching
TASK_EXEMPLARS: Dict[str, List[str]] = {
    "coding": [
        "write a python function to parse json and validate schemas",
        "debug this null pointer exception in typescript backend",
        "create a dockerfile and docker-compose service configuration",
        "implement a binary search tree in rust with memory safety",
        "refactor this sql query for improved postgres index performance",
    ],
    "reasoning": [
        "analyze the architectural trade-offs between monolithic and microservice designs",
        "compare postgresql and clickhouse for real-time telemetry analytics",
        "what are the pros and cons of event sourcing versus snapshot state?",
        "explain why this distributed consensus protocol failed during network partition",
        "evaluate the security and latency implications of edge token validation",
    ],
    "math": [
        "calculate the bayesian posterior probability given prior distribution",
        "solve this system of differential equations step by step",
        "compute the eigenvalues and eigenvectors of this covariance matrix",
        "prove by induction that the sum of first n odd numbers is n squared",
        "derive the gradient of cross-entropy loss with softmax activation",
    ],
    "long_context": [
        "read through this 50 page specification and summarize key requirements",
        "extract all action items and deadlines from the complete meeting transcript",
        "analyze the full codebase repository architecture from the attached files",
        "find all mentions of security vulnerabilities across all attached audit logs",
    ],
    "multilingual": [
        "translate this contract from English to French preserving legal terminology",
        "translate user support inquiries into Spanish and German accurately",
        "convert this documentation into Japanese and Korean with natural phrasing",
        "summarize this Arabic article in English with key takeaways",
    ],
    "quality": [
        "generate the highest quality production ready security review for executive board",
        "perform a rigorous peer review for publication in top tier journal",
        "provide the most accurate and critical analysis with zero tolerance for hallucinations",
    ],
    "governance": [
        "verify data residency and sovereignty requirements under GDPR guidelines",
        "audit this payload for PII, HIPAA compliance, and sensitive credentials",
        "check if this confidential dataset is permitted to cross regional boundary",
    ],
    "fast_chat": [
        "hello there, how are you today?",
        "thanks for the help!",
        "what is the time?",
        "yes, proceed please",
    ],
}

_TECH_SHORT_TOKENS = {
    "go", "ai", "ui", "db", "os", "ip", "ci", "cd", "ml", "js", "ts", "py", "v1", "o1", "o3", "pi", "vm", "s3"
}

_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "any", "can", "had",
    "her", "was", "one", "our", "out", "has", "him", "his", "new", "now",
    "old", "see", "two", "way", "who", "did", "its", "let", "put", "say",
    "she", "too", "use", "this", "that", "with", "from", "have",
}


def extract_content_text(content: Any) -> str:
    """Safely extract text string from string or multimodal list content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content or "")


def _tokenize(text: str) -> List[str]:
    """Tokenize lowercase alphanumeric terms, retaining high-signal 2-char technical tokens."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return [
        w for w in clean.split()
        if (len(w) >= 3 or w in _TECH_SHORT_TOKENS) and w not in _STOPWORDS
    ]


def _build_term_vector(tokens: List[str], idf: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Build L2-normalized TF-IDF vector."""
    if not tokens:
        return {}
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    weighted: Dict[str, float] = {}
    for t, c in counts.items():
        w = (1.0 + math.log(c)) * (idf.get(t, 1.5) if idf else 1.0)
        weighted[t] = w
    norm = math.sqrt(sum(w * w for w in weighted.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in weighted.items()}


def _cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Compute cosine similarity between two sparse term vectors."""
    if not v1 or not v2:
        return 0.0
    common = set(v1.keys()) & set(v2.keys())
    return sum(v1[k] * v2[k] for k in common)


@dataclass
class SemanticRouteDecision:
    task: str
    confidence: float
    model_group: str
    route_candidates: List[str]
    justification: str
    features: List[str] = field(default_factory=list)


class SemanticRouter:
    """Cognitive intent classifier and task router."""

    def __init__(self):
        # Compute IDF weights across exemplar corpus
        doc_freq: Dict[str, int] = {}
        total_docs = 0
        tokenized_corpus: Dict[str, List[List[str]]] = {}
        for task, exemplars in TASK_EXEMPLARS.items():
            tokenized_corpus[task] = []
            for ex in exemplars:
                total_docs += 1
                toks = _tokenize(ex)
                tokenized_corpus[task].append(toks)
                for unique_tok in set(toks):
                    doc_freq[unique_tok] = doc_freq.get(unique_tok, 0) + 1

        self._idf: Dict[str, float] = {
            tok: math.log((1.0 + total_docs) / (1.0 + df)) + 1.0
            for tok, df in doc_freq.items()
        }

        self._exemplar_vectors: Dict[str, List[Dict[str, float]]] = {}
        for task, tok_lists in tokenized_corpus.items():
            self._exemplar_vectors[task] = [
                _build_term_vector(toks, self._idf) for toks in tok_lists
            ]

    def classify(self, prompt: str, classification: Optional[str] = None) -> SemanticRouteDecision:
        """Classify prompt intent and determine optimal route."""
        text = extract_content_text(prompt).strip()
        tokens = _tokenize(text)
        query_vec = _build_term_vector(tokens, self._idf)

        detected_features: List[str] = []
        scores: Dict[str, float] = {k: 0.05 for k in TASK_ROUTES.keys()}

        # 1. Pattern-based prior scoring
        for task_name, pattern in PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                weight = min(len(matches) * 0.35, 0.85)
                scores[task_name] = scores.get(task_name, 0.0) + weight
                detected_features.append(f"regex:{task_name}({len(matches)})")

        # 2. TF-IDF Semantic exemplar similarity scoring
        if query_vec:
            for task_name, vectors in self._exemplar_vectors.items():
                sims = [_cosine_similarity(query_vec, v) for v in vectors]
                max_sim = max(sims) if sims else 0.0
                if max_sim > 0.15:
                    scores[task_name] = scores.get(task_name, 0.0) + (max_sim * 0.5)
                    detected_features.append(f"exemplar:{task_name}({max_sim:.2f})")

        # 3. Fast chat heuristics for very short conversational queries
        word_count = len(text.split())
        if word_count <= 6 and not any(
            f.startswith(("regex:coding", "regex:math", "regex:multilingual", "regex:governance"))
            for f in detected_features
        ):
            scores["fast_chat"] = scores.get("fast_chat", 0.0) + 0.45
            detected_features.append("heuristic:short_chat")

        # Sort tasks by score
        ranked_tasks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_task, top_score = ranked_tasks[0]

        # Normalize confidence to [0.0, 1.0]
        confidence = min(round(top_score / (top_score + 0.5), 2), 0.99)
        if top_score < 0.20:
            top_task = "general"
            confidence = 0.50

        routes = TASK_ROUTES.get(top_task, TASK_ROUTES["general"])
        primary_group = routes[0]

        justification = (
            f"Cognitive router classified as '{top_task}' (confidence: {confidence:.2f}) "
            f"based on features: [{', '.join(detected_features[:4])}]. "
            f"Primary route: {primary_group}."
        )

        return SemanticRouteDecision(
            task=top_task,
            confidence=confidence,
            model_group=primary_group,
            route_candidates=routes,
            justification=justification,
            features=detected_features,
        )


semantic_router = SemanticRouter()
