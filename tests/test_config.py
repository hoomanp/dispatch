"""Validate litellm_config.yaml and autonomy_boundary.yaml internal consistency."""
import sys, yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "dispatcher"))

CFG = yaml.safe_load((ROOT / "config/litellm_config.yaml").read_text())
AUT = yaml.safe_load((ROOT / "config/autonomy_boundary.yaml").read_text())


def defined_groups():
    return {m["model_name"] for m in CFG["model_list"]}


def test_yaml_parses():
    assert CFG and AUT


def test_every_model_has_tier():
    for m in CFG["model_list"]:
        tier = m.get("model_info", {}).get("tier", "")
        assert tier.split("-")[0] in {"0", "1", "2", "3"}, \
            f"{m['model_name']} missing/invalid tier: {tier!r}"


def test_group_prefix_matches_tier():
    prefix_tier = {"local": "0", "free": "1", "budget": "2", "premium": "3"}
    for m in CFG["model_list"]:
        prefix = m["model_name"].split("-")[0]
        tier = m["model_info"]["tier"].split("-")[0]
        assert prefix_tier[prefix] == tier, \
            f"{m['model_name']}: prefix implies tier {prefix_tier[prefix]}, config says {tier}"


def test_fallbacks_reference_defined_groups():
    groups = defined_groups()
    for entry in CFG["router_settings"]["fallbacks"]:
        for src, targets in entry.items():
            assert src in groups, f"fallback source '{src}' undefined"
            for t in targets:
                assert t in groups, f"fallback target '{t}' undefined (from {src})"


def test_context_fallbacks_reference_defined_groups():
    groups = defined_groups()
    for entry in CFG["router_settings"]["context_window_fallbacks"]:
        for src, targets in entry.items():
            assert src in groups and all(t in groups for t in targets)


def test_task_routes_reference_defined_groups():
    try:
        from main import TASK_ROUTES
    except (ImportError, ModuleNotFoundError):
        import ast
        main_src = (ROOT / "dispatcher/main.py").read_text()
        tree = ast.parse(main_src)
        TASK_ROUTES = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "TASK_ROUTES":
                        TASK_ROUTES = ast.literal_eval(node.value)
    groups = defined_groups()
    for task, chain in TASK_ROUTES.items():
        assert chain, f"task '{task}' has empty route"
        for g in chain:
            assert g in groups, f"TASK_ROUTES['{task}'] references undefined group '{g}'"


def test_cache_embedding_model_defined():
    embed = CFG["litellm_settings"]["cache_params"][
        "qdrant_semantic_cache_embedding_model"]
    assert embed in defined_groups()


def test_autonomy_ceilings_valid():
    for cls, ceil in AUT["classification_tier_ceiling"].items():
        assert ceil in (0, 1, 2, 3), f"{cls}: invalid ceiling {ceil}"
    assert AUT["classification_tier_ceiling"]["regulated"] == 0, \
        "regulated data must be local-only"
    for action, level in AUT["actions"].items():
        assert level in ("autonomous", "approval", "forbidden"), \
            f"{action}: invalid level {level}"
