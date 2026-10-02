"""Scenario metadata and directory names for the original experiment Config."""
import math


def scenario_fields(scenario, background_ratio=None, defense=None):
    """Validate the three campaign scenarios without importing training code."""
    defense = dict(defense or {})
    if scenario not in ("closed-world", "open-world", "defense"):
        raise ValueError("Unknown experiment scenario")
    if scenario == "open-world":
        if defense or not isinstance(background_ratio, (int, float)) or isinstance(background_ratio, bool):
            raise ValueError("Open world requires a background ratio and no defense")
        if not math.isfinite(background_ratio) or not 0 < background_ratio < 1:
            raise ValueError("background_ratio must be strictly between zero and one")
    elif background_ratio is not None:
        raise ValueError("Background and defense/closed-world cannot be combined")
    if scenario == "defense":
        if set(defense) not in ({"padding"}, {"dummy"}, {"fragmentation"},
                               {"padding", "dummy", "fragmentation"}):
            raise ValueError("Use one defense or all three at equal intensity")
        values = list(defense.values())
        if any(isinstance(p, bool) or not isinstance(p, (int, float))
               or not math.isfinite(p) or not 0 < p <= 1 for p in values):
            raise ValueError("Defense probabilities must be in (0, 1]")
        if len(set(values)) != 1:
            raise ValueError("Combined defense requires equal probabilities")
    elif defense:
        raise ValueError("Defense requires scenario='defense' and no background")
    return {"scenario": scenario, "background_ratio": background_ratio,
            "defense": defense,
            "defense_type": ("combined" if len(defense) == 3 else next(iter(defense)))
                            if defense else None,
            "defense_intensity": next(iter(defense.values())) if defense else None}


def scenario_tag(fields):
    """Use the campaign's cell names, with a separate baseline cell."""
    def probability(value):
        return f"{value:.2f}" if value == round(value, 2) else str(value)

    if fields["scenario"] == "open-world":
        return f"openworld_r{probability(fields['background_ratio'])}"
    if fields["scenario"] == "defense":
        return f"defense_{fields['defense_type']}_p{probability(fields['defense_intensity'])}"
    return "baseline"


def experiment_metadata(cfg):
    """Describe actual Config values; the original Config owns the seed rule."""
    if cfg.n1_head:
        raise ValueError("Explainability bundles require the original two-head FOLD")
    if (isinstance(cfg.experiment_id, bool) or not isinstance(cfg.experiment_id, int)
            or cfg.experiment_id < 0 or cfg.seed > 2**32 - 1):
        raise ValueError("experiment_id must produce a valid nonnegative random seed")
    fields = scenario_fields(cfg.scenario,
                             cfg.background_ratio if cfg.is_open_world else None,
                             cfg.defense)
    return {**fields, "provenance_version": 1, "forest_role": "final",
            "experiment_id": cfg.experiment_id, "seed": cfg.seed,
            "seeds": {"balance": 0, "split": cfg.seed, "initialization": cfg.seed,
                      "defense": cfg.seed if cfg.is_defended else None,
                      "background_sampling": cfg.seed if cfg.is_open_world else None,
                      "forest_final": cfg.experiment_id, "oof_split": cfg.experiment_id,
                      "oof_forests": [cfg.experiment_id + i for i in range(5)]},
            "training": {"epochs": cfg.epochs, "batch_size": cfg.batch_size}}


def validate_split(split):
    """Validate ordered, disjoint IDs bound to one dataset file."""
    if not isinstance(split, dict):
        raise ValueError("split_ids must be a dictionary")
    digest = split.get("dataset_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Split requires a SHA-256 dataset fingerprint")
    seen = set()
    for name in ("train", "val", "test"):
        ids = split.get(name)
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(tid, str) or not tid for tid in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValueError("Split IDs must be nonempty, unique and disjoint")
        seen.update(ids)
    return split
