# -*- coding: utf-8 -*-
"""
Synthetic StepFact Dataset Generator

Generates synthetic reasoning trails with claims labeled as True, False, or Neutral.
"""

import random
import pandas as pd
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    n_trails: int = 600
    claims_per_trail: int = 4
    seed: int = 20261112

    # Entity generation
    first_names: List[str] = field(default_factory=lambda: [
        "Alden", "Mira", "Tarin", "Selene", "Orin", "Kael", "Neris", "Veya",
        "Dorian", "Elara", "Riven", "Maelis", "Corin", "Liora", "Soren", "Anika"
    ])
    last_names: List[str] = field(default_factory=lambda: [
        "Vale", "Morrow", "Kestrel", "Arden", "Solari", "Fenwick", "Thorne", "Ashford"
    ])
    org_prefix: List[str] = field(default_factory=lambda: [
        "Northstar", "Silverline", "Cedar", "Bluehaven", "Meridian",
        "Pioneer", "Juniper", "Aurora", "Westbridge", "Keystone"
    ])
    org_suffix: List[str] = field(default_factory=lambda: [
        "Research", "Institute", "Laboratory", "Collective", "Foundation",
        "Systems", "Academy", "Works"
    ])
    city_prefix: List[str] = field(default_factory=lambda: [
        "Luma", "Veyra", "Orlen", "Caldris", "Maren", "Tavik",
        "Eldora", "Neris", "Solven", "Aster"
    ])
    city_suffix: List[str] = field(default_factory=lambda: [
        "City", "Bay", "Point", "Falls", "Ridge", "Harbor", "Vale", "Crossing"
    ])


@dataclass
class Trail:
    """A reasoning trail with its evidence and steps."""
    trail_id: str
    question: str
    reference_answer: str
    evidence: str
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)


def generate_synthetic_data(config: SyntheticConfig = None) -> tuple:
    """
    Generate synthetic evidence and reasoning trails.

    Returns:
        tuple: (dataframe of rows, list of trail dicts)
    """
    if config is None:
        config = SyntheticConfig()

    rng = random.Random(config.seed)
    trails = []
    rows = []

    for i in range(config.n_trails):
        person = f"{rng.choice(config.first_names)} {rng.choice(config.last_names)}-{i+1:04d}"
        org = f"{rng.choice(config.org_prefix)} {rng.choice(config.org_suffix)} {i+1:04d}"
        city = f"{rng.choice(config.city_prefix)} {rng.choice(config.city_suffix)}"

        # Ensure wrong_city is different
        wrong_city = f"{rng.choice(config.city_prefix)} {rng.choice(config.city_suffix)}"
        while wrong_city == city:
            wrong_city = f"{rng.choice(config.city_prefix)} {rng.choice(config.city_suffix)}"

        founded_year = rng.randint(1980, 2023)
        director = f"{rng.choice(config.first_names)} {rng.choice(config.last_names)}-{i+1:04d}"

        evidence = (
            f"Fictional record {i+1:04d}: {person} founded {org} in {founded_year}. "
            f"{org} is based in {city}. The director of {org} is {director}. "
            f"The record does not provide an annual operating budget."
        )
        question = f"According to the record, where is {org} based?"

        claims = [
            (f"{person} founded {org} in {founded_year}.", "True",
             "The evidence explicitly states the founder and founding year."),
            (f"{org} is based in {city}.", "True",
             "The evidence explicitly states this location."),
            (f"{org} is based in {wrong_city}.", "False",
             "The claim gives a location that conflicts with the location in the evidence."),
            (f"The annual operating budget of {org} is {rng.randint(2, 90)} million credits.", "Neutral",
             "The evidence does not state an annual operating budget.")
        ]
        rng.shuffle(claims)

        trail_id = f"SC{i+1:04d}"
        trail_steps = []
        for j, (claim, label, rationale) in enumerate(claims, start=1):
            step_id = f"{trail_id}_S{j}"
            record = {
                "trail_id": trail_id,
                "step_id": step_id,
                "question": question,
                "evidence": evidence,
                "claim": claim,
                "gold_label": label,
                "annotation_rationale": rationale
            }
            rows.append(record)
            trail_steps.append(record)

        trails.append({
            "trail_id": trail_id,
            "question": question,
            "reference_answer": city,
            "evidence": evidence,
            "reasoning_steps": trail_steps
        })

    df = pd.DataFrame(rows)
    return df, trails


def create_splits(df: pd.DataFrame, seed: int = 20261112) -> pd.DataFrame:
    """
    Create leakage-safe train/val/test splits by trail_id.

    Args:
        df: DataFrame with trail_id column
        seed: Random seed for reproducibility

    Returns:
        DataFrame with 'split' column added
    """
    # Ensure df is a regular Pandas DataFrame
    if isinstance(df, type(None)):
        raise ValueError("DataFrame is None")

    df = df.reset_index(drop=True)

    # Verify required columns
    required_columns = {
        "trail_id", "step_id", "evidence", "claim",
        "gold_label", "annotation_rationale"
    }
    missing_columns = required_columns - set(df.columns)
    assert not missing_columns, f"Dataset is missing columns: {sorted(missing_columns)}"

    # Label composition audit
    expected_counts = {"True": 2, "False": 1, "Neutral": 1}
    for trail_id, group in df.groupby("trail_id"):
        counts = group["gold_label"].value_counts().to_dict()
        # Handle both int and str keys
        counts = {str(k): v for k, v in counts.items()}
        assert counts == expected_counts, (
            f"Unexpected label composition in {trail_id}: {counts}"
        )

    # Location-claim consistency audit
    import re
    for row in df.itertuples():
        if "is based in" in row.claim:
            claim_city = row.claim.split("is based in", 1)[1].strip().rstrip(".")
            evidence_match = re.search(
                r"is based in (.*?)[.]",
                row.evidence
            )
            assert evidence_match is not None, f"Location missing from evidence: {row.step_id}"
            evidence_city = evidence_match.group(1).strip()
            if row.gold_label == "True":
                assert claim_city == evidence_city, (
                    row.step_id, row.gold_label, claim_city, evidence_city
                )
            elif row.gold_label == "False":
                assert claim_city != evidence_city, (
                    row.step_id, row.gold_label, claim_city, evidence_city
                )

    # Split whole trails deterministically: 80% train, 10% validation, 10% test
    trail_ids = sorted(df["trail_id"].astype(str).unique().tolist())
    split_rng = random.Random(seed)
    split_rng.shuffle(trail_ids)

    n_trails = len(trail_ids)
    n_train = int(n_trails * 0.80)
    n_val = int(n_trails * 0.10)

    train_ids = set(trail_ids[:n_train])
    val_ids = set(trail_ids[n_train:n_train + n_val])
    test_ids = set(trail_ids[n_train + n_val:])

    def assign_split(tid):
        tid = str(tid)
        if tid in train_ids:
            return "train"
        if tid in val_ids:
            return "validation"
        return "test"

    df = df.copy()
    df["split"] = df["trail_id"].astype(str).map(assign_split)

    # Verify no leakage
    assert df.groupby("trail_id")["split"].nunique().max() == 1
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    return df


__all__ = [
    "SyntheticConfig",
    "Trail",
    "generate_synthetic_data",
    "create_splits"
]