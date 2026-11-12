# -*- coding: utf-8 -*-
"""
Visualization utilities for StepCheck

Includes plotting functions for retrieval quality, reward distributions, and comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


def plot_retrieval_quality(
    retrieval_metrics: Dict[str, float],
    title: str = "Synthetic evidence retrieval"
):
    """Plot BM25 retrieval quality."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["BM25 top-1", "BM25 top-3"],
        [
            retrieval_metrics["top1_source_match_rate"],
            retrieval_metrics["top3_source_match_rate"]
        ]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Source-trail match rate")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_chain_reward_distribution(
    chain_results: pd.DataFrame,
    title: str = "Synthetic chain reward distribution"
):
    """Plot distribution of chain rewards."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(chain_results["mean_soft_fact_reward"], bins=20)
    ax.set_xlabel("Mean soft factuality reward per trail")
    ax.set_ylabel("Number of trails")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_reward_comparison(
    reward_comparison: pd.DataFrame,
    plot_cols: List[str] = None,
    title: str = "Reward variant comparison (not policy performance)"
):
    """Plot mean reward across different variants."""
    if plot_cols is None:
        plot_cols = [
            "answer_only_baseline",
            "grpo_plus_soft_fact_reward",
            "stepcheck_hard_reward",
            "stepcheck_soft_reward"
        ]

    fig, ax = plt.subplots(figsize=(8, 4))

    reward_comparison[plot_cols].mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean reward signal")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_three_model_comparison(
    three_model_comparison: pd.DataFrame,
    title: str = "Base vs Plain GRPO vs StepCheck-GRPO"
):
    """Plot comparison of three models across multiple metrics."""
    plot_df = three_model_comparison.copy()

    x = np.arange(len(plot_df))
    width = 0.25

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width,
        plot_df["Strict Accuracy"] * 100,
        width,
        label="Strict Accuracy"
    )

    plt.bar(
        x,
        plot_df["Location Mention Accuracy"] * 100,
        width,
        label="Location Mention Accuracy"
    )

    plt.bar(
        x + width,
        plot_df["Format Validity"] * 100,
        width,
        label="Format Validity"
    )

    plt.xticks(x, plot_df["Model"], rotation=10)
    plt.ylabel("Percentage (%)")
    plt.title(title)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_checker_reward_comparison(
    reward_distributions: Dict[str, float],
    title: str = "Checker-based reward comparison"
):
    """Plot mean step soft reward across models."""
    plt.figure(figsize=(9, 5))

    plt.bar(
        reward_distributions.keys(),
        reward_distributions.values()
    )

    plt.ylabel("Mean step soft reward")
    plt.title(title)
    plt.ylim(-1, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.show()


def create_comparison_heatmap(
    data: pd.DataFrame,
    title: str = "Model Comparison Heatmap"
):
    """Create a heatmap of model metrics."""
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt=".4f", cmap="YlOrRd")
    plt.title(title)
    plt.tight_layout()
    plt.show()


__all__ = [
    "plot_retrieval_quality",
    "plot_chain_reward_distribution",
    "plot_reward_comparison",
    "plot_three_model_comparison",
    "plot_checker_reward_comparison",
    "create_comparison_heatmap",
]