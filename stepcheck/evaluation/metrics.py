# -*- coding: utf-8 -*-
"""
Evaluation Metrics for StepCheck

Includes accuracy, F1, confusion matrix analysis, and reward calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
import re


def calculate_model_metrics(
    results_df: pd.DataFrame,
    checker_fn,
    label2id: Dict[str, int]
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for model results.

    Args:
        results_df: DataFrame with 'predicted_answer', 'reference_answer', 'generated_response'
        checker_fn: Function to check claims against evidence
        label2id: Mapping from label strings to IDs

    Returns:
        Dictionary of metrics
    """
    answer_correct = []
    format_valid = []
    step_rewards = []
    step_true_probabilities = []

    for _, row in results_df.iterrows():
        predicted = normalize_answer(row["predicted_answer"])
        expected = normalize_answer(row["reference_answer"])

        answer_correct.append(
            int(bool(predicted) and predicted == expected)
        )

        steps = row["reasoning_steps"]
        has_final = bool(row["predicted_answer"])

        format_valid.append(
            int(len(steps) >= 1 and has_final)
        )

        for step in steps:
            checker_result = checker_fn(row["evidence"], step)
            step_rewards.append(checker_result["soft_fact_reward"])
            step_true_probabilities.append(checker_result["p_true"])

    return {
        "answer_accuracy": float(np.mean(answer_correct)),
        "format_validity": float(np.mean(format_valid)),
        "mean_step_soft_reward": (
            float(np.mean(step_rewards)) if step_rewards else 0.0
        ),
        "mean_step_true_probability": (
            float(np.mean(step_true_probabilities))
            if step_true_probabilities else 0.0
        ),
        "total_reasoning_steps": len(step_rewards),
        "total_examples": len(results_df)
    }


def recalculate_answer_accuracy(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Recalculate answer accuracy from extracted answers.

    Returns:
        Tuple of (results DataFrame with correct column, accuracy float)
    """
    predictions = []

    for _, row in results_df.iterrows():
        extracted = extract_final_answer(row["generated_response"])
        correct = (
            normalize_answer(extracted)
            == normalize_answer(row["reference_answer"])
        )

        predictions.append({
            "reference_answer": row["reference_answer"],
            "extracted_answer": extracted,
            "correct": correct
        })

    result_df = pd.DataFrame(predictions)
    accuracy = result_df["correct"].mean()

    return result_df, float(accuracy)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = str(answer).lower().strip()
    answer = re.sub(r"[^\w\s]", "", answer)
    answer = re.sub(r"\s+", " ", answer)
    return answer


def extract_final_answer(text: str) -> str:
    """Extract the final answer from generated text."""
    match = re.search(r"FINAL\s*:\s*(.+)", text, flags=re.IGNORECASE)

    if not match:
        return ""

    answer = match.group(1).strip().splitlines()[0]
    answer = answer.strip(" \t\n\"'`.,;:")

    patterns = [
        r"\bis based in\s+(.+)",
        r"\bis located in\s+(.+)",
        r"\bis based at\s+(.+)",
        r"\blocated at\s+(.+)",
        r"\bthe location is\s+(.+)",
        r"\banswer is\s+(.+)"
    ]

    for pattern in patterns:
        result = re.search(pattern, answer, flags=re.IGNORECASE)
        if result:
            answer = result.group(1).strip()
            break

    return answer.strip(" \t\n\"'`.,;:")


def location_mention_match(prediction: str, reference: str) -> bool:
    """Check if the prediction mentions the correct location."""
    prediction = normalize_answer(prediction)
    reference = normalize_answer(reference)

    if not prediction or not reference:
        return False

    pattern = r"(?<!\w)" + re.escape(reference) + r"(?!\w)"
    return bool(re.search(pattern, prediction))


def evaluate_answer_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate answer metrics including strict accuracy and location mention.

    Returns:
        Dictionary with strict_accuracy, location_mention_accuracy, total_examples
    """
    rows = []

    for _, row in results_df.iterrows():
        extracted = extract_final_answer(row["generated_response"])
        reference = row["reference_answer"]

        rows.append({
            "strict_match": (
                normalize_answer(extracted)
                == normalize_answer(reference)
            ),
            "location_mention": location_mention_match(
                extracted, reference
            )
        })

    metrics_df = pd.DataFrame(rows)

    return {
        "strict_accuracy": metrics_df["strict_match"].mean(),
        "location_mention_accuracy": metrics_df["location_mention"].mean(),
        "total_examples": len(metrics_df)
    }


def evaluate_checker(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluate classifier predictions.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label names (default: ["False", "Neutral", "True"])

    Returns:
        Dictionary with accuracy, F1, classification report, confusion matrix
    """
    if labels is None:
        labels = ["False", "Neutral", "True"]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels,
            digits=4, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels)
    }


def calculate_error_analysis(
    stepcheck_results_df: pd.DataFrame,
    stepcheck_answer_results: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Perform error analysis on model predictions.

    Returns:
        Tuple of (error analysis DataFrame, error type counts)
    """
    error_analysis = stepcheck_results_df.copy()

    error_analysis["extracted_answer"] = (
        stepcheck_answer_results["extracted_answer"].values
    )
    error_analysis["correct"] = (
        stepcheck_answer_results["correct"].values
    )

    def classify_error(row):
        if row["correct"]:
            return "Correct"

        if not str(row["extracted_answer"]).strip():
            return "Missing final answer"

        reference = normalize_answer(row["reference_answer"])
        extracted = normalize_answer(row["extracted_answer"])

        if reference and reference in extracted:
            return "Correct location embedded in extra text"

        return "Wrong or unsupported answer"

    error_analysis["error_type"] = error_analysis.apply(classify_error, axis=1)

    return error_analysis, error_analysis["error_type"].value_counts()


def calculate_false_acceptance_rejection(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label2id: Dict[str, int]
) -> Dict[str, float]:
    """
    Calculate false acceptance and false rejection rates.

    - False acceptance: gold-False predicted as True
    - False rejection: gold-True predicted as False
    """
    false_accept = int(((y_true == label2id["False"]) &
                          (y_pred == label2id["True"])).sum())
    false_reject = int(((y_true == label2id["True"]) &
                         (y_pred == label2id["False"])).sum())

    n_gold_false = int((y_true == label2id["False"]).sum())
    n_gold_true = int((y_true == label2id["True"]).sum())

    far = false_accept / n_gold_false if n_gold_false else float("nan")
    frr = false_reject / n_gold_true if n_gold_true else float("nan")

    return {
        "false_acceptance": false_accept,
        "false_acceptance_rate": far,
        "false_rejection": false_reject,
        "false_rejection_rate": frr,
        "n_gold_false": n_gold_false,
        "n_gold_true": n_gold_true
    }


def compare_models(
    model_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compare metrics across multiple models.

    Args:
        model_results: Dict mapping model names to their metrics

    Returns:
        DataFrame with comparison
    """
    rows = []
    for model_name, metrics in model_results.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


__all__ = [
    "calculate_model_metrics",
    "recalculate_answer_accuracy",
    "classify_error",
    "location_mention_match",
    "evaluate_answer_metrics",
    "evaluate_checker",
    "calculate_error_analysis",
    "calculate_false_acceptance_rejection",
    "compare_models",
    "normalize_answer",
    "extract_final_answer",
]