# -*- coding: utf-8 -*-
"""
GRPO (Generalized Relative Policy Optimization) Reward Functions

Implements StepCheck reward functions for training language models.
"""

import re
import numpy as np
from typing import Dict, Any, List, Optional, Callable


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = str(answer).lower().strip()
    answer = re.sub(r"[^\w\s]", "", answer)
    answer = re.sub(r"\s+", " ", answer)
    return answer


def get_completion_text(completion) -> str:
    """Extract text from various completion formats."""
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
        return "\n".join(parts)

    if isinstance(completion, dict):
        return str(completion.get("content", ""))

    return str(completion)


def extract_final_answer(text: str) -> str:
    """Extract the final answer from generated text."""
    match = re.search(r"FINAL\s*:\s*(.+)", text, flags=re.IGNORECASE)

    if not match:
        return ""

    answer = match.group(1).strip().splitlines()[0]

    # Remove quotation marks and surrounding punctuation
    answer = answer.strip(" \t\n\"'`.,;:")

    # Extract location after common patterns
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


def extract_reasoning_steps(text: str) -> List[str]:
    """Extract reasoning steps from generated text."""
    matches = re.findall(r"STEP\s*:\s*(.+)", text, flags=re.IGNORECASE)

    return [
        step.strip()
        for step in matches
        if step.strip()
    ]


def stepcheck_soft_reward(p_true: float, p_false: float) -> float:
    """
    Calculate StepCheck soft factuality reward.

    r_j = P_true(z_j) - P_false(z_j)
    """
    r = np.asarray(p_true) - np.asarray(p_false)
    return np.clip(r, -1.0, 1.0)


def plain_grpo_reward(
    completions: List[str],
    reference_answer: List[str],
    **kwargs
) -> List[float]:
    """
    Simple GRPO reward function based only on answer correctness.

    Returns 1.0 for correct, -1.0 for incorrect.
    """
    rewards = []

    for completion, expected_answer in zip(completions, reference_answer):
        text = get_completion_text(completion)
        predicted_answer = extract_final_answer(text)

        predicted = normalize_answer(predicted_answer)
        expected = normalize_answer(expected_answer)

        reward = 1.0 if predicted and predicted == expected else -1.0
        rewards.append(float(reward))

    return rewards


def create_stepcheck_grpo_reward(check_claim_fn: Callable) -> Callable:
    """
    Create a StepCheck GRPO reward function.

    Args:
        check_claim_fn: Function that takes (evidence, claim) and returns
                       dict with p_false, p_neutral, p_true, soft_fact_reward

    Returns:
        Reward function suitable for GRPOTrainer
    """

    def stepcheck_grpo_reward(
        completions: List[str],
        reference_answer: List[str],
        evidence: List[str],
        **kwargs
    ) -> List[float]:
        rewards = []

        for completion, expected_answer, source_evidence in zip(
            completions, reference_answer, evidence
        ):
            text = get_completion_text(completion)

            # Answer correctness reward
            predicted_answer = extract_final_answer(text)
            answer_reward = 1.0 if normalize_answer(predicted_answer) == \
                                   normalize_answer(expected_answer) else -1.0

            # Soft factuality reward from the trained DeBERTa checker
            reasoning_steps = extract_reasoning_steps(text)

            if reasoning_steps:
                step_rewards = []
                for step in reasoning_steps:
                    result = check_claim_fn(source_evidence, step)
                    step_rewards.append(result["soft_fact_reward"])

                factuality_reward = float(np.mean(step_rewards))
            else:
                factuality_reward = 0.0

            # Combined StepCheck reward
            total_reward = answer_reward + factuality_reward
            rewards.append(float(total_reward))

        return rewards

    return stepcheck_grpo_reward


def stepcheck_adjust_advantage(base_advantage: float, step_reward: float) -> float:
    """
    Calculate StepCheck advantage adjustment.

    A_hat_{i,t} = (1 - |r_j|) * A_i + |r_j| * |A_i| * sign(r_j)
    """
    r = float(np.clip(step_reward, -1.0, 1.0))
    w = abs(r)
    return float((1.0 - w) * base_advantage +
                 w * abs(base_advantage) * np.sign(r))


def adjust_step_advantages(
    sequence_advantage: float,
    step_rewards: Dict[int, float],
    token_step_ids: List[int]
) -> np.ndarray:
    """
    Apply step-level advantage adjustment to token-level advantages.

    Args:
        sequence_advantage: Scalar A_i for one completion
        step_rewards: Dict mapping step_index to r_j
        token_step_ids: List of step indices per token (-1 for non-step tokens)

    Returns:
        Array of adjusted advantages per token
    """
    adjusted = []

    for step_id in token_step_ids:
        if step_id not in step_rewards:
            adjusted.append(float(sequence_advantage))
            continue

        r_j = float(step_rewards[step_id])
        w_j = abs(r_j)

        adjusted_advantage = (
            (1.0 - w_j) * sequence_advantage
            + w_j * abs(sequence_advantage) * np.sign(r_j)
        )

        adjusted.append(float(adjusted_advantage))

    return np.asarray(adjusted, dtype=np.float32)


def response_reward(answer_correct: float, step_rewards: np.ndarray) -> float:
    """
    Calculate combined response reward.

    R(y) = R_ans(y) + (1/N) * sum(r_j)
    """
    step_rewards = np.asarray(step_rewards, dtype=np.float64)
    if step_rewards.size == 0:
        return float(answer_correct)
    return float(answer_correct) + float(step_rewards.mean())


__all__ = [
    "normalize_answer",
    "get_completion_text",
    "extract_final_answer",
    "extract_reasoning_steps",
    "stepcheck_soft_reward",
    "plain_grpo_reward",
    "create_stepcheck_grpo_reward",
    "stepcheck_adjust_advantage",
    "adjust_step_advantages",
    "response_reward",
]