"""
HarmonyShield Consensus Decision Engine

Implements majority voting across the three detection methods:
1. Classical Random Forest (Feature Set B, 22 spectral features)
2. ResNet18 LOGO Ensemble (5 frozen checkpoints on Mel-spectrograms)
3. Pretrained AST Baseline (Audio Spectrogram Transformer, 60s windows)

Decision Policy (Strict Majority Voting):
- 3 models predict AI-generated -> AI-GENERATED ("AI-Generated")
- 2 models predict AI-generated and 1 predicts Human -> LIKELY AI-GENERATED ("Likely AI-Generated")
- 2 models predict Human and 1 predicts AI-generated -> LIKELY HUMAN-CREATED ("Likely Human-Created")
- 3 models predict Human -> HUMAN-CREATED ("Human-Created")
- Fewer than 2 valid model predictions -> UNABLE TO DETERMINE ("Unable to Determine")

Threshold:
- Default threshold: 0.5
- AI if P(AI) >= threshold, else Human.
- Diagnostic telemetry (mean probability, variance, disagreement) is preserved
  for transparency but does not override the majority decision.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np

DEFAULT_THRESHOLD: float = 0.5


def is_valid_prob(p: Any) -> bool:
    """
    Checks whether a probability value is a valid numeric float in [0.0, 1.0].
    Safely rejects None, NaN, Inf, non-numeric types, and out-of-range values.
    """
    if p is None:
        return False
    try:
        val = float(p)
        return not (np.isnan(val) or np.isinf(val)) and 0.0 <= val <= 1.0
    except (TypeError, ValueError):
        return False


def compute_majority_consensus(
    classical_prob: Optional[float] = None,
    resnet_prob: Optional[float] = None,
    ast_prob: Optional[float] = None,
    threshold: float = DEFAULT_THRESHOLD,
    model_entries: Optional[List[Tuple[str, Optional[float]]]] = None,
) -> Dict[str, Any]:
    """
    Computes the final forensic classification using majority voting.

    Args:
        classical_prob: Probability of AI from Classical Random Forest (or None)
        resnet_prob: Probability of AI from ResNet18 ensemble (or None)
        ast_prob: Probability of AI from Pretrained AST (or None)
        threshold: Classification threshold (defaults to 0.5)
        model_entries: Optional explicit list of (model_name, probability) tuples

    Returns:
        Dict containing:
            - status: Machine-readable status code
            - label: Human-readable primary verdict
            - sub: Short explanatory badge (e.g. "MAJORITY VOTE (2-TO-1)")
            - message: Descriptive assessment text
            - vote_summary: Human-readable vote tally (e.g. "2 of 3 models indicate AI-generated.")
            - num_ai_votes: Integer count of AI votes
            - num_human_votes: Integer count of Human votes
            - total_valid_votes: Number of functioning models
            - mean_ai_probability: Mean AI probability across valid models
            - variance: Statistical variance of valid probabilities
            - threshold_used: Threshold applied (0.5)
            - disagreement: Boolean indicating whether models disagreed
    """
    if model_entries is None:
        model_entries = [
            ("Classical Random Forest", classical_prob),
            ("ResNet18 Ensemble", resnet_prob),
            ("Pretrained AST", ast_prob),
        ]

    # Filter out missing, failed, or invalid probabilities
    valid_entries = []
    for name, prob in model_entries:
        if is_valid_prob(prob):
            p = float(prob)
            pred = "AI" if p >= threshold else "Human"
            valid_entries.append({"name": name, "prob": p, "pred": pred})

    total_valid = len(valid_entries)
    num_ai = sum(1 for m in valid_entries if m["pred"] == "AI")
    num_human = sum(1 for m in valid_entries if m["pred"] == "Human")

    valid_probs = [m["prob"] for m in valid_entries]
    mean_prob = float(np.mean(valid_probs)) if valid_probs else None
    variance = float(np.var(valid_probs)) if valid_probs else None

    # Disagreement occurs if probabilities show high variance (>0.05) or votes are split
    disagreement = bool(
        (variance is not None and variance > 0.05)
        or (num_ai > 0 and num_human > 0)
    )

    # 1. Fewer than two valid model predictions
    if total_valid < 2:
        return {
            "status": "UNABLE_TO_DETERMINE",
            "label": "Unable to Determine",
            "sub": "INSUFFICIENT MODEL EVIDENCE",
            "message": (
                "Fewer than two models produced valid predictions. HarmonyShield "
                "requires at least two functioning detection methods to determine a majority vote."
            ),
            "vote_summary": (
                f"{num_ai} of {total_valid} models indicate AI-generated."
                if total_valid > 0
                else "No valid model predictions available."
            ),
            "num_ai_votes": num_ai,
            "num_human_votes": num_human,
            "total_valid_votes": total_valid,
            "mean_ai_probability": mean_prob,
            "variance": variance,
            "threshold_used": threshold,
            "disagreement": disagreement,
        }

    # 2. Tied vote (e.g. 1 AI vs 1 Human when only 2 models succeed)
    if num_ai == num_human:
        return {
            "status": "UNABLE_TO_DETERMINE",
            "label": "Unable to Determine",
            "sub": "TIED VOTE (1-TO-1)",
            "message": (
                "The two functioning models are tied (1 AI vote vs 1 Human vote). "
                "A majority decision cannot be established."
            ),
            "vote_summary": "1 of 2 models indicates AI-generated (tie).",
            "num_ai_votes": num_ai,
            "num_human_votes": num_human,
            "total_valid_votes": total_valid,
            "mean_ai_probability": mean_prob,
            "variance": variance,
            "threshold_used": threshold,
            "disagreement": True,
        }

    # 3. Unanimous AI vote (3-0, or 2-0 when only 2 models are available)
    if num_ai == total_valid:
        return {
            "status": "AI_GENERATED",
            "label": "AI-Generated",
            "sub": "UNANIMOUS VOTE",
            "message": (
                f"All {total_valid} models unanimously indicate AI-generated characteristics. "
                "Note that majority voting is a model-based assessment, not definitive proof of authenticity."
            ),
            "vote_summary": f"{num_ai} of {total_valid} models indicate AI-generated.",
            "num_ai_votes": num_ai,
            "num_human_votes": num_human,
            "total_valid_votes": total_valid,
            "mean_ai_probability": mean_prob,
            "variance": variance,
            "threshold_used": threshold,
            "disagreement": disagreement,
        }

    # 4. Unanimous Human vote (0-3, or 0-2 when only 2 models are available)
    if num_human == total_valid:
        return {
            "status": "HUMAN_CREATED",
            "label": "Human-Created",
            "sub": "UNANIMOUS VOTE",
            "message": (
                f"All {total_valid} models unanimously indicate human-composed characteristics. "
                "Note that majority voting is a model-based assessment, not definitive proof of authenticity."
            ),
            "vote_summary": f"{num_human} of {total_valid} models indicate human-created (0 of {total_valid} indicate AI-generated).",
            "num_ai_votes": num_ai,
            "num_human_votes": num_human,
            "total_valid_votes": total_valid,
            "mean_ai_probability": mean_prob,
            "variance": variance,
            "threshold_used": threshold,
            "disagreement": disagreement,
        }

    # 5. 2-to-1 Majority AI vote
    if num_ai > num_human:
        return {
            "status": "LIKELY_AI_GENERATED",
            "label": "Likely AI-Generated",
            "sub": "MAJORITY VOTE (2-TO-1)",
            "message": (
                f"{num_ai} of {total_valid} models indicate AI-generated patterns while "
                f"{num_human} indicates human-composed. Disagreement between models is expected "
                "across diverse acoustic architectures; this result reflects a majority-model assessment, not definitive proof."
            ),
            "vote_summary": f"{num_ai} of {total_valid} models indicate AI-generated.",
            "num_ai_votes": num_ai,
            "num_human_votes": num_human,
            "total_valid_votes": total_valid,
            "mean_ai_probability": mean_prob,
            "variance": variance,
            "threshold_used": threshold,
            "disagreement": True,
        }

    # 6. 2-to-1 Majority Human vote
    return {
        "status": "LIKELY_HUMAN_CREATED",
        "label": "Likely Human-Created",
        "sub": "MAJORITY VOTE (2-TO-1)",
        "message": (
            f"{num_human} of {total_valid} models indicate human-composed patterns while "
            f"{num_ai} indicates AI-generated. Disagreement between models is expected "
            "across diverse acoustic architectures; this result reflects a majority-model assessment, not definitive proof."
        ),
        "vote_summary": f"{num_human} of {total_valid} models indicate human-created ({num_ai} of {total_valid} indicates AI-generated).",
        "num_ai_votes": num_ai,
        "num_human_votes": num_human,
        "total_valid_votes": total_valid,
        "mean_ai_probability": mean_prob,
        "variance": variance,
        "threshold_used": threshold,
        "disagreement": True,
    }
