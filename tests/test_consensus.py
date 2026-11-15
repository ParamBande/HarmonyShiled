"""
Unit and integration tests for HarmonyShield's Majority Voting Decision Logic.
Tests all five consensus scenarios, edge cases, failed models, and invalid probabilities.
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.consensus import (
    compute_majority_consensus,
    is_valid_prob,
    DEFAULT_THRESHOLD,
)
from backend.main import _consensus, ConsensusResult


class TestMajorityVotingConsensus:
    """Tests the five core voting scenarios."""

    def test_scenario_1_three_ai_predictions(self):
        """Scenario 1: 3 models predict AI -> AI-GENERATED (unanimous vote)."""
        res = compute_majority_consensus(0.90, 0.85, 0.95)
        assert res["status"] == "AI_GENERATED"
        assert res["label"] == "AI-Generated"
        assert res["sub"] == "UNANIMOUS VOTE"
        assert res["num_ai_votes"] == 3
        assert res["num_human_votes"] == 0
        assert res["total_valid_votes"] == 3
        assert "3 of 3 models indicate AI-generated." in res["vote_summary"]
        assert pytest.approx(res["mean_ai_probability"], 0.01) == 0.90
        assert res["threshold_used"] == DEFAULT_THRESHOLD

    def test_scenario_2_two_ai_one_human_prediction(self):
        """Scenario 2: 2 models predict AI, 1 predicts Human -> LIKELY AI-GENERATED (2-to-1)."""
        res = compute_majority_consensus(0.90, 0.75, 0.20)
        assert res["status"] == "LIKELY_AI_GENERATED"
        assert res["label"] == "Likely AI-Generated"
        assert res["sub"] == "MAJORITY VOTE (2-TO-1)"
        assert res["num_ai_votes"] == 2
        assert res["num_human_votes"] == 1
        assert res["total_valid_votes"] == 3
        assert "2 of 3 models indicate AI-generated." in res["vote_summary"]
        assert res["disagreement"] is True
        assert pytest.approx(res["mean_ai_probability"], 0.01) == 0.6167

    def test_scenario_3_two_human_one_ai_prediction(self):
        """Scenario 3: 2 models predict Human, 1 predicts AI -> LIKELY HUMAN-CREATED (2-to-1)."""
        res = compute_majority_consensus(0.15, 0.25, 0.85)
        assert res["status"] == "LIKELY_HUMAN_CREATED"
        assert res["label"] == "Likely Human-Created"
        assert res["sub"] == "MAJORITY VOTE (2-TO-1)"
        assert res["num_ai_votes"] == 1
        assert res["num_human_votes"] == 2
        assert res["total_valid_votes"] == 3
        assert "human-created" in res["vote_summary"]
        assert res["disagreement"] is True
        assert pytest.approx(res["mean_ai_probability"], 0.01) == 0.4167

    def test_scenario_4_three_human_predictions(self):
        """Scenario 4: 3 models predict Human -> HUMAN-CREATED (unanimous vote)."""
        res = compute_majority_consensus(0.10, 0.05, 0.25)
        assert res["status"] == "HUMAN_CREATED"
        assert res["label"] == "Human-Created"
        assert res["sub"] == "UNANIMOUS VOTE"
        assert res["num_ai_votes"] == 0
        assert res["num_human_votes"] == 3
        assert res["total_valid_votes"] == 3
        assert "human-created" in res["vote_summary"]
        assert pytest.approx(res["mean_ai_probability"], 0.01) == 0.1333

    def test_scenario_5_fewer_than_two_valid_predictions(self):
        """Scenario 5: Fewer than 2 valid model predictions -> UNABLE TO DETERMINE."""
        # 0 valid predictions
        res_zero = compute_majority_consensus(None, None, None)
        assert res_zero["status"] == "UNABLE_TO_DETERMINE"
        assert res_zero["label"] == "Unable to Determine"
        assert res_zero["total_valid_votes"] == 0

        # Only 1 valid prediction (AI)
        res_one_ai = compute_majority_consensus(0.95, None, None)
        assert res_one_ai["status"] == "UNABLE_TO_DETERMINE"
        assert res_one_ai["label"] == "Unable to Determine"
        assert res_one_ai["total_valid_votes"] == 1

        # Only 1 valid prediction (Human)
        res_one_human = compute_majority_consensus(None, 0.15, None)
        assert res_one_human["status"] == "UNABLE_TO_DETERMINE"
        assert res_one_human["label"] == "Unable to Determine"
        assert res_one_human["total_valid_votes"] == 1


class TestEdgeCasesAndRobustness:
    """Tests missing models, failed models, invalid probabilities, and tie handling."""

    def test_missing_model_two_agree_ai(self):
        """One model fails (None), but remaining 2 models both predict AI -> AI_GENERATED."""
        res = compute_majority_consensus(0.80, 0.90, None)
        assert res["status"] == "AI_GENERATED"
        assert res["label"] == "AI-Generated"
        assert res["total_valid_votes"] == 2
        assert res["num_ai_votes"] == 2
        assert res["num_human_votes"] == 0

    def test_missing_model_two_agree_human(self):
        """One model fails (None), but remaining 2 models both predict Human -> HUMAN_CREATED."""
        res = compute_majority_consensus(None, 0.10, 0.20)
        assert res["status"] == "HUMAN_CREATED"
        assert res["label"] == "Human-Created"
        assert res["total_valid_votes"] == 2
        assert res["num_ai_votes"] == 0
        assert res["num_human_votes"] == 2

    def test_tied_vote_two_models(self):
        """One model fails (None), remaining 2 models split 1 AI vs 1 Human -> UNABLE_TO_DETERMINE."""
        res = compute_majority_consensus(0.85, 0.15, None)
        assert res["status"] == "UNABLE_TO_DETERMINE"
        assert res["label"] == "Unable to Determine"
        assert res["total_valid_votes"] == 2
        assert res["num_ai_votes"] == 1
        assert res["num_human_votes"] == 1
        assert "tie" in res["vote_summary"].lower()

    def test_invalid_probabilities_filtering(self):
        """Tests that NaN, Inf, non-numbers, and out-of-bounds numbers are safely ignored."""
        assert is_valid_prob(None) is False
        assert is_valid_prob(float("nan")) is False
        assert is_valid_prob(float("inf")) is False
        assert is_valid_prob(float("-inf")) is False
        assert is_valid_prob(-0.01) is False
        assert is_valid_prob(1.05) is False
        assert is_valid_prob("corrupted_string") is False
        assert is_valid_prob(0.0) is True
        assert is_valid_prob(1.0) is True
        assert is_valid_prob(0.5) is True

        # Pass invalid types directly to consensus engine
        res = compute_majority_consensus(float("nan"), 0.85, 0.90)
        # 1 invalid + 2 valid AI -> 2 valid total, both AI -> AI_GENERATED
        assert res["total_valid_votes"] == 2
        assert res["status"] == "AI_GENERATED"
        assert res["num_ai_votes"] == 2

    def test_threshold_boundary_behavior(self):
        """Tests exact 0.50 boundary condition (>= 0.5 is AI, < 0.5 is Human)."""
        # Exactly 0.50 should count as AI
        res_exact = compute_majority_consensus(0.50, 0.50, 0.50)
        assert res_exact["status"] == "AI_GENERATED"
        assert res_exact["num_ai_votes"] == 3

        # 0.49999 should count as Human
        res_below = compute_majority_consensus(0.49999, 0.49999, 0.49999)
        assert res_below["status"] == "HUMAN_CREATED"
        assert res_below["num_human_votes"] == 3

    def test_diagnostic_telemetry_fields_present(self):
        """Verifies that mean probability and variance are preserved as diagnostics."""
        res = compute_majority_consensus(0.80, 0.60, 0.70)
        assert res["mean_ai_probability"] is not None
        assert res["variance"] is not None
        assert pytest.approx(res["mean_ai_probability"], 0.001) == 0.70
        assert res["threshold_used"] == 0.5


class TestBackendIntegration:
    """Verifies that backend.main._consensus returns the expected Pydantic schema."""

    def test_backend_consensus_schema(self):
        c_res = _consensus(c_prob=0.88, r_prob=0.72, ast_prob=0.15)
        assert isinstance(c_res, ConsensusResult)
        assert c_res.status == "LIKELY_AI_GENERATED"
        assert c_res.label == "Likely AI-Generated"
        assert c_res.num_ai_votes == 2
        assert c_res.num_human_votes == 1
        assert c_res.total_valid_votes == 3
        assert c_res.disagreement is True
        assert "2 of 3" in c_res.vote_summary
