"""Whole-episode optimal stopping objective and its deterministic evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_ID = "HV2"
CONTRACT = {
    "model_id": MODEL_ID, "fit_version": 1, "task": "whole_episode_net_r_stopping_policy",
    "sample": "first_executable_fill_per_original_signal_complete_natural_path",
    "input": "completed_adjusted_stock_market_ohlcv_300_plus_position_v1",
    "window": 300, "state_width": 6,
    "architecture": "conv16k7_conv32k5_mean_state_mlp32_stopping_v1",
    "objective": "mean_episode_expected_net_r_gain_over_natural_p16",
    "sampling": "all_states_whole_episodes_no_length_division",
    "selection": "hard_first_stop_mean_episode_gain_on_inner_validation",
    "threshold_probability": 0.5, "initial_stop_probability": 0.01,
    "max_epochs": 12, "early_stopping": "none_hard_policy_plateaus", "episode_batch_size": 16,
    "min_train_episodes": 64, "min_validation_episodes": 16,
    "learning_rate": 0.001, "gradient_clip": 5.0,
    "linear_baseline": "identical_flattened_300x10_plus_6_linear_sigmoid_same_objective",
    "missing": "known_unscorable_state_continue_original_rules",
    "execution": "next_session_open_persistent_request_original_protection_priority",
    "split": "whole_mature_episode_prior_calendar_year_validation",
}
DEVELOPMENT = {
    "seed": 42, "years": [2018, 2019, 2020], "truth_cutoff": "2020-12-31",
    "cohort": "entry_year_complete_episodes_mature_by_truth_cutoff",
    "prediction": "annual_pit_model_by_next_decision_session",
    "gate": "equal_year_mean_gain_gt0_at_least2_positive_years_and_mean_gain_vs_linear_gt0",
    "censoring": "exclude_and_report_whole_unmatured_episodes_no_forced_end_liquidation",
}


def build_policy_network(kind="dl"):
    import torch
    from filters.breakout_quality.holding_value import build_network
    if kind == "dl":
        model = build_network()
        final = model.head[-1]
    elif kind == "linear":
        class LinearPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(CONTRACT["window"] * 10 + CONTRACT["state_width"], 1)

            def forward(self, sequence, state):
                return self.head(torch.cat((sequence.flatten(1), state), dim=1)).squeeze(-1)
        model = LinearPolicy()
        final = model.head
    else:
        raise ValueError(f"Unknown stopping architecture: {kind}")
    torch.nn.init.zeros_(final.weight)
    torch.nn.init.constant_(final.bias, float(np.log(CONTRACT["initial_stop_probability"] /
                                                   (1 - CONTRACT["initial_stop_probability"]))))
    return model


def expected_episode_gain(logits, gain_if_stop, allowed, lengths):
    """Natural terminal payoff is subtracted once: terminal gain is exactly zero."""
    import torch
    if sum(lengths) != len(logits) or not lengths or min(lengths) < 1:
        raise ValueError("Invalid complete episode partition")
    probability = torch.where(allowed, torch.sigmoid(logits), torch.zeros_like(logits))
    results = []
    offset = 0
    for length in lengths:
        p = probability[offset:offset + length]
        survival = torch.cumprod(torch.cat((p.new_ones(1), 1 - p[:-1])), dim=0)
        results.append((survival * p * gain_if_stop[offset:offset + length]).sum())
        offset += length
    return torch.stack(results)


def episode_outcomes(rows, logits):
    """One first request, one fully settled canonical payoff per episode."""
    if len(rows) != len(logits) or not np.isfinite(logits).all():
        raise ValueError("Incomplete/nonfinite policy predictions")
    result = []
    for episode, positions in rows.groupby("episode", sort=False).indices.items():
        path = rows.iloc[positions]
        if not pd.to_datetime(path.information_date).is_monotonic_increasing:
            raise ValueError("Episode states are not chronological")
        first = path.iloc[0]
        threshold = np.log(CONTRACT["threshold_probability"] / (1 - CONTRACT["threshold_probability"]))
        selected = np.flatnonzero((np.asarray(logits)[positions] >= threshold) & path.decision_allowed.to_numpy(bool))
        chosen = path.iloc[int(selected[0])] if len(selected) else None
        natural_r = float(first.natural_net_pnl / first.initial_risk_money)
        gain = -float(chosen.target) if chosen is not None else 0.0
        exit_date = chosen.early_exit_date if chosen is not None else first.natural_exit_date
        result.append({"episode": episode, "entry_date": first.entry_date,
                       "natural_r": natural_r, "policy_r": natural_r + gain, "gain_r": gain,
                       "model_exit": chosen is not None,
                       "exit_information_date": None if chosen is None else chosen.information_date,
                       "exit_date": exit_date,
                       "holding_days": int((pd.Timestamp(exit_date) - pd.Timestamp(first.entry_date)).days)})
    return pd.DataFrame(result)


def outcome_metrics(outcomes):
    if outcomes.empty:
        raise ValueError("No complete episodes to evaluate")
    winners = outcomes.nlargest(max(1, int(np.ceil(len(outcomes) / 10))), "natural_r")
    return {"episodes": len(outcomes), "mean_gain_r": float(outcomes.gain_r.mean()),
            "mean_net_r": float(outcomes.policy_r.mean()), "natural_mean_net_r": float(outcomes.natural_r.mean()),
            "early_exit_fraction": float(outcomes.model_exit.mean()),
            "holding_days_median": float(outcomes.holding_days.median()),
            "top_decile_natural_r_sum": float(winners.natural_r.sum()),
            "top_decile_policy_r_sum": float(winners.policy_r.sum()),
            "top_decile_lost_positive_r": float(np.maximum(-winners.gain_r, 0).sum())}


def development_gate(years):
    if sorted(int(y) for y in years) != DEVELOPMENT["years"]:
        raise ValueError("Development evaluation requires all three prespecified years")
    gains = [years[str(y)]["dl"]["mean_gain_r"] for y in DEVELOPMENT["years"]]
    differences = [years[str(y)]["dl"]["mean_gain_r"] - years[str(y)]["linear"]["mean_gain_r"]
                   for y in DEVELOPMENT["years"]]
    if not np.isfinite(gains + differences).all():
        raise ValueError("Nonfinite development gate evidence")
    checks = {"mean_gain_positive": float(np.mean(gains)) > 0,
              "at_least_two_years_positive": sum(x > 0 for x in gains) >= 2,
              "mean_gain_over_linear_positive": float(np.mean(differences)) > 0}
    return {"passed": all(checks.values()), "checks": checks,
            "equal_year_mean_gain_r": float(np.mean(gains)),
            "positive_years": sum(x > 0 for x in gains),
            "equal_year_mean_gain_over_linear_r": float(np.mean(differences))}
