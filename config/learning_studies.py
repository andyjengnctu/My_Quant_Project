"""Authorized controlled learning studies; values are frozen before evaluation."""

CO_PROFILE = "daily_universal_formal_mintrade_fullcycle_netpnl_per_seed_full_list_ndcg_pairwise"
HISTORY_PROFILE = CO_PROFILE + "_extended_history"

HISTORY_EXTENSION = {
    "first_oos_date": "2008-01-01", "last_oos_date": "2013-12-31",
    "namespace": "mintrade_early_label_history",
    "parameter_set": "MINTRADE_P16_EARLY_LABEL_HISTORY_V1",
}

LEARNING_STUDIES = {
    "co_history": {
        "label": "真實突破歷史擴充 A/B", "kind": "history_extension",
        "model_id": "MR-13DN", "profile": HISTORY_PROFILE,
        "control_profile": CO_PROFILE, "architecture": "inception_time_v1",
        "arm_id": "C193", "control_arm": "C123", "reference_arm": "C106",
        "history_extension": HISTORY_EXTENSION,
        "screen_years": (2018, 2019, 2020), "screen_seed": 42,
        "screen_gate": {"minimum_positive_years": 2, "pair_delta_min": 0.0},
        "robustness_gate": {"minimum_wins": 3, "mean_mdd_delta_max": 0.0},
    },

}

STUDY_OUTPUT_ROOT = "outputs/research/learning_studies"
