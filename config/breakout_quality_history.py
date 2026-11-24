"""Declared training-history extensions for canonical Research model sources."""

CO_PROFILE = "daily_universal_formal_mintrade_fullcycle_netpnl_per_seed_full_list_ndcg_pairwise"
HISTORY_PROFILE = CO_PROFILE + "_extended_history"

HISTORY_EXTENSION = {
    "first_oos_date": "2008-01-01", "last_oos_date": "2013-12-31",
    "namespace": "mintrade_early_label_history",
    "parameter_set": "MINTRADE_P16_EARLY_LABEL_HISTORY_V1",
}

TRAINING_HISTORY_EXTENSIONS = {HISTORY_PROFILE: HISTORY_EXTENSION}
