"""Strategy Compare 歷史唯讀相容定義。

本模組保存退役 research arms、contrasts、DL sources 與 parameter sources，
只供舊工件解讀／重現。Current Extending matrix只存在
``config/strategy_compare.py``；歷史 Selection／Forward／Pre-Test 的 source、arm 與 contrast
全部留在本 compatibility catalog，只有歷史 profile 解析時可讀，除非先明確重新納入 active config。
"""

from __future__ import annotations

from config.execution_policy import DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT
from config.training_policy import (
    OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
    OPTIMIZER_RANDOM_SEED_DEFAULT,
    OUTER_ROLLING_OOS_HORIZON_MONTHS,
    OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
)

# 只供 SR-C26 歷史相容的固定語意。
STRATEGY_COMPARE_STALE_SCORE_MEMBERSHIP_GUARD_MAX_AGE_DAYS = 22

HISTORICAL_STRATEGY_PARAM_SOURCES = {
    "extending_min_roos": {
        "path_template": (
            "models/research/breakout_quality/strategy_compare/extending_min_roos/"
            "active_params/{param_filename}"
        ),
        "description": (
            "Extending-Window Min rolling schedule；stitch既有2014-2020 historical P2與2021+ "
            "current P2 rolling schedules；只在兩段rolling/search contract一致時建立，不重新最佳化。"
        ),
        "identity_manifest_path": (
            "models/research/breakout_quality/strategy_compare/extending_min_roos/"
            "extending_stitch_manifest.json"
        ),
        "trained_with_dl_id": None,
        "artifact_contract": {
            "breakout_quality_param_adaptation": {
                "mode": "min_roos_training",
                "parameter_set": "P2_EXTENDING",
                "search_fields": [
                    "high_len", "atr_len", "atr_buy_tol", "atr_times_init", "atr_times_trail"
                ],
                "fixed_rule_contract": "all_rule_filters_off",
                "training_dl_enabled": False,
            }
        },
        "builder": {
            "enabled": True,
            "builder_type": "extending_min_roos_stitch",
            "options": {
                "historical_params_path": (
                    "models/research/breakout_quality/trade_path_label/a2_teacher_params/"
                    "p2_dl_off_trained/active_params/{param_filename}"
                ),
                "current_params_path": (
                    "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
                    "risk_only_rolling/p2_dl_off_trained/active_params/{param_filename}"
                ),
                "output_relative_dir": (
                    "models/research/breakout_quality/strategy_compare/extending_min_roos"
                ),
                "quiet": False,
            },
        },
    },
    "extending_full_roos": {
        "path_template": (
            "models/research/breakout_quality/strategy_compare/extending_full_roos/"
            "active_params/{param_filename}"
        ),
        "description": (
            "Extending-Window Full rolling schedule；stitch既有2014-2020 historical P4與2021+ "
            "canonical Full rolling schedules；只在rolling/search contract一致時建立，不重新最佳化。"
        ),
        "identity_manifest_path": (
            "models/research/breakout_quality/strategy_compare/extending_full_roos/"
            "extending_stitch_manifest.json"
        ),
        "trained_with_dl_id": None,
        "artifact_contract": {
            "breakout_quality_param_adaptation": {
                "mode": "extending_full_roos_stitch",
                "parameter_set": "P4_EXTENDING",
                "training_dl_enabled": False,
            }
        },
        "builder": {
            "enabled": True,
            "builder_type": "extending_full_roos_stitch",
            "options": {
                "historical_params_path": (
                    "models/research/breakout_quality/strategy_compare/selection_full_roos/"
                    "active_params/{param_filename}"
                ),
                "current_params_path": "models/{param_filename}",
                "output_relative_dir": (
                    "models/research/breakout_quality/strategy_compare/extending_full_roos"
                ),
                "quiet": False,
            },
        },
    },
    "oos_min_roos": {
        "path_template": (
            "models/research/breakout_quality/strategy_compare/oos_min_roos/"
            "active_params/{param_filename}"
        ),
        "description": (
            "OOS Test fixed-cutoff Min參數；只取Extending Min schedule在2021-01-01當下合法的"
            "2020-cutoff參數，凍結使用至最新，不使用任何2021後重新fit的策略參數。"
        ),
        "identity_manifest_path": (
            "models/research/breakout_quality/strategy_compare/oos_min_roos/"
            "oos_freeze_manifest.json"
        ),
        "trained_with_dl_id": None,
        "artifact_contract": {
            "breakout_quality_param_adaptation": {
                "mode": "oos_param_freeze",
                "training_dl_enabled": False,
                "freeze_effective_date": "2021-01-01",
                "freeze_cutoff_date": "2020-12-31",
            }
        },
        "builder": {
            "enabled": True,
            "builder_type": "oos_param_freeze",
            "options": {
                "source_param_source_id": "extending_min_roos",
                "output_relative_dir": (
                    "models/research/breakout_quality/strategy_compare/oos_min_roos"
                ),
                "freeze_effective_date": "2021-01-01",
                "freeze_cutoff_date": "2020-12-31",
                "display_name": "Min OOS",
                "quiet": False,
            },
        },
    },
    "oos_full_roos": {
        "path_template": (
            "models/research/breakout_quality/strategy_compare/oos_full_roos/"
            "active_params/{param_filename}"
        ),
        "description": (
            "OOS Test fixed-cutoff Full參數；只取Extending Full schedule在2021-01-01當下合法的"
            "2020-cutoff參數，凍結使用至最新，不使用任何2021後重新fit的策略參數。"
        ),
        "identity_manifest_path": (
            "models/research/breakout_quality/strategy_compare/oos_full_roos/"
            "oos_freeze_manifest.json"
        ),
        "trained_with_dl_id": None,
        "artifact_contract": {
            "breakout_quality_param_adaptation": {
                "mode": "oos_param_freeze",
                "training_dl_enabled": False,
                "freeze_effective_date": "2021-01-01",
                "freeze_cutoff_date": "2020-12-31",
            }
        },
        "builder": {
            "enabled": True,
            "builder_type": "oos_param_freeze",
            "options": {
                "source_param_source_id": "extending_full_roos",
                "output_relative_dir": (
                    "models/research/breakout_quality/strategy_compare/oos_full_roos"
                ),
                "freeze_effective_date": "2021-01-01",
                "freeze_cutoff_date": "2020-12-31",
                "display_name": "Full OOS",
                "quiet": False,
            },
        },
    },
    "min_dl_tp1_roos": {
        "path_template": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p3_dl_on_trained/active_params/{param_filename}"
        ),
        "description": "rule-based filters全關、TP1-on環境訓練的Min-TP1 ROOS",
        "identity_manifest_path": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p3_dl_on_trained/rolling_preflight.json"
        ),
        "trained_with_dl_id": "TP1",
        "builder": {
            "enabled": True,
            "builder_type": "binary_dl_min_roos_rolling",
            "options": {
                "parameter_set": "p3",
                "model_source_id": "TP1",
                "p3_variant": None,
                "trials_per_fold": OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
                "resume": True,
                "fixed_risk": DEFAULT_FIXED_RISK,
                "max_position_cap_pct": DEFAULT_MAX_POSITION_CAP_PCT,
                "build_binary_pit": True,
                "binary_pit_resume": True,
                "quiet": False,
            },
        },
    },
    "min_dl_a9_roos": {
        "path_template": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p3_dl_on_trained/A9/active_params/{param_filename}"
        ),
        "description": "rule-based filters全關、A9-on環境訓練的Min-A9 ROOS",
        "identity_manifest_path": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p3_dl_on_trained/A9/rolling_preflight.json"
        ),
        "trained_with_dl_id": "A9",
        "builder": {
            "enabled": True,
            "builder_type": "binary_dl_min_roos_rolling",
            "options": {
                "parameter_set": "p3",
                "model_source_id": "A9",
                "p3_variant": "A9",
                "trials_per_fold": OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
                "resume": True,
                "fixed_risk": DEFAULT_FIXED_RISK,
                "max_position_cap_pct": DEFAULT_MAX_POSITION_CAP_PCT,
                "build_binary_pit": True,
                "binary_pit_resume": True,
                "quiet": False,
            },
        },
    },

    # Historical-only definitions migrated out of current Strategy Compare catalog.
    "full_roos": {
        "path_template": None,
        "description": "正式Full ROOS；依param_policy解析canonical rolling工件",
        "identity_manifest_path": None,
        "trained_with_dl_id": None,
        "builder": None,
    },
    "min_roos": {
        "path_template": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p2_dl_off_trained/active_params/{param_filename}"
        ),
        "description": "forward rolling期間、rule-based filters全關、DL-off訓練的Min ROOS",
        "identity_manifest_path": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p2_dl_off_trained/rolling_preflight.json"
        ),
        "trained_with_dl_id": None,
        "builder": {
            "enabled": True,
            "builder_type": "binary_dl_min_roos_rolling",
            "options": {
                "parameter_set": "p2",
                "model_source_id": "TP1",
                "trials_per_fold": OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
                "resume": True,
                "fixed_risk": DEFAULT_FIXED_RISK,
                "max_position_cap_pct": DEFAULT_MAX_POSITION_CAP_PCT,
                "build_binary_pit": False,
                "binary_pit_resume": True,
                "quiet": False,
            },
        },
    },
    "selection_min_roos": {
        "path_template": (
            "models/research/breakout_quality/trade_path_label/a2_teacher_params/"
            "p2_dl_off_trained/active_params/{param_filename}"
        ),
        "description": (
            "Selection 2014～2020 historical Min ROOS；rules全關、DL-off，"
            "單階段rolling直接搜尋high_len＋4個ATR欄位"
        ),
        "identity_manifest_path": None,
        "trained_with_dl_id": None,
        "artifact_contract": {
            "breakout_quality_param_adaptation": {
                "mode": "min_roos_training",
                "parameter_set": "P2_HISTORY",
                "search_fields": [
                    "high_len",
                    "atr_len",
                    "atr_buy_tol",
                    "atr_times_init",
                    "atr_times_trail",
                ],
                "fixed_rule_contract": "all_rule_filters_off",
                "training_dl_enabled": False,
            }
        },
        "builder": {
            "enabled": True,
            "builder_type": "selection_historical_p2",
            "options": {
                "parameter_set": "p2_history",
                "trials_per_fold": OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
                "train_window_months": OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
                "oos_months": OUTER_ROLLING_OOS_HORIZON_MONTHS,
                "resume": True,
                "fixed_risk": DEFAULT_FIXED_RISK,
                "max_position_cap_pct": DEFAULT_MAX_POSITION_CAP_PCT,
                "optimizer_seed": OPTIMIZER_RANDOM_SEED_DEFAULT,
                "quiet": False,
            },
        },
    },
    "selection_full_roos": {
        "path_template": (
            "models/research/breakout_quality/strategy_compare/selection_full_roos/"
            "active_params/{param_filename}"
        ),
        "description": (
            "Selection 2014～2020 historical Full ROOS；使用canonical Full optimizer search space，"
            "DL filter/ranking與TP維持optimizer正式固定契約"
        ),
        "identity_manifest_path": (
            "models/research/breakout_quality/strategy_compare/selection_full_roos/"
            "rolling_preflight.json"
        ),
        "trained_with_dl_id": None,
        "artifact_contract": {
            "breakout_quality_param_adaptation": {
                "mode": "selection_full_roos_training",
                "parameter_set": "P4_HISTORY",
                "training_dl_enabled": False,
            }
        },
        "builder": {
            "enabled": True,
            "builder_type": "selection_historical_full_roos",
            "options": {
                "parameter_set": "p4_history",
                "trials_per_fold": OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
                "train_window_months": OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
                "oos_months": OUTER_ROLLING_OOS_HORIZON_MONTHS,
                "resume": True,
                "fixed_risk": DEFAULT_FIXED_RISK,
                "max_position_cap_pct": DEFAULT_MAX_POSITION_CAP_PCT,
                "optimizer_seed": OPTIMIZER_RANDOM_SEED_DEFAULT,
                "quiet": False,
            },
        },
    },
}

HISTORICAL_STRATEGY_DL_SOURCES = {
    "TP1": {
        "filter_id": "breakout_quality_a2_trade_path_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "unique_group_sampling",
        "threshold": 0.5,
        "score_source": "canonical_runtime",
        "description": "A2 realized trade-path Binary DL模型",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "forward_oos_scores",
            "options": {
                "scope": "forward_oos",
                "inference_batch_size": 4096,
                "inference_workers": 4,
                "device": "auto",
                "mixed_precision": True,
                "mixed_precision_dtype": "bfloat16",
                "deterministic_algorithms": True,
                "allow_tf32": False,
                "preload_feature_bank": True,
            },
        },
    },
    "A9": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "unique_group_sampling",
        "threshold": 0.5,
        "score_source": "canonical_runtime",
        "description": "既有MFE／MAE 9A Binary DL模型",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "forward_oos_scores",
            "options": {
                "scope": "forward_oos",
                "inference_batch_size": 4096,
                "inference_workers": 4,
                "device": "auto",
                "mixed_precision": True,
                "mixed_precision_dtype": "bfloat16",
                "deterministic_algorithms": True,
                "allow_tf32": False,
                "preload_feature_bank": True,
            },
        },
    },
    "CONT11G": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_pass_magnitude_mse",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": "MR-11G frozen OOS continuous ranker；只允許受控strategy research replay",
        "forward_scores_builder": None,
    },
    "CONT12A": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_all_event_mse",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": "MR-12A all-event no-time frozen OOS continuous ranker；只允許受控strategy research replay",
        "forward_scores_builder": None,
    },
    "CONT12C": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_all_event_listwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": "MR-12C all-event no-time ListNet top-one listwise ranker frozen OOS score；只允許受控strategy research replay",
        "forward_scores_builder": None,
    },

    # Historical-only definitions migrated out of current Strategy Compare catalog.
    "CONT13E": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_no_time_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": (
            "MR-13E Daily Universal full-list Delta-NDCG frozen Forward-OOS score；"
            "盤前使用最新已完成交易日資訊，供current MR-13E exact strategy research"
        ),
        "forward_scores_builder": None,
    },
    "CONT13E_PIT": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_no_time_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": (
            "MR-13E Daily Universal full-list Delta-NDCG Selection PIT score；"
            "盤前只使用最新已完成交易日資訊，供2014～2020 current MR-13E exact strategy research"
        ),
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {
                "resume": True,
                "allow_stale_source": False,
            },
        },
    },
    "CONT13K": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": (
            "MR-13K Daily Universal full-horizon pure-MFE frozen Forward-OOS score；"
            "盤前使用最新已完成交易日資訊，供C54與C44作同K/R0/exact source-only策略比較"
        ),
        "forward_scores_builder": None,
    },
    "CONT13M": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_low_adverse_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": (
            "MR-13M Daily Universal full-horizon low-adverse frozen Forward-OOS score；"
            "只作C55/C56 secondary path-safety source；C55使用raw safety floor，C56使用同日rank-space residual safety floor；皆不與MR-13K score加權、不單獨作portfolio objective"
        ),
        "forward_scores_builder": None,
    },
    "CONT13M_PIT": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_low_adverse_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": (
            "MR-13M Daily Universal full-horizon low-adverse Selection PIT score；"
            "只供C57作C56 B2的Selection secondary residual-safety source，不單獨作portfolio objective"
        ),
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {
                "resume": True,
                "allow_stale_source": False,
            },
        },
    },
    "CONT13K_PIT": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": (
            "MR-13K Daily Universal full-horizon pure-MFE Selection PIT score；"
            "盤前只使用最新已完成交易日資訊，供C53與C42作同K/R0/exact source-only策略比較"
        ),
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {
                "resume": True,
                "allow_stale_source": False,
            },
        },
    },
}

HISTORICAL_STRATEGY_COMPARE_ARMS = {
    "C2": {
        "name": "Full ROOS: TP1-on",
        "description": "Full ROOS參數，runtime開TP1",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "TP1",
        "dl_runtime_mode": "hard-filter",
    },
    "C4": {
        "name": "Min ROOS: TP1-on",
        "description": "Min ROOS參數，runtime開TP1",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "TP1",
        "dl_runtime_mode": "hard-filter",
    },
    "C5": {
        "name": "Min-TP1 ROOS",
        "description": "TP1-on環境訓練參數，runtime DL-off",
        "param_source": "min_dl_tp1_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
    },
    "C6": {
        "name": "Min-TP1 ROOS: DL-on",
        "description": "TP1-on環境訓練參數，runtime開其配對TP1",
        "param_source": "min_dl_tp1_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "TP1",
        "dl_runtime_mode": "hard-filter",
    },
    "C7": {
        "name": "Full ROOS: A9-on",
        "description": "Full ROOS參數，runtime開A9",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "hard-filter",
    },
    "C8": {
        "name": "Min ROOS: A9-on",
        "description": "Min ROOS參數，runtime開A9",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "hard-filter",
    },
    "C9": {
        "name": "Min-A9 ROOS",
        "description": "A9-on環境訓練參數，runtime DL-off",
        "param_source": "min_dl_a9_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
    },
    "C10": {
        "name": "Min-A9 ROOS: DL-on",
        "description": "A9-on環境訓練參數，runtime開其配對A9",
        "param_source": "min_dl_a9_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "hard-filter",
    },
    "C11": {
        "name": "Min ROOS: A9 resource-aware",
        "description": "Min ROOS參數；A9只在盤前cash先成瓶頸時以first-improvement改善PASS預留資金",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "resource-aware-binary",
    },
    "C12": {
        "name": "Min ROOS: A9 resource-aware basket",
        "description": "Min ROOS參數；沿用相同cash-binding Gate，每輪評估全部可行PASS promotion並採用最佳改善",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "resource-aware-binary-basket",
    },
    "C14": {
        "name": "Min ROOS: Continuous resource-aware",
        "description": "歷史對照；capital-utilization first + MR-11G PASS-only continuous score",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT11G",
        "dl_runtime_mode": "resource-aware-continuous",
    },
    "C15": {
        "name": "Min ROOS: All-event Continuous resource-aware",
        "description": "Min ROOS先維持資本利用；只有cash-binding的DL Selection Mode才使用MR-12A all-event frozen OOS continuous score排序",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12A",
        "dl_runtime_mode": "resource-aware-continuous",
    },
    "C16": {
        "name": "Min ROOS: All-event Continuous capital-preserving",
        "description": (
            "Min ROOS exact reservation建立每日baseline；MR-12A frozen continuous score可在"
            "cash/slot瓶頸日重排，但selected count與reserved capital均不得低於baseline"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12A",
        "dl_runtime_mode": "resource-aware-continuous-capital-preserving",
    },
    "C17": {
        "name": "Min ROOS: All-event Continuous max-DL constrained basket",
        "description": (
            "Min ROOS只固定每日預留單數K與exact reserved-capital floor；"
            "MR-12A frozen score先取純DL Top-K，不合法時只做deterministic minimum-repair，"
            "最後只允許K筆盤前預留單"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12A",
        "dl_runtime_mode": "resource-aware-continuous-max-dl",
    },
    "C18": {
        "name": "Min ROOS: All-event Continuous max-DL feasible-ascent",
        "description": (
            "與C17使用完全相同K/R0、MR-12A與basket內Min ROOS執行順序；"
            "C17合法seed之後持續做best-feasible single-swap DL改善直到1-swap local optimum，"
            "capital只作hard feasibility，不參與objective"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12A",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
    },
    "C19": {
        "name": "Min ROOS: MR-12B pairwise max-DL constrained basket",
        "description": (
            "與C17使用完全相同K/R0、minimum-repair與basket內Min ROOS執行順序；"
            "唯一模型差異為DL source改成MR-12B pairwise ranker"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12B",
        "dl_runtime_mode": "resource-aware-continuous-max-dl",
    },
    "C21": {
        "name": "Min ROOS: MR-12C listwise max-DL constrained basket",
        "description": (
            "與C19使用完全相同C17 K/R0、minimum-repair與basket內Min ROOS執行順序；"
            "唯一模型差異為DL source改成MR-12C ListNet top-one listwise ranker"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12C",
        "dl_runtime_mode": "resource-aware-continuous-max-dl",
    },
    "C22": {
        "name": "Min ROOS: MR-12C listwise max-DL feasible-ascent",
        "description": (
            "與C20使用完全相同C18 K/R0、feasible-ascent與basket內Min ROOS執行順序；"
            "唯一模型差異為DL source改成MR-12C ListNet top-one listwise ranker"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12C",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
    },
    "C24": {
        "name": "Selection PIT: MR-12B minimum-repair",
        "description": (
            "與C23使用完全相同historical P2 Min ROOS params；"
            "使用MR-12B Selection PIT score並完全沿用C17 minimum-repair selector"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12B_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl",
    },
    "C26": {
        "name": "Selection PIT: MR-12B feasible-ascent stale-score guard",
        "description": (
            "與C25完全相同MR-12B Selection PIT與feasible-ascent；"
            f"唯一變更為score age超過Selection預先凍結{STRATEGY_COMPARE_STALE_SCORE_MEMBERSHIP_GUARD_MAX_AGE_DAYS}日門檻時，"
            "該舊score不得驅動DL membership swap，候選本身仍保留並沿用Min ROOS資源契約"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12B_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard",
        "dl_runtime_options": {
            "stale_score_membership_guard_max_age_days": STRATEGY_COMPARE_STALE_SCORE_MEMBERSHIP_GUARD_MAX_AGE_DAYS,
        },
    },
    "C27": {
        "name": "Selection PIT: MR-13A daily minimum-repair",
        "description": (
            "與C24使用完全相同historical P2 Min ROOS params、K/R0與minimum-repair selector；"
            "唯一DL差異為score source改成MR-13A Daily Universal Selection PIT，"
            "盤前每日依最新已完成交易日score重排"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13A_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl",
    },
    "C30": {
        "name": "Full ROOS: MR-12B feasible-ascent",
        "description": (
            "與C1使用相同Full ROOS active params與formal rules；"
            "feasible-ascent的K/R0由同參數DL-off baseline逐日建立，DL source為MR-12B"
        ),
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "CONT12B",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
    },
    "C31": {
        "name": "Full ROOS: MR-13A daily feasible-ascent",
        "description": (
            "與C30使用相同Full ROOS active params、formal rules及frozen feasible-ascent；"
            "唯一DL source差異為MR-13A Daily Universal Forward-OOS score"
        ),
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "CONT13A",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
    },
    "C33": {
        "name": "Selection Full ROOS: MR-12B feasible-ascent",
        "description": (
            "與C32使用相同historical Full ROOS與formal rules；K/R0由同參數DL-off baseline建立；"
            "使用MR-12B Selection PIT與frozen feasible-ascent"
        ),
        "param_source": "selection_full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "CONT12B_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
    },
    "C34": {
        "name": "Selection Full ROOS: MR-13A daily feasible-ascent",
        "description": (
            "與C33使用相同historical Full ROOS、formal rules與frozen feasible-ascent；"
            "唯一DL source差異為MR-13A Daily Universal Selection PIT"
        ),
        "param_source": "selection_full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "CONT13A_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
    },
    "C37": {
        "name": "Min MR-13E Expected-PnL",
        "description": (
            "MR-13E權重/PIT score frozen；以Selection expanding/PIT daily percentile校準Expected R，"
            "在與C35完全相同K/R0、sizing、cash、execution下，basket objective唯一改為"
            "Σ(Expected R × canonical planned initial risk)"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E_PIT",
        "dl_runtime_mode": "resource-aware-continuous-expected-pnl-feasible-ascent",
        "dl_runtime_options": {
            "expected_r_fit_dl_id": "CONT13E_PIT",
            "expected_r_calibration_method": "daily_score_percentile_nonnegative_affine_v1",
            "negative_expected_r_allowed": True,
            "preserve_k_r0": True,
        },
        "robustness_role": "off",
    },
    "C46": {
        "name": "Min MR-13E Score Exact No-R0",
        "description": (
            "SR-C46 historical rejected R0 ablation：historical Min params/all-off + frozen MR-13E PIT score；"
            "固定K與canonical sizing/cash/orderability/execution，移除baseline R0，exact maximize ΣScore"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E_PIT",
        "dl_runtime_mode": "resource-aware-continuous-score-no-r0-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k": True,
            "preserve_r0": False,
            "r0_minimum_repair": False,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C48": {
        "name": "Min MR-13E Pareto Exact No-R0",
        "description": (
            "SR-C48 historical rejected basket-level Pareto No-R0：與C42同historical Min params/all-off、"
            "frozen MR-13E PIT score、K、canonical 1% risk sizing/cash/orderability/execution；移除R0。"
            "exact三pass先求最大Score coverage下的Score/Capital端點，再以端點min-max normalization"
            "全域最大化Q_norm×C_norm；Selection結果明顯輸C42，因此不建Forward"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E_PIT",
        "dl_runtime_mode": "resource-aware-continuous-score-capital-pareto-no-r0-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k": True,
            "preserve_r0": False,
            "r0_minimum_repair": False,
            "constrained_solver": "exact_branch_and_bound_v1",
            "pareto_selection": "normalized_product_v1",
            "pareto_quality": "score_sum_max_coverage_first",
            "pareto_capital": "canonical_reserved_cost_milli",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C47": {
        "name": "Min MR-13E Score×Capital Exact No-R0",
        "description": (
            "SR-C47 historical rejected objective：與C46同K/No-R0/canonical constraints與frozen MR-13E PIT score；"
            "exact maximize Σ(score_i × canonical reserved_cost_i)，不做normalization、shift或lambda"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E_PIT",
        "dl_runtime_mode": "resource-aware-continuous-score-capital-no-r0-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k": True,
            "preserve_r0": False,
            "r0_minimum_repair": False,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C45": {
        "name": "Full MR-13E Constrained",
        "description": (
            "SR-C45 historical-only rejected Full ROOS transfer：重用MR-13E Selection PIT score、"
            "原始ΣMR-13E score objective與exact_branch_and_bound_v1；策略體系為"
            "selection_full_roos + formal，K/R0、sizing、cash、orderability與execution由同日C32 semantics建立"
        ),
        "param_source": "selection_full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "CONT13E_PIT",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C38": {
        "name": "Min MR-13E Expected-PnL",
        "description": (
            "MR-13E Forward score frozen；Expected-R mapping只用2021-01-01前成熟Selection PIT target fit，"
            "與C36完全相同K/R0、sizing、cash、execution，basket objective唯一改為"
            "Σ(Expected R × canonical planned initial risk)；不得讀Forward target fit calibration"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E",
        "dl_runtime_mode": "resource-aware-continuous-expected-pnl-feasible-ascent",
        "dl_runtime_options": {
            "expected_r_fit_dl_id": "CONT13E_PIT",
            "expected_r_calibration_method": "daily_score_percentile_nonnegative_affine_v1",
            "negative_expected_r_allowed": True,
            "preserve_k_r0": True,
        },
        "robustness_role": "off",
    },

    # Historical-only definitions migrated out of current Strategy Compare catalog.
    "C1": {
        "name": 'Full ROOS',
        "description": "Full optimizer rolling active params；DL-off baseline",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
    "C3": {
        "name": 'Min ROOS',
        "description": "只搜尋high_len＋4個ATR；rules全關；DL-off baseline",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
    "C23": {
        "name": 'Min ROOS',
        "description": (
            "2014～2020 historical P2 Min ROOS active params；rules全關；DL關閉；"
            "作Selection PIT策略經濟驗證共同baseline"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
    "C42": {
        "name": 'Min MR-13E Constrained',
        "description": (
            "Selection PIT current research arm：historical Min params/all-off + frozen MR-13E PIT score；"
            "固定K/R0、canonical sizing/cash/orderability/execution，以deterministic exact branch-and-bound"
            "在完整候選universe求K/R0/canonical-cash feasible MR-13E score-sum global optimum"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E_PIT",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C44": {
        "name": 'Min MR-13E Constrained',
        "description": (
            "Forward-OOS current research arm：current Min params/all-off + frozen MR-13E Forward score；"
            "固定K/R0、canonical sizing/cash/orderability/execution，以與C42同源deterministic exact "
            "branch-and-bound在完整候選universe求K/R0/canonical-cash feasible MR-13E score-sum global optimum"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": False,
        },
        "robustness_role": "off",
    },
    "C56": {
        "name": 'Min MR-13K + MR-13M Residual Safety',
        "description": (
            "Forward-only Plan B2 controlled arm：完全沿用C54的Min params/all-off、K/R0、canonical "
            "sizing/cash/orderability/execution與MR-13K primary score objective；每天在當日orderable候選中，"
            "先把MR-13K/MR-13M frozen scores各自轉average-rank percentile，再以含intercept OLS估計"
            "expected safety rank given upside rank，Residual Safety=actual safety percentile−expected safety percentile。"
            "選中basket的Residual Safety coverage與score-sum不得低於同日DL-off Min ROOS baseline；13M residual"
            "只作hard floor，不與13K objective加權、不使用未來Target、不新增numeric threshold。"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13K",
        "dl_runtime_mode": "resource-aware-continuous-score-residual-safety-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": False,
            "safety_dl_id": "CONT13M",
            "safety_constraint": "baseline_residual_coverage_and_score_sum_floor_v1",
            "safety_residualization": "same_day_rank_ols_v1",
        },
        "robustness_role": "off",
    },
    "C57": {
        "name": 'Min MR-13K + MR-13M Residual Safety',
        "description": (
            "Selection PIT C56 full-flow counterpart：historical Min params/all-off、K/R0、canonical "
            "sizing/cash/orderability/execution與exact solver固定；primary改用CONT13K_PIT，secondary改用"
            "CONT13M_PIT。每天只在當日orderable候選把兩個PIT-safe score轉rank percentile並做同日OLS residual；"
            "Residual Safety只作baseline-relative hard floor，primary objective仍只最大化MR-13K score-sum。"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13K_PIT",
        "dl_runtime_mode": "resource-aware-continuous-score-residual-safety-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
            "safety_dl_id": "CONT13M_PIT",
            "safety_constraint": "baseline_residual_coverage_and_score_sum_floor_v1",
            "safety_residualization": "same_day_rank_ols_v1",
        },
        "robustness_role": "off",
    },
    "C32": {
        "name": 'Full ROOS',
        "description": "2014～2020 historical Full ROOS active params；formal rules；DL-off共同baseline",
        "param_source": "selection_full_roos",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
}

HISTORICAL_STRATEGY_COMPARE_CONTRASTS = {
    "C8-C3": {"left": "C8", "right": "C3", "description": "Min ROOS下A9 hard-filter效果（既有對照重現）"},
    "C11-C3": {"left": "C11", "right": "C3", "description": "Min ROOS下A9 resource-aware first-improvement效果"},
    "C12-C3": {"left": "C12", "right": "C3", "description": "Min ROOS下A9 resource-aware best-improvement效果"},
    "C14-C3": {"left": "C14", "right": "C3", "description": "Capital-utilization first下MR-11G continuous排序效果"},
    "C14-C12": {"left": "C14", "right": "C12", "description": "MR-11G Continuous resource-aware相對A9 max-PASS resource-aware效果"},
    "C15-C3": {"left": "C15", "right": "C3", "description": "Capital-utilization first下MR-12A all-event continuous排序效果"},
    "C15-C12": {"left": "C15", "right": "C12", "description": "All-event continuous resource-aware相對A9 max-PASS resource-aware效果"},
    "C16-C15": {"left": "C16", "right": "C15", "description": "同一MR-12A source下capital-preserving selector相對C15 cash-binding selector的純runtime效果"},
    "C16-C3": {"left": "C16", "right": "C3", "description": "Capital-preserving MR-12A selector相對Min ROOS正式研究基準"},
    "C17-C16": {"left": "C17", "right": "C16", "description": "同一MR-12A source下max-DL constrained basket相對C16 capital-preserving heuristic的純selector效果"},
    "C17-C3": {"left": "C17", "right": "C3", "description": "Max-DL constrained basket在固定Min ROOS資源底線下相對正式研究基準"},
    "C18-C17": {"left": "C18", "right": "C17", "description": "相同MR-12A與K/R0資源契約下，feasible-ascent相對C17 minimum-repair的純selector搜尋效果"},
    "C18-C3": {"left": "C18", "right": "C3", "description": "Max-DL feasible-ascent在固定Min ROOS資源底線下相對正式研究基準"},
    "C21-C19": {"left": "C21", "right": "C19", "description": "固定C17 selector下MR-12C listwise相對MR-12B pairwise的純DL模型效果"},
    "C22-C20": {"left": "C22", "right": "C20", "description": "固定C18 selector下MR-12C listwise相對MR-12B pairwise的純DL模型效果"},
    "C22-C21": {"left": "C22", "right": "C21", "description": "同一MR-12C source下C18 feasible-ascent相對C17 minimum-repair的selector轉化效果"},
    "C21-C3": {"left": "C21", "right": "C3", "description": "MR-12C在C17 selector下相對Min ROOS研究基準"},
    "C22-C3": {"left": "C22", "right": "C3", "description": "MR-12C在C18 selector下相對Min ROOS研究基準"},
    "C19-C17": {"left": "C19", "right": "C17", "description": "固定C17 selector下MR-12B pairwise相對MR-12A MSE的純DL模型效果"},
    "C20-C18": {"left": "C20", "right": "C18", "description": "固定C18 selector下MR-12B pairwise相對MR-12A MSE的純DL模型效果"},
    "C20-C19": {"left": "C20", "right": "C19", "description": "同一MR-12B source下C18 feasible-ascent相對C17 minimum-repair的selector轉化效果"},
    "C19-C3": {"left": "C19", "right": "C3", "description": "MR-12B在C17 selector下相對Min ROOS研究基準"},
    "C24-C23": {"left": "C24", "right": "C23", "description": "Selection PIT下固定historical Min ROOS與C17 selector，MR-12B PIT ranking相對DL-off baseline的經濟效果"},
    "C26-C25": {"left": "C26", "right": "C25", "description": "Selection PIT MR-12B feasible-ascent固定其餘條件下，stale-score membership guard的純runtime效果"},
    "C26-C23": {"left": "C26", "right": "C23", "description": "Selection PIT下固定historical Min ROOS，MR-12B feasible-ascent加stale-score membership guard相對DL-off baseline的經濟效果"},
    "C25-C24": {"left": "C25", "right": "C24", "description": "Selection PIT MR-12B固定score source下，C18 feasible-ascent相對C17 minimum-repair的selector轉化效果"},
    "C27-C24": {"left": "C27", "right": "C24", "description": "固定historical Min ROOS與minimum-repair selector，MR-13A daily PIT相對MR-12B event PIT的純DL source效果"},
    "C27-C23": {"left": "C27", "right": "C23", "description": "Selection PIT下MR-13A daily minimum-repair相對DL-off historical Min ROOS baseline的策略經濟效果"},
    "C28-C27": {"left": "C28", "right": "C27", "description": "同一MR-13A daily PIT source下，feasible-ascent相對minimum-repair的selector轉化效果"},
    "C30-C1": {"left": "C30", "right": "C1", "description": "Full ROOS下MR-12B feasible-ascent相對DL-off baseline的Forward-OOS策略效果"},
    "C31-C1": {"left": "C31", "right": "C1", "description": "Full ROOS下MR-13A daily feasible-ascent相對DL-off baseline的Forward-OOS策略效果"},
    "C31-C30": {"left": "C31", "right": "C30", "description": "固定Full ROOS與feasible-ascent，MR-13A daily相對MR-12B event source的純DL Forward-OOS效果"},
    "C31-C29": {"left": "C31", "right": "C29", "description": "固定MR-13A daily與feasible-ascent下，Full ROOS相對Min ROOS的完整策略體系interaction"},
    "C12-C11": {"left": "C12", "right": "C11", "description": "Best-improvement相對first-improvement改善"},
    "C11-C8": {"left": "C11", "right": "C8", "description": "Resource-aware相對A9 hard-filter改善"},
    "C2-C1": {"left": "C2", "right": "C1", "description": "Full ROOS下TP1 runtime效果"},
    "C7-C1": {"left": "C7", "right": "C1", "description": "Full ROOS下A9 runtime效果"},
    "C4-C3": {"left": "C4", "right": "C3", "description": "Min ROOS下TP1 runtime效果"},
    "C6-C5": {"left": "C6", "right": "C5", "description": "Min-TP1 ROOS下配對DL效果"},
    "C10-C9": {"left": "C10", "right": "C9", "description": "Min-A9 ROOS下配對DL效果"},
    "C3-C1": {"left": "C3", "right": "C1", "description": "Min ROOS相對Full ROOS"},
    "C5-C3": {"left": "C5", "right": "C3", "description": "TP1-aware參數本身效果"},
    "C9-C3": {"left": "C9", "right": "C3", "description": "A9-aware參數本身效果"},
    "C6-C4": {"left": "C6", "right": "C4", "description": "TP1-on下參數適應效果"},
    "C10-C8": {"left": "C10", "right": "C8", "description": "A9-on下參數適應效果"},
    "C6-C1": {"left": "C6", "right": "C1", "description": "Min-TP1完整方案相對正式基準"},
    "C10-C1": {"left": "C10", "right": "C1", "description": "Min-A9完整方案相對正式基準"},
    "C33-C32": {"left": "C33", "right": "C32", "description": "Selection Full ROOS下MR-12B feasible-ascent相對DL-off baseline的策略效果"},
    "C34-C32": {"left": "C34", "right": "C32", "description": "Selection Full ROOS下MR-13A daily feasible-ascent相對DL-off baseline的策略效果"},
    "C34-C33": {"left": "C34", "right": "C33", "description": "固定Selection Full ROOS與feasible-ascent，MR-13A daily相對MR-12B event PIT的純DL source效果"},
    "C34-C28": {"left": "C34", "right": "C28", "description": "固定MR-13A daily PIT與feasible-ascent下，Selection Full相對Min的完整策略體系interaction"},
    "C37-C23": {"left": "C37", "right": "C23", "description": "Selection PIT下frozen MR-13E Expected-PnL相對DL-off Min ROOS的策略經濟效果"},
    "C37-C35": {"left": "C37", "right": "C35", "description": "同一MR-13E PIT source與同K/R0；只比較Expected-Dollar-PnL objective相對score-sum objective"},
    "C37-C25": {"left": "C37", "right": "C25", "description": "Selection PIT frozen MR-13E Expected-PnL相對MR-12B runtime anchor"},
    "C38-C3": {"left": "C38", "right": "C3", "description": "Forward-OOS frozen MR-13E Expected-PnL相對DL-off Min ROOS的策略經濟效果"},
    "C38-C36": {"left": "C38", "right": "C36", "description": "同一frozen MR-13E Forward source與同K/R0；只比較Expected-Dollar-PnL objective相對score-sum objective"},
    "C38-C20": {"left": "C38", "right": "C20", "description": "Forward-OOS frozen MR-13E Expected-PnL相對MR-12B runtime anchor"},
    "C46-C42": {"left": "C46", "right": "C42", "description": "SR-C46 historical純R0 ablation：同MR-13E score/K/exact/cash下移除baseline R0 floor"},
    "C47-C46": {"left": "C47", "right": "C46", "description": "SR-C47 historical同No-R0 exact下由ΣScore改為Σ(Score×canonical reserved capital)"},
    "C47-C42": {"left": "C47", "right": "C42", "description": "SR-C47 historical Score×Capital No-R0相對C42 R0-constrained control"},
    "C48-C42": {"left": "C48", "right": "C42", "description": "SR-C48 historical basket-level Pareto Exact No-R0相對C42 R0-constrained control"},
    "C45-C32": {"left": "C45", "right": "C32", "description": "SR-C45 historical Full ROOS/formal MR-13E exact constrained相對C32 Full ROOS DL-off baseline"},
    "C45-C42": {"left": "C45", "right": "C42", "description": "SR-C45 historical同MR-13E PIT score/exact solver下Full/formal相對Min/all-off整體策略體系差異"},

    # Historical-only definitions migrated out of current Strategy Compare catalog.
    "C42-C23": {"left": "C42", "right": "C23", "description": "Selection PIT frozen MR-13E score exact constrained optimum相對DL-off Min ROOS的策略經濟效果"},
    "C42-C32": {"left": "C42", "right": "C32", "description": "Selection PIT active research最終候選：Min MR-13E exact constrained相對Full ROOS的整體策略結果；不是單一參數或單一DL效果"},
    "C57-C42": {"left": "C57", "right": "C42", "description": "C56 full-flow Selection primary contrast：同historical Min/K/R0/exact/cash/execution下，以MR-13K PIT primary + MR-13M PIT residual safety對production MR-13E exact reference"},
    "C57-C23": {"left": "C57", "right": "C23", "description": "C56 full-flow Selection相對DL-off Min ROOS的策略經濟效果"},
    "C57-C32": {"left": "C57", "right": "C32", "description": "C56 full-flow Selection相對Full ROOS的整體策略結果；不是單一參數效果"},
    "C56-C44": {"left": "C56", "right": "C44", "description": "Plan B2 Forward相對production MR-13E exact constrained reference的整體策略結果"},
    "C56-C3": {"left": "C56", "right": "C3", "description": "Plan B2 Forward相對DL-off Min ROOS的策略經濟效果"},
    "C56-C1": {"left": "C56", "right": "C1", "description": "Plan B2 Forward相對Full ROOS的整體策略結果；不是單一參數效果"},
    "C44-C3": {"left": "C44", "right": "C3", "description": "current Min ROOS下MR-13E exact constrained score selector相對DL-off baseline的Forward-OOS策略效果"},
    "C44-C1": {"left": "C44", "right": "C1", "description": "Forward-OOS active research最終候選：Min MR-13E exact constrained相對Full ROOS的整體策略結果；不是單一參數或單一DL效果"},
    "C1-C3": {"left": "C1", "right": "C3", "description": "Full ROOS相對current Min ROOS的完整策略體系差異；不是單一參數效果"},
    "C32-C23": {"left": "C32", "right": "C23", "description": "Selection Full ROOS相對Selection Min ROOS的完整策略體系差異；不是單一參數效果"},
}


# 2026-08-15 active research matrix瘦身：下列已完成controls退出current profile，
# 僅保留舊工件解讀／重現所需的唯讀identity。
HISTORICAL_STRATEGY_DL_SOURCES.update({'CONT12B': {'filter_id': 'breakout_quality_v1',
             'model_architecture': 'inception_time_v1',
             'experiment_profile': 'strategy_aligned_no_time_all_event_pairwise',
             'threshold': None,
             'score_source': 'continuous_ranker_oos',
             'description': 'MR-12B all-event no-time pairwise ranker frozen OOS score；只允許受控strategy '
                            'research replay',
             'forward_scores_builder': None},
 'CONT13A': {'filter_id': 'breakout_quality_v1',
             'model_architecture': 'inception_time_v1',
             'experiment_profile': 'daily_universal_no_time_pairwise',
             'threshold': None,
             'score_source': 'continuous_ranker_oos',
             'description': 'MR-13A Daily Universal frozen Forward-OOS continuous '
                            'score；每個盤前決策使用最新已完成交易日資訊，供frozen strategy Gate',
             'forward_scores_builder': None},
 'CONT12B_PIT': {'filter_id': 'breakout_quality_v1',
                 'model_architecture': 'inception_time_v1',
                 'experiment_profile': 'strategy_aligned_no_time_all_event_pairwise',
                 'threshold': None,
                 'score_source': 'selection_point_in_time',
                 'description': 'MR-12B Selection point-in-time continuous score；只供2014～2020無前視策略經濟驗證',
                 'forward_scores_builder': {'enabled': True,
                                            'builder_type': 'selection_pit_from_existing_folds',
                                            'options': {'resume': True, 'allow_stale_source': False}}},
 'CONT13A_PIT': {'filter_id': 'breakout_quality_v1',
                 'model_architecture': 'inception_time_v1',
                 'experiment_profile': 'daily_universal_no_time_pairwise',
                 'threshold': None,
                 'score_source': 'selection_point_in_time',
                 'description': 'MR-13A Daily Universal Selection point-in-time '
                                'score；每個盤前決策只使用最新已完成交易日資訊，供2014～2020無前視策略經濟驗證',
                 'forward_scores_builder': {'enabled': True,
                                            'builder_type': 'selection_pit_from_existing_folds',
                                            'options': {'resume': True, 'allow_stale_source': False}}}})

HISTORICAL_STRATEGY_COMPARE_ARMS.update({'C20': {'name': 'Min MR-12B',
         'description': '與C18使用完全相同K/R0、feasible-ascent與basket內Min ROOS執行順序；唯一模型差異為DL source改成MR-12B '
                        'pairwise ranker',
         'param_source': 'min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT12B',
         'dl_runtime_mode': 'resource-aware-continuous-max-dl-feasible-ascent',
         'robustness_role': 'stochastic'},
 'C25': {'name': 'Min MR-12B',
         'description': '與C23使用完全相同historical P2 Min ROOS params；使用MR-12B Selection PIT score並完全沿用C18 '
                        'feasible-ascent selector',
         'param_source': 'selection_min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT12B_PIT',
         'dl_runtime_mode': 'resource-aware-continuous-max-dl-feasible-ascent',
         'robustness_role': 'stochastic'},
 'C28': {'name': 'Min MR-13A',
         'description': '與C25使用完全相同historical P2 Min ROOS params、K/R0與feasible-ascent selector；唯一DL差異為score '
                        'source改成MR-13A Daily Universal Selection PIT，盤前每日依最新已完成交易日score重排',
         'param_source': 'selection_min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13A_PIT',
         'dl_runtime_mode': 'resource-aware-continuous-max-dl-feasible-ascent',
         'robustness_role': 'stochastic'},
 'C35': {'name': 'Min MR-13E',
         'description': '與C25/C28使用完全相同historical P2 Min ROOS params、K/R0、feasible-ascent '
                        'selector與execution；唯一DL差異為MR-13E Selection PIT source',
         'param_source': 'selection_min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13E_PIT',
         'dl_runtime_mode': 'resource-aware-continuous-max-dl-feasible-ascent',
         'robustness_role': 'off'},
 'C39': {'name': 'Min MR-13E Excess-Alpha',
         'description': '與C35使用完全相同MR-13E Selection PIT source、historical Min '
                        'params、K/R0、sizing、cash、orderability與execution；唯一portfolio變更為以PIT daily '
                        'percentile→Expected Excess-R的單調isotonic mapping，最大化Σ(Expected Excess-R × canonical '
                        'planned initial risk)',
         'param_source': 'selection_min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13E_PIT',
         'dl_runtime_mode': 'resource-aware-continuous-excess-alpha-feasible-ascent',
         'dl_runtime_options': {'expected_excess_r_fit_dl_id': 'CONT13E_PIT',
                                'expected_excess_r_calibration_method': 'daily_score_percentile_isotonic_excess_r_v1',
                                'preserve_k_r0': True,
                                'negative_expected_excess_r_allowed': True,
                                'selection_only': True},
         'robustness_role': 'off'},
 'C40': {'name': 'Min MR-13E Excess-Alpha No-R0',
         'description': 'C39的Selection-only resource ablation：完全重用同一MR-13E PIT score、Expected Excess-R '
                        'calibration、historical Min params、K、sizing、cash、orderability與execution；唯一移除Min ROOS '
                        'reserved-capital R0 hard floor與R0-driven minimum repair。若raw Top-K因真正cash '
                        'constraint無法掛出K筆，只以同參數baseline作K-only cash-feasible seed，再最大化Σ(Expected Excess-R × '
                        'canonical planned initial risk)做single-swap ascent',
         'param_source': 'selection_min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13E_PIT',
         'dl_runtime_mode': 'resource-aware-continuous-excess-alpha-no-r0-feasible-ascent',
         'dl_runtime_options': {'expected_excess_r_fit_dl_id': 'CONT13E_PIT',
                                'expected_excess_r_calibration_method': 'daily_score_percentile_isotonic_excess_r_v1',
                                'preserve_k': True,
                                'preserve_r0': False,
                                'r0_minimum_repair': False,
                                'negative_expected_excess_r_allowed': True,
                                'selection_only': True},
         'robustness_role': 'off'},
 'C41': {'name': 'Min MR-13E Excess-Alpha Constrained',
         'description': 'C39的Selection-only solver ablation：完全重用同一MR-13E PIT score、Expected Excess-R '
                        'calibration、historical Min '
                        'params、K/R0、sizing、cash、orderability與execution；不再使用Top-K→R0 minimum repair→1-swap '
                        'ascent，而是直接以deterministic exact branch-and-bound在完整候選universe中最大化Σ(Expected '
                        'Excess-R × canonical planned initial risk)，subject to exact K、R0與canonical cash '
                        'feasibility',
         'param_source': 'selection_min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13E_PIT',
         'dl_runtime_mode': 'resource-aware-continuous-excess-alpha-constrained-optimal',
         'dl_runtime_options': {'expected_excess_r_fit_dl_id': 'CONT13E_PIT',
                                'expected_excess_r_calibration_method': 'daily_score_percentile_isotonic_excess_r_v1',
                                'preserve_k_r0': True,
                                'constrained_solver': 'exact_branch_and_bound_v1',
                                'negative_expected_excess_r_allowed': True,
                                'selection_only': True},
         'robustness_role': 'off'},
 'C43': {'name': 'Min MR-13A Constrained',
         'description': 'C28的Selection-only solver ablation：完全重用同一MR-13A PIT score、historical Min '
                        'params、K/R0、sizing、cash、orderability與execution；objective仍為原始ΣMR-13A '
                        'score，只把Top-K→R0 minimum repair→1-swap ascent替換為C42同源deterministic exact '
                        'branch-and-bound，在完整候選universe直接求K/R0/canonical-cash feasible score-sum global '
                        'optimum',
         'param_source': 'selection_min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13A_PIT',
         'dl_runtime_mode': 'resource-aware-continuous-score-constrained-optimal',
         'dl_runtime_options': {'preserve_k_r0': True,
                                'constrained_solver': 'exact_branch_and_bound_v1',
                                'selection_only': True},
         'robustness_role': 'off'},
 'C29': {'name': 'Min MR-13A',
         'description': '與C20使用相同current Min ROOS、all-off rules與frozen feasible-ascent；唯一DL source差異為MR-13A '
                        'Daily Universal Forward-OOS score',
         'param_source': 'min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13A',
         'dl_runtime_mode': 'resource-aware-continuous-max-dl-feasible-ascent',
         'robustness_role': 'stochastic'},
 'C36': {'name': 'Min MR-13E',
         'description': '與C20/C29使用相同current Min ROOS、all-off rules與frozen feasible-ascent；唯一DL '
                        'source差異為MR-13E Daily Universal Forward-OOS score',
         'param_source': 'min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13E',
         'dl_runtime_mode': 'resource-aware-continuous-max-dl-feasible-ascent',
         'robustness_role': 'off'}})

HISTORICAL_STRATEGY_COMPARE_CONTRASTS.update({'C20-C3': {'left': 'C20',
            'right': 'C3',
            'description': 'current Min ROOS下MR-12B feasible-ascent相對DL-off baseline的Forward-OOS策略效果'},
 'C25-C23': {'left': 'C25',
             'right': 'C23',
             'description': 'Selection PIT下固定historical Min ROOS與C18 selector，MR-12B PIT ranking相對DL-off '
                            'baseline的經濟效果'},
 'C28-C25': {'left': 'C28',
             'right': 'C25',
             'description': '固定historical Min ROOS與feasible-ascent selector，MR-13A daily PIT相對MR-12B event '
                            'PIT的純DL source效果'},
 'C28-C23': {'left': 'C28',
             'right': 'C23',
             'description': 'Selection PIT下MR-13A daily feasible-ascent相對DL-off historical Min ROOS '
                            'baseline的策略經濟效果'},
 'C29-C3': {'left': 'C29',
            'right': 'C3',
            'description': 'current Min ROOS下MR-13A daily feasible-ascent相對DL-off baseline的Forward-OOS策略效果'},
 'C29-C20': {'left': 'C29',
             'right': 'C20',
             'description': '固定current Min ROOS與feasible-ascent，MR-13A daily相對MR-12B event source的純DL '
                            'Forward-OOS效果'},
 'C35-C23': {'left': 'C35',
             'right': 'C23',
             'description': 'Selection PIT下MR-13E daily feasible-ascent相對DL-off historical Min ROOS '
                            'baseline的策略經濟效果'},
 'C35-C25': {'left': 'C35',
             'right': 'C25',
             'description': '固定Selection Min ROOS與feasible-ascent，MR-13E相對MR-12B的純DL source效果'},
 'C35-C28': {'left': 'C35',
             'right': 'C28',
             'description': '固定Selection Min ROOS與feasible-ascent，MR-13E相對MR-13A的純DL source效果'},
 'C39-C23': {'left': 'C39',
             'right': 'C23',
             'description': 'Selection PIT frozen MR-13E Excess-Alpha相對DL-off Min ROOS的策略經濟效果'},
 'C39-C25': {'left': 'C39',
             'right': 'C25',
             'description': 'Selection PIT frozen MR-13E Excess-Alpha相對MR-12B runtime anchor'},
 'C39-C35': {'left': 'C39',
             'right': 'C35',
             'description': '同一MR-13E PIT source與同K/R0；只比較Expected Excess-R×Risk objective相對score-sum '
                            'objective'},
 'C40-C23': {'left': 'C40',
             'right': 'C23',
             'description': 'Selection PIT frozen MR-13E Excess-Alpha No-R0相對DL-off Min ROOS的策略經濟效果'},
 'C40-C35': {'left': 'C40',
             'right': 'C35',
             'description': '同一MR-13E PIT source；比較No-R0 Excess-Alpha×Risk相對原C35 score-sum+K/R0 selector'},
 'C40-C39': {'left': 'C40',
             'right': 'C39',
             'description': '同一MR-13E PIT、同Expected Excess-R objective與同K；唯一移除R0 hard floor與R0-driven '
                            'minimum repair'},
 'C41-C23': {'left': 'C41',
             'right': 'C23',
             'description': 'Selection PIT frozen MR-13E Excess-Alpha constrained optimum相對DL-off Min '
                            'ROOS的策略經濟效果'},
 'C41-C35': {'left': 'C41',
             'right': 'C35',
             'description': '同一MR-13E PIT與同K/R0；比較exact constrained Excess-Alpha objective相對原C35 score-sum '
                            'selector'},
 'C41-C39': {'left': 'C41',
             'right': 'C39',
             'description': '同一MR-13E PIT、Expected Excess-R objective、K/R0與execution；唯一把repair+1-swap '
                            'heuristic改為完整候選exact constrained optimization'},
 'C41-C40': {'left': 'C41',
             'right': 'C40',
             'description': '同一MR-13E Excess-Alpha objective與K；比較恢復R0且直接exact constrained '
                            'optimization相對No-R0 ablation'},
 'C42-C35': {'left': 'C42',
             'right': 'C35',
             'description': '同一MR-13E PIT、score objective、K/R0與execution；唯一把repair+1-swap '
                            'heuristic改為完整候選exact constrained optimization'},
 'C42-C41': {'left': 'C42',
             'right': 'C41',
             'description': '同一MR-13E PIT、K/R0、exact constrained solver與execution；唯一objective由Expected '
                            'Excess-R×Risk改回原始MR-13E score'},
 'C43-C23': {'left': 'C43',
             'right': 'C23',
             'description': 'Selection PIT frozen MR-13A score exact constrained optimum相對DL-off Min '
                            'ROOS的策略經濟效果'},
 'C43-C28': {'left': 'C43',
             'right': 'C28',
             'description': '同一MR-13A PIT、score objective、K/R0與execution；唯一把repair+1-swap '
                            'heuristic改為完整候選exact constrained optimization'},
 'C42-C43': {'left': 'C42',
             'right': 'C43',
             'description': '同一K/R0、exact constrained solver與execution；MR-13E PIT相對MR-13A PIT的純DL source效果'},
 'C36-C3': {'left': 'C36',
            'right': 'C3',
            'description': 'current Min ROOS下MR-13E daily feasible-ascent相對DL-off baseline的Forward-OOS策略效果'},
 'C36-C20': {'left': 'C36',
             'right': 'C20',
             'description': '固定current Min ROOS與feasible-ascent，MR-13E相對MR-12B的純DL Forward-OOS效果'},
 'C36-C29': {'left': 'C36',
             'right': 'C29',
             'description': '固定current Min ROOS與feasible-ascent，MR-13E相對MR-13A的純DL Forward-OOS效果'},
 'C44-C20': {'left': 'C44',
             'right': 'C20',
             'description': 'current Min ROOS下MR-13E exact constrained score selector相對MR-12B Forward '
                            'runtime anchor的策略效果'},
 'C44-C36': {'left': 'C44',
             'right': 'C36',
             'description': '同一MR-13E frozen Forward score、current Min '
                            'params、K/R0與execution；唯一把repair+1-swap heuristic改為exact constrained '
                            'optimization'}})


# 2026-08-17 MR-13J strategy-conversion研究結案：保留唯讀歷史重現 identity。
HISTORICAL_STRATEGY_DL_SOURCES.update({
    "CONT13J_PIT": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_risk_context_v1",
        "experiment_profile": "daily_universal_risk_context_net_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": (
            "MR-13J Daily Universal risk-context canonical-cost Selection PIT score；"
            "C49/C50 Selection研究已結案，只供歷史工件解讀／重現"
        ),
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
})

HISTORICAL_STRATEGY_COMPARE_ARMS.update({
    "C49": {
        "name": "Min MR-13J Constrained",
        "description": (
            "SR-C49 historical rejected model-only arm：與C42同historical Min params/all-off、"
            "K/R0、canonical sizing/cash/orderability/execution與exact solver；只換MR-13J PIT score"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13J_PIT",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C50": {
        "name": "Min MR-13J No-R0",
        "description": (
            "SR-C50 historical rejected R0 ablation：與C49同MR-13J PIT、historical Min params/all-off、"
            "K、canonical sizing/cash/orderability/execution與exact solver；只移除baseline R0 floor"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13J_PIT",
        "dl_runtime_mode": "resource-aware-continuous-score-no-r0-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k": True,
            "preserve_r0": False,
            "r0_minimum_repair": False,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
})

HISTORICAL_STRATEGY_COMPARE_CONTRASTS.update({
    "C49-C42": {
        "left": "C49", "right": "C42",
        "description": "SR-C49 historical：同Min/K/R0/exact/cash下只替換MR-13E PIT為MR-13J PIT",
    },
    "C50-C49": {
        "left": "C50", "right": "C49",
        "description": "SR-C50 historical：同MR-13J PIT/K/exact/cash下只移除R0",
    },
    "C50-C42": {
        "left": "C50", "right": "C42",
        "description": "historical MR-13J No-R0最終架構相對MR-13E + R0 C42",
    },
})



# 2026-08-17 MR-13H完整Selection/Forward 16-seed robustness結案：保留唯讀歷史重現 identity。
HISTORICAL_STRATEGY_DL_SOURCES.update({
    "CONT13H": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_no_breach_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": "MR-13H full-horizon no-breach frozen Forward-OOS score；研究已結案，只供歷史重現",
        "forward_scores_builder": None,
    },
    "CONT13H_PIT": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_full_horizon_no_breach_full_list_ndcg_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": "MR-13H full-horizon no-breach Selection PIT score；研究已結案，只供歷史重現",
        "forward_scores_builder": {
            "enabled": True,
            "builder_type": "selection_pit_from_existing_folds",
            "options": {"resume": True, "allow_stale_source": False},
        },
    },
})

HISTORICAL_STRATEGY_COMPARE_ARMS.update({
    "C51": {
        "name": "Min MR-13H Constrained",
        "description": (
            "SR-C51 historical MR-13H Selection arm：與C42同historical Min params/all-off、K/R0、"
            "canonical sizing/cash/orderability/execution與exact solver；只替換MR-13H PIT score"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13H_PIT",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C52": {
        "name": "Min MR-13H Constrained",
        "description": (
            "SR-C52 historical MR-13H Forward arm：與C44同current Min params/all-off、K/R0、"
            "canonical sizing/cash/orderability/execution與exact solver；只替換MR-13H Forward score"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13H",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": False,
        },
        "robustness_role": "off",
    },
})

HISTORICAL_STRATEGY_COMPARE_CONTRASTS.update({
    "C51-C42": {"left": "C51", "right": "C42", "description": "historical MR-13H Selection source-only contrast"},
    "C51-C23": {"left": "C51", "right": "C23", "description": "historical MR-13H Selection相對Min ROOS"},
    "C51-C32": {"left": "C51", "right": "C32", "description": "historical MR-13H Selection相對Full ROOS"},
    "C52-C44": {"left": "C52", "right": "C44", "description": "historical MR-13H Forward source-only contrast"},
    "C52-C3": {"left": "C52", "right": "C3", "description": "historical MR-13H Forward相對Min ROOS"},
    "C52-C1": {"left": "C52", "right": "C1", "description": "historical MR-13H Forward相對Full ROOS"},
})


# 2026-08-18 C56 full-flow獨立驗證：C53/C54/C55與其舊對照退出current matrix，保留唯讀歷史identity。
HISTORICAL_STRATEGY_COMPARE_ARMS.update({'C53': {'name': 'Min MR-13K Constrained',
         'description': 'Selection PIT MR-13K controlled arm：與C42完全相同historical Min params/all-off、K/R0、canonical '
                        'sizing/cash/orderability/execution與exact branch-and-bound；唯一scientific change是DL '
                        'source由MR-13E PIT替換為MR-13K PIT',
         'param_source': 'selection_min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13K_PIT',
         'dl_runtime_mode': 'resource-aware-continuous-score-constrained-optimal',
         'dl_runtime_options': {'preserve_k_r0': True,
                                'constrained_solver': 'exact_branch_and_bound_v1',
                                'selection_only': True},
         'robustness_role': 'off'},
 'C54': {'name': 'Min MR-13K Constrained',
         'description': 'Forward-OOS MR-13K controlled arm：與C44完全相同current Min params/all-off、K/R0、canonical '
                        'sizing/cash/orderability/execution與exact branch-and-bound；唯一scientific change是DL '
                        'source由MR-13E Forward替換為MR-13K Forward',
         'param_source': 'min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13K',
         'dl_runtime_mode': 'resource-aware-continuous-score-constrained-optimal',
         'dl_runtime_options': {'preserve_k_r0': True,
                                'constrained_solver': 'exact_branch_and_bound_v1',
                                'selection_only': False},
         'robustness_role': 'off'},
 'C55': {'name': 'Min MR-13K + MR-13M Safety',
         'description': 'Forward-only Plan B controlled arm：完全沿用C54的Min params/all-off、K/R0、canonical '
                        'sizing/cash/orderability/execution與MR-13K primary score objective；額外要求選中basket的MR-13M score '
                        'coverage與score-sum不得低於同日DL-off Min ROOS baseline。MR-13M只作hard safety '
                        'floor，不與MR-13K加權、不新增numeric threshold。',
         'param_source': 'min_roos',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13K',
         'dl_runtime_mode': 'resource-aware-continuous-score-safety-constrained-optimal',
         'dl_runtime_options': {'preserve_k_r0': True,
                                'constrained_solver': 'exact_branch_and_bound_v1',
                                'selection_only': False,
                                'safety_dl_id': 'CONT13M',
                                'safety_constraint': 'baseline_coverage_and_score_sum_floor_v1'},
         'robustness_role': 'off'}})

HISTORICAL_STRATEGY_COMPARE_CONTRASTS.update({'C53-C42': {'left': 'C53',
             'right': 'C42',
             'description': '同Min/K/R0/exact/cash/execution下只將MR-13E PIT替換為MR-13K PIT，隔離pure-MFE '
                            'Target模型本身的Selection策略轉化效果'},
 'C53-C23': {'left': 'C53',
             'right': 'C23',
             'description': 'Selection PIT MR-13K exact constrained相對DL-off Min ROOS的策略經濟效果'},
 'C53-C32': {'left': 'C53',
             'right': 'C32',
             'description': 'Selection PIT MR-13K exact constrained相對Full ROOS的整體策略結果；不是單一參數效果'},
 'C54-C44': {'left': 'C54',
             'right': 'C44',
             'description': '同Min/K/R0/exact/cash/execution下只將MR-13E Forward替換為MR-13K Forward，隔離pure-MFE '
                            'Target模型本身的Forward策略轉化效果'},
 'C54-C3': {'left': 'C54',
            'right': 'C3',
            'description': 'Forward-OOS MR-13K exact constrained相對DL-off Min ROOS的策略經濟效果'},
 'C54-C1': {'left': 'C54',
            'right': 'C1',
            'description': 'Forward-OOS MR-13K exact constrained相對Full ROOS的整體策略結果；不是單一參數效果'},
 'C55-C54': {'left': 'C55',
             'right': 'C54',
             'description': 'Plan B primary contrast：同MR-13K objective/K/R0/exact/cash/execution下，只新增MR-13M '
                            'baseline-relative safety floor，隔離dual-model safety constraint的Forward策略效果'},
 'C55-C44': {'left': 'C55',
             'right': 'C44',
             'description': 'Plan B dual-model Forward相對production MR-13E exact constrained reference的整體策略結果'},
 'C55-C3': {'left': 'C55', 'right': 'C3', 'description': 'Plan B dual-model Forward相對DL-off Min ROOS的策略經濟效果'},
 'C55-C1': {'left': 'C55', 'right': 'C1', 'description': 'Plan B dual-model Forward相對Full ROOS的整體策略結果；不是單一參數效果'},
 'C56-C54': {'left': 'C56',
             'right': 'C54',
             'description': 'Plan B2 primary contrast：同MR-13K objective/K/R0/exact/cash/execution下，只新增MR-13M same-day '
                            'rank residual safety floor，隔離conditional residual safety的Forward策略效果'},
 'C56-C55': {'left': 'C56',
             'right': 'C55',
             'description': 'Plan B2相對C55 raw-safety hard floor：比較residualizing 13M是否能保留更多13K upside'}})

# 2026-08-25 current-suite focus reduction.  These completed arms retain their
# exact scientific definitions for historical result/artifact interpretation;
# they are no longer part of the active extending_current matrix.
HISTORICAL_STRATEGY_COMPARE_ARMS.update({
    "C62": {
        "name": "Full Base-Finalists-Agree",
        "description": (
            "Extending-Window Full DL-off parameter-policy reference；與C61完全相同formal rules／execution，"
            "唯一差異為canonical parameter policy使用base-finalists-agree。"
        ),
        "param_source": "full_rolling",
        "param_policy": "base-finalists-agree",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "off",
    },
    "C63": {
        "name": "Min Base-Finalists-Agree",
        "description": (
            "Extending-Window Min DL-off parameter-policy reference；與C58完全相同all-off rules／execution，"
            "唯一差異為canonical parameter policy使用base-finalists-agree。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalists-agree",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "off",
    },
    "C68": {
        "name": "Min MR-13E Fixed-K Feasible-Ascent",
        "description": (
            "SR-C68 fixed-K local-search control：與C59使用相同MR-13E score、Min base-finalist-best、"
            "all-off、baseline K/R0與canonical sizing/cash/orderability/execution；唯一差異為"
            "global exact solver改為既有deterministic best-improvement feasible-ascent。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-matched-feasible-ascent",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "membership_proposal": "score_priority_then_canonical_execution_v1",
            "local_search": "deterministic_single_swap_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C69": {
        "name": "Min MR-13E K-Flex R0 Feasible-Ascent",
        "description": (
            "SR-C69 matched K-Flex local-search treatment：與C68完全相同MR-13E source與"
            "deterministic feasible-ascent family，R0仍為hard floor；唯一scientific change是"
            "允許single-add把planned-order count由baseline K擴到physical free slots，"
            "count lexicographically優先，count內仍最大化原始MR-13E score。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-k-flex-r0-feasible-ascent",
        "dl_runtime_options": {
            "count_constraint": "baseline_k_to_physical_free_slots_local_v1",
            "preserve_r0": True,
            "membership_proposal": "score_priority_then_canonical_execution_v1",
            "local_search": "deterministic_single_add_swap_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
    "C60": {
        "name": "Min MR-13K + MR-13M Residual Safety",
        "description": (
            "Extending-Window B2：MR-13K primary + MR-13M same-day rank OLS residual-safety floor；"
            "兩個score必須來自同一evaluation mode／同一information-cutoff namespace。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13K_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-residual-safety-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
            "safety_dl_id": "CONT13M_ROLL",
            "safety_constraint": "baseline_residual_coverage_and_score_sum_floor_v1",
            "safety_residualization": "same_day_rank_ols_v1",
        },
        "robustness_role": "off",
    },
    "C65": {
        "name": "Min MR-13Q Conditional-MFE Single",
        "description": (
            "MR-13Q Single-head reverse-Conditional-MFE conversion arm；與C58共用Min base-finalist-best、"
            "all-off、K/R0與canonical execution，唯一model change是直接最大化J=U-E(U|S) prediction。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13Q_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-constrained-optimal",
        "dl_runtime_options": {
            "preserve_k_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
})

HISTORICAL_STRATEGY_COMPARE_CONTRASTS.update({
    "C62-C61": {"left": "C62", "right": "C61", "description": "{left}相對{right}的Full parameter-policy單變量差異"},
    "C63-C58": {"left": "C63", "right": "C58", "description": "{left}相對{right}的Min parameter-policy單變量差異"},
    "C62-C63": {"left": "C62", "right": "C63", "description": "base-finalists-agree政策下Full相對Min的完整策略體系差異"},
    "C68-C59": {"left": "C68", "right": "C59", "description": "dedicated matched fixed-K feasible-ascent相對exact C59；隔離global exact→matched deterministic local-search solver effect"},
    "C69-C68": {"left": "C69", "right": "C68", "description": "同一feasible-ascent family下K-Flex相對fixed-K；唯一scientific change為baseline K可擴到physical free slots，R0保留"},
    "C69-C59": {"left": "C69", "right": "C59", "description": "practical K-Flex local treatment相對current exact-K C59的淨策略效果"},
    "C69-C58": {"left": "C69", "right": "C58", "description": "K-Flex / R0-preserved feasible-ascent MR-13E相對Min baseline整體效果"},
    "C60-C58": {"left": "C60", "right": "C58", "description": "{left}相對{right}的增量策略效果"},
    "C60-C61": {"left": "C60", "right": "C61", "description": "{left}相對{right}的整體策略結果；不是單一DL效果"},
    "C60-C59": {"left": "C60", "right": "C59", "description": "{left}相對{right}的同政策比較"},
    "C64-C60": {"left": "C64", "right": "C60", "description": "model-level conditional safety相對C60 runtime post-hoc residual safety"},
    "C65-C58": {"left": "C65", "right": "C58", "description": "Single-head Conditional-MFE相對Min baseline的增量策略效果"},
    "C65-C59": {"left": "C65", "right": "C59", "description": "Single-head Conditional-MFE相對C59 conversion reference"},
    "C65-C64": {"left": "C65", "right": "C64", "description": "直接maximize Conditional-MFE相對MR-13P safety hard-floor formulation"},
    "C66-C65": {"left": "C66", "right": "C65", "description": "顯式Safety condition head相對Single-head的architecture controlled contrast"},
})

# 2026-08-25 SR-C67 compute-blocked K-Flex exact experiment.  Identity is retained
# for historical artifact interpretation only; current suite moved to the matched
# C68/C69 deterministic feasible-ascent pair because large-K global exact
# certification was operationally intractable.
HISTORICAL_STRATEGY_COMPARE_ARMS.update({
    "C67": {
        "name": "Min MR-13E K-Flex R0 Exact (Compute-Blocked)",
        "description": (
            "SR-C67 historical compute-blocked arm：沿用C59 MR-13E/Min/all-off/R0，"
            "將count放寬到baseline K至physical free slots後仍要求global exact score optimum；"
            "兩次實際OOS執行均在C67 replay長時間無結果，因此未取得策略績效。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13E_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-k-flex-r0-constrained-optimal",
        "dl_runtime_options": {
            "count_constraint": "baseline_k_to_physical_free_slots_v1",
            "preserve_r0": True,
            "constrained_solver": "exact_branch_and_bound_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
})
HISTORICAL_STRATEGY_COMPARE_CONTRASTS.update({
    "C67-C59": {"left": "C67", "right": "C59", "description": "historical compute-blocked K-Flex exact contrast"},
    "C67-C58": {"left": "C67", "right": "C58", "description": "historical compute-blocked K-Flex exact baseline contrast"},
})



# 2026-08-26 SR-C70 left the current suite after C71 Safety-gate mechanism evidence.
# Keep its exact runtime identity/results interpretable as historical evidence.
HISTORICAL_STRATEGY_COMPARE_ARMS.update({
    "C70": {
        "name": "Min MR-13R Conditional-MFE No-K No-R0",
        "description": (
            "SR-C70 joint resource-ablation：與C66使用完全相同MR-13R CONT13R_ROLL final "
            "Conditional-MFE model_score、Min base-finalist-best、all-off、canonical sizing/cash/"
            "orderability/execution；唯一scientific change是移除C58-derived K與R0。"
        ),
        "param_source": "min_rolling",
        "param_policy": "base-finalist-best",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13R_ROLL",
        "dl_runtime_mode": "resource-aware-continuous-score-no-k-no-r0",
        "dl_runtime_options": {
            "preserve_k": False,
            "preserve_r0": False,
            "selection_order": "model_score_desc_then_canonical_tie_v1",
            "selection_only": True,
        },
        "robustness_role": "off",
    },
})

HISTORICAL_STRATEGY_COMPARE_CONTRASTS.update({
    "C70-C66": {"left": "C70", "right": "C66", "description": "MR-13R完全相同model/ranking下移除C58-derived K與R0的joint resource-ablation；primary contrast"},
    "C70-C59": {"left": "C70", "right": "C59", "description": "No-K/No-R0 MR-13R相對current MR-13E constrained reference的淨策略效果"},
    "C70-C58": {"left": "C70", "right": "C58", "description": "No-K/No-R0 MR-13R相對Min DL-off baseline的完整增量效果"},
    "C70-C64": {"left": "C70", "right": "C64", "description": "No-K/No-R0 MR-13R相對MR-13P Conditional Safety的策略效果"},
    "C71-C70": {"left": "C71", "right": "C70", "description": "完全相同MR-13R No-K/No-R0下只新增Raw Safety同日orderable percentile>=0.50 eligibility gate；primary contrast"},
})

HISTORICAL_STRATEGY_COMPARE_CONTRASTS.update({
    "C71-C59": {"left": "C71", "right": "C59", "description": "Raw-Safety gated No-K/No-R0 MR-13R相對MR-13E constrained reference"},
    "C71-C58": {"left": "C71", "right": "C58", "description": "Raw-Safety gated No-K/No-R0 MR-13R相對Min DL-off baseline"},
})

__all__ = [
    "HISTORICAL_STRATEGY_PARAM_SOURCES",
    "HISTORICAL_STRATEGY_DL_SOURCES",
    "HISTORICAL_STRATEGY_COMPARE_ARMS",
    "HISTORICAL_STRATEGY_COMPARE_CONTRASTS",
]

# 2026-08-28 current matrix cleanup: retired completed C64/C66/C71-C75 from active catalog.
HISTORICAL_STRATEGY_DL_SOURCES.update({'CONT13K_ROLL': {'filter_id': 'breakout_quality_v1',
                  'model_architecture': 'inception_time_v1',
                  'experiment_profile': 'daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise',
                  'threshold': None,
                  'score_source': 'selection_point_in_time',
                  'description': 'MR-13K Extending-Window Rolling PIT-safe pure-MFE score；expanding history + '
                                 'mode-specific refit cadence。',
                  'forward_scores_builder': {'enabled': True,
                                             'builder_type': 'selection_pit_from_existing_folds',
                                             'options': {'resume': True, 'allow_stale_source': False}}},
 'CONT13M_ROLL': {'filter_id': 'breakout_quality_v1',
                  'model_architecture': 'inception_time_v1',
                  'experiment_profile': 'daily_universal_full_horizon_low_adverse_full_list_ndcg_pairwise',
                  'threshold': None,
                  'score_source': 'selection_point_in_time',
                  'description': 'MR-13M Extending-Window Rolling PIT-safe low-adverse score；只作C60 residual-safety '
                                 'secondary source。',
                  'forward_scores_builder': {'enabled': True,
                                             'builder_type': 'selection_pit_from_existing_folds',
                                             'options': {'resume': True, 'allow_stale_source': False}}},
 'CONT13P_ROLL': {'filter_id': 'breakout_quality_v1',
                  'model_architecture': 'inception_time_conditional_mfe_safety_v1',
                  'experiment_profile': 'daily_universal_conditional_mfe_safety_full_list_ndcg_pairwise',
                  'threshold': None,
                  'score_source': 'selection_point_in_time',
                  'description': 'MR-13P Extending-Window single-model dual-head PIT '
                                 'score；primary=Pure-MFE，secondary=learned Conditional Safety。',
                  'forward_scores_builder': {'enabled': True,
                                             'builder_type': 'selection_pit_from_existing_folds',
                                             'options': {'resume': True, 'allow_stale_source': False}}},
 'CONT13Q_ROLL': {'filter_id': 'breakout_quality_v1',
                  'model_architecture': 'inception_time_v1',
                  'experiment_profile': 'daily_universal_conditional_mfe_single_head_full_list_ndcg_pairwise',
                  'threshold': None,
                  'score_source': 'selection_point_in_time',
                  'description': 'MR-13Q reverse-conditional MFE single-head PIT score；final score直接代表J=U-E(U|S)。',
                  'forward_scores_builder': {'enabled': True,
                                             'builder_type': 'selection_pit_from_existing_folds',
                                             'options': {'resume': True, 'allow_stale_source': False}}},
 'CONT13R_ROLL': {'filter_id': 'breakout_quality_v1',
                  'model_architecture': 'inception_time_safety_conditional_mfe_v1',
                  'experiment_profile': 'daily_universal_safety_conditional_mfe_duo_head_full_list_ndcg_pairwise',
                  'threshold': None,
                  'score_source': 'selection_point_in_time',
                  'description': 'MR-13R Safety→Conditional-MFE duo-head PIT score；strategy只使用final Conditional-MFE '
                                 'head。',
                  'forward_scores_builder': {'enabled': True,
                                             'builder_type': 'selection_pit_from_existing_folds',
                                             'options': {'resume': True, 'allow_stale_source': False}}},
 'CONT13Z_ROLL': {'filter_id': 'breakout_quality_v1',
                  'model_architecture': 'patch_token_transformer_safety_raw_mfe_joint_attn_mlp_v1',
                  'experiment_profile': 'daily_universal_safety_raw_mfe_joint_min_patch_transformer_attn_pool_mlp_head_full_list_ndcg_pairwise',
                  'threshold': None,
                  'score_source': 'selection_point_in_time',
                  'description': 'MR-13Z Patch Transformer Joint-Min PIT-safe tri-head score；Strategy '
                                 'C75明確消費joint_min_score，不把Raw-MFE breakout_quality_score偷換語意。',
                  'forward_scores_builder': {'enabled': True,
                                             'builder_type': 'selection_pit_from_existing_folds',
                                             'options': {'resume': True, 'allow_stale_source': False}}}})

HISTORICAL_STRATEGY_COMPARE_ARMS.update({'C64': {'name': 'Min MR-13P Conditional Safety',
         'description': 'MR-13P single-model conversion Gate：Primary直接最大化Pure-MFE head；同一PIT '
                        'artifact的conditional_safety_score直接作baseline-relative safety floor，training '
                        'target已conditionalize，因此runtime不再做OLS residualization。',
         'param_source': 'min_rolling',
         'param_policy': 'base-finalist-best',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13P_ROLL',
         'dl_runtime_mode': 'resource-aware-continuous-score-safety-constrained-optimal',
         'dl_runtime_options': {'preserve_k_r0': True,
                                'constrained_solver': 'exact_branch_and_bound_v1',
                                'selection_only': True,
                                'safety_dl_id': 'CONT13P_ROLL',
                                'safety_score_column': 'conditional_safety_score',
                                'safety_constraint': 'baseline_coverage_and_score_sum_floor_v1'},
         'robustness_role': 'off'},
 'C66': {'name': 'Min MR-13R Conditional-MFE Duo',
         'description': 'MR-13R Duo-head Safety→Conditional-MFE conversion arm；與C65完全相同strategy contract，唯一差異是模型內顯式Raw '
                        'Safety condition head；strategy仍只最大化final Conditional-MFE score。',
         'param_source': 'min_rolling',
         'param_policy': 'base-finalist-best',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13R_ROLL',
         'dl_runtime_mode': 'resource-aware-continuous-score-constrained-optimal',
         'dl_runtime_options': {'preserve_k_r0': True,
                                'constrained_solver': 'exact_branch_and_bound_v1',
                                'selection_only': True},
         'robustness_role': 'off'},
 'C71': {'name': 'Min MR-13R Conditional-MFE Raw-Safety P50 No-K No-R0',
         'description': 'SR-C71 Safety-only eligibility treatment：與C70完全相同MR-13R CONT13R_ROLL final Conditional-MFE '
                        'model_score、No-K/No-R0、Min base-finalist-best、all-off、canonical '
                        'sizing/cash/orderability/execution；唯一scientific change是同一MR-13R PIT artifact的Raw Safety '
                        'head在當日orderable candidate cross-section轉average-rank '
                        'percentile，percentile>=0.50才可進既有score-desc selector。',
         'param_source': 'min_rolling',
         'param_policy': 'base-finalist-best',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13R_ROLL',
         'dl_runtime_mode': 'resource-aware-continuous-score-no-k-no-r0-raw-safety-gate',
         'dl_runtime_options': {'preserve_k': False,
                                'preserve_r0': False,
                                'selection_order': 'model_score_desc_then_canonical_tie_v1',
                                'safety_gate': 'same_day_orderable_percentile_gte_v1',
                                'safety_percentile_cutoff': 0.5,
                                'safety_dl_id': 'CONT13R_ROLL',
                                'safety_score_column': 'raw_safety_score',
                                'selection_only': True},
         'robustness_role': 'off'},
 'C72': {'name': 'Min MR-13R Conditional-MFE Raw-Safety P60 No-K No-R0',
         'description': 'SR-C72 Safety-gate sensitivity treatment：與C71完全相同MR-13R CONT13R_ROLL final Conditional-MFE '
                        'model_score、No-K/No-R0、Min base-finalist-best、all-off、canonical '
                        'sizing/cash/orderability/execution；唯一scientific change是同一MR-13R PIT artifact的Raw Safety '
                        'head在當日orderable candidate cross-section轉average-rank '
                        'percentile，percentile>=0.60才可進既有score-desc selector。',
         'param_source': 'min_rolling',
         'param_policy': 'base-finalist-best',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13R_ROLL',
         'dl_runtime_mode': 'resource-aware-continuous-score-no-k-no-r0-raw-safety-gate',
         'dl_runtime_options': {'preserve_k': False,
                                'preserve_r0': False,
                                'selection_order': 'model_score_desc_then_canonical_tie_v1',
                                'safety_gate': 'same_day_orderable_percentile_gte_v1',
                                'safety_percentile_cutoff': 0.6,
                                'safety_dl_id': 'CONT13R_ROLL',
                                'safety_score_column': 'raw_safety_score',
                                'selection_only': True},
         'robustness_role': 'off'},
 'C73': {'name': 'Min MR-13R Conditional-MFE Raw-Safety P70 No-K No-R0',
         'description': 'SR-C73 Safety-gate sensitivity treatment：與C71完全相同MR-13R CONT13R_ROLL final Conditional-MFE '
                        'model_score、No-K/No-R0、Min base-finalist-best、all-off、canonical '
                        'sizing/cash/orderability/execution；唯一scientific change是同一MR-13R PIT artifact的Raw Safety '
                        'head在當日orderable candidate cross-section轉average-rank '
                        'percentile，percentile>=0.70才可進既有score-desc selector。',
         'param_source': 'min_rolling',
         'param_policy': 'base-finalist-best',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13R_ROLL',
         'dl_runtime_mode': 'resource-aware-continuous-score-no-k-no-r0-raw-safety-gate',
         'dl_runtime_options': {'preserve_k': False,
                                'preserve_r0': False,
                                'selection_order': 'model_score_desc_then_canonical_tie_v1',
                                'safety_gate': 'same_day_orderable_percentile_gte_v1',
                                'safety_percentile_cutoff': 0.7,
                                'safety_dl_id': 'CONT13R_ROLL',
                                'safety_score_column': 'raw_safety_score',
                                'selection_only': True},
         'robustness_role': 'off'},
 'C74': {'name': 'Min MR-13R Raw-Safety × Conditional-MFE Product No-K No-R0',
         'description': 'SR-C74 joint-selector treatment：與C71-C73共用同一MR-13R CONT13R_ROLL checkpoint/PIT source、Min '
                        'base-finalist-best、all-off、No-K/No-R0與canonical sizing/cash/orderability/execution。不使用Raw '
                        'Safety hard gate；在每日orderable candidate cross-section內，Raw Safety與final '
                        'Conditional-MFE各自轉average-rank percentile，selector唯一排序分數為兩者乘積。沒有threshold、沒有fitted '
                        'weight、沒有score fusion calibration；缺任一head score者不產生joint order。',
         'param_source': 'min_rolling',
         'param_policy': 'base-finalist-best',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13R_ROLL',
         'dl_runtime_mode': 'resource-aware-continuous-score-no-k-no-r0-safety-mfe-product',
         'dl_runtime_options': {'preserve_k': False,
                                'preserve_r0': False,
                                'selection_order': 'same_day_orderable_percentile_product_desc_then_canonical_tie_v1',
                                'joint_score_transform': 'raw_safety_pct_x_conditional_mfe_pct_v1',
                                'safety_dl_id': 'CONT13R_ROLL',
                                'safety_score_column': 'raw_safety_score',
                                'selection_only': True},
         'robustness_role': 'off'},
 'C75': {'name': 'Min MR-13Z Joint-Min No-K No-R0',
         'description': 'MR-13Z Model-Gate通過後的第一個PIT-safe conversion arm；Min '
                        'base-finalist-best、all-off、No-K/No-R0、canonical '
                        'sizing/cash/orderability/execution與C74一致。唯一selector score直接使用同一CONT13Z_ROLL PIT '
                        'artifact的joint_min_score descending；不加Safety gate、threshold、percentile product、fitted '
                        'weight或calibration。',
         'param_source': 'min_rolling',
         'param_policy': 'base-finalist-best',
         'rule_policy': 'all_off',
         'dl_enabled': True,
         'dl_id': 'CONT13Z_ROLL',
         'dl_runtime_mode': 'resource-aware-continuous-score-no-k-no-r0',
         'dl_runtime_options': {'preserve_k': False,
                                'preserve_r0': False,
                                'selection_order': 'model_score_desc_then_canonical_tie_v1',
                                'primary_score_column': 'joint_min_score',
                                'selection_only': True},
         'robustness_role': 'off'}})

HISTORICAL_STRATEGY_COMPARE_CONTRASTS.update({'C64-C58': {'left': 'C64', 'right': 'C58', 'description': 'MR-13P conditional conversion相對Min baseline的增量策略效果'},
 'C64-C59': {'left': 'C64',
             'right': 'C59',
             'description': 'MR-13P conditional conversion相對目前C59 MR-13E conversion reference'},
 'C66-C58': {'left': 'C66', 'right': 'C58', 'description': 'Duo-head Conditional-MFE相對Min baseline的增量策略效果'},
 'C66-C59': {'left': 'C66', 'right': 'C59', 'description': 'Duo-head Conditional-MFE相對C59 conversion reference'},
 'C66-C64': {'left': 'C66',
             'right': 'C64',
             'description': 'Duo-head Conditional-MFE相對MR-13P safety hard-floor formulation'},
 'C71-C66': {'left': 'C71', 'right': 'C66', 'description': 'Raw-Safety gated No-K/No-R0 MR-13R相對原fixed-K/R0 C66的淨策略效果'},
 'C71-C64': {'left': 'C71',
             'right': 'C64',
             'description': 'MR-13R Raw-Safety eligibility formulation相對MR-13P Conditional Safety resource-floor '
                            'formulation'},
 'C72-C71': {'left': 'C72',
             'right': 'C71',
             'description': '完全相同MR-13R No-K/No-R0 Raw-Safety gate下，cutoff 0.60相對0.50的Safety/upside邊際效果；primary '
                            'sensitivity contrast'},
 'C73-C72': {'left': 'C73',
             'right': 'C72',
             'description': '完全相同MR-13R No-K/No-R0 Raw-Safety gate下，cutoff 0.70相對0.60的Safety/upside邊際效果；primary '
                            'sensitivity contrast'},
 'C73-C66': {'left': 'C73',
             'right': 'C66',
             'description': 'P70 Raw-Safety gated No-K/No-R0 MR-13R相對原fixed-K/R0 C66 reference'},
 'C73-C64': {'left': 'C73',
             'right': 'C64',
             'description': 'P70 Raw-Safety gated MR-13R相對MR-13P Conditional Safety reference'},
 'C74-C71': {'left': 'C74',
             'right': 'C71',
             'description': '同一MR-13R No-K/No-R0下，parameter-free Safety×Conditional-MFE percentile product相對P50 '
                            'hard-gate+Conditional-MFE排序；primary joint-selector contrast'},
 'C74-C72': {'left': 'C74',
             'right': 'C72',
             'description': 'Safety×Conditional-MFE percentile product相對P60 hard-gate reference'},
 'C74-C73': {'left': 'C74',
             'right': 'C73',
             'description': 'Safety×Conditional-MFE percentile product相對P70 hard-gate reference'},
 'C74-C66': {'left': 'C74',
             'right': 'C66',
             'description': 'Safety×Conditional-MFE percentile product No-K/No-R0相對原MR-13R fixed-K/R0 C66 reference'},
 'C74-C64': {'left': 'C74',
             'right': 'C64',
             'description': 'Safety×Conditional-MFE percentile product相對MR-13P Conditional Safety reference'},
 'C75-C74': {'left': 'C75',
             'right': 'C74',
             'description': '同為Min/all-off/No-K/No-R0，MR-13Z learned Joint-Min direct '
                            'score相對MR-13R手工Safety×Conditional-MFE percentile product；primary conversion contrast'},
 'C75-C71': {'left': 'C75',
             'right': 'C71',
             'description': 'MR-13Z learned Joint-Min direct score相對MR-13R P50 Raw-Safety gate + Conditional-MFE '
                            'ranking；secondary conversion reference'}})

