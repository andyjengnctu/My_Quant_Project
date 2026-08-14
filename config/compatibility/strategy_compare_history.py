"""Strategy Compare 歷史唯讀相容定義。

本模組保存退役 research arms、contrasts、DL sources 與 parameter sources，
只供舊工件解讀／重現。目前正式 Selection／Forward 設定位於
``config/strategy_compare.py``；歷史定義不得直接加入 current profile，除非先
明確重新納入 active config。
"""

from __future__ import annotations

from config.execution_policy import DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT
from config.training_policy import OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT

# 只供 SR-C26 歷史相容的固定語意。
STRATEGY_COMPARE_STALE_SCORE_MEMBERSHIP_GUARD_MAX_AGE_DAYS = 22

HISTORICAL_STRATEGY_PARAM_SOURCES = {
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
}

__all__ = [
    "HISTORICAL_STRATEGY_PARAM_SOURCES",
    "HISTORICAL_STRATEGY_DL_SOURCES",
    "HISTORICAL_STRATEGY_COMPARE_ARMS",
    "HISTORICAL_STRATEGY_COMPARE_CONTRASTS",
]
