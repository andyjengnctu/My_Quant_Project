"""策略績效比較設定。

比較對象、差異、工件來源與前置建立政策全部逐項列出，直接以
``enabled`` 或本檔欄位調整。正式 App 與執行引擎不保存特定實驗矩陣。
"""

from __future__ import annotations

from core.strategy_comparison import (
    StrategyArtifactBuilder,
    StrategyComparisonArm,
    StrategyComparisonContrast,
    StrategyComparisonSettings,
    StrategyDLSource,
    StrategyParameterSource,
    StrategyPreparationPolicy,
    validate_strategy_comparison_settings,
)

STRATEGY_COMPARE_SCHEMA_VERSION = 8

# =============================================================================
# 1. 共用執行設定
# =============================================================================

STRATEGY_COMPARE_DATASET = "full"
STRATEGY_COMPARE_START_DATE: str | None = None
STRATEGY_COMPARE_END_DATE: str | None = None
STRATEGY_COMPARE_PARAM_POLICY = "base-finalist-best"
STRATEGY_COMPARE_MAX_POSITIONS = 10
STRATEGY_COMPARE_ROTATION = "off"
STRATEGY_COMPARE_OUTPUT_ROOT = "outputs/strategy_compare"

# =============================================================================
# 2. 前置工件政策
# =============================================================================

STRATEGY_COMPARE_PREPARATION = {
    "auto_prepare": True,
    "reuse_ready_artifacts": True,
    "rebuild_stale_artifacts": True,
    "resume_parameter_training": True,
    "require_confirmation": True,
}

# =============================================================================
# 3. 策略參數工件來源
# =============================================================================

STRATEGY_PARAM_SOURCES = {
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
            "builder_type": "binary_dl_risk_only_rolling",
            "options": {
                "parameter_set": "p2",
                "model_source_id": "TP1",
                "trials_per_fold": 200,
                "resume": True,
                "fixed_risk": 0.01,
                "max_position_cap_pct": 0.30,
                "build_binary_pit": False,
                "binary_pit_resume": True,
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
            "builder_type": "binary_dl_risk_only_rolling",
            "options": {
                "parameter_set": "p3",
                "model_source_id": "TP1",
                "p3_variant": None,
                "trials_per_fold": 200,
                "resume": True,
                "fixed_risk": 0.01,
                "max_position_cap_pct": 0.30,
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
            "builder_type": "binary_dl_risk_only_rolling",
            "options": {
                "parameter_set": "p3",
                "model_source_id": "A9",
                "p3_variant": "A9",
                "trials_per_fold": 200,
                "resume": True,
                "fixed_risk": 0.01,
                "max_position_cap_pct": 0.30,
                "build_binary_pit": True,
                "binary_pit_resume": True,
                "quiet": False,
            },
        },
    },
}

# =============================================================================
# 4. DL工件來源
# =============================================================================

STRATEGY_DL_SOURCES = {
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
}

# =============================================================================
# 5. 要比較的對象：逐項用enabled開關
# =============================================================================
# 同一param_source／rule_policy只定義一個共用DL-off基準，並可掛多個DL-on模型。
# 各DL-on arm與contrast可獨立開關；只要仍有DL-on啟用，共用DL-off就必須啟用。

STRATEGY_COMPARE_ARMS = {
    "C1": {
        "enabled": False,
        "name": "Full ROOS",
        "description": "原正式基準",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
    },
    "C2": {
        "enabled": False,
        "name": "Full ROOS: TP1-on",
        "description": "Full ROOS參數，runtime開TP1",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "TP1",
        "dl_runtime_mode": "hard-filter",
    },
    "C3": {
        "enabled": True,
        "name": "Min ROOS",
        "description": "rules全關、DL-off環境訓練的Min基準",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
    },
    "C4": {
        "enabled": False,
        "name": "Min ROOS: TP1-on",
        "description": "Min ROOS參數，runtime開TP1",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "TP1",
        "dl_runtime_mode": "hard-filter",
    },
    "C5": {
        "enabled": False,
        "name": "Min-TP1 ROOS",
        "description": "TP1-on環境訓練參數，runtime DL-off",
        "param_source": "min_dl_tp1_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
    },
    "C6": {
        "enabled": False,
        "name": "Min-TP1 ROOS: DL-on",
        "description": "TP1-on環境訓練參數，runtime開其配對TP1",
        "param_source": "min_dl_tp1_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "TP1",
        "dl_runtime_mode": "hard-filter",
    },
    "C7": {
        "enabled": False,
        "name": "Full ROOS: A9-on",
        "description": "Full ROOS參數，runtime開A9",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "hard-filter",
    },
    "C8": {
        "enabled": False,
        "name": "Min ROOS: A9-on",
        "description": "Min ROOS參數，runtime開A9",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "hard-filter",
    },
    "C9": {
        "enabled": False,
        "name": "Min-A9 ROOS",
        "description": "A9-on環境訓練參數，runtime DL-off",
        "param_source": "min_dl_a9_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
    },
    "C10": {
        "enabled": False,
        "name": "Min-A9 ROOS: DL-on",
        "description": "A9-on環境訓練參數，runtime開其配對A9",
        "param_source": "min_dl_a9_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "hard-filter",
    },
    "C11": {
        "enabled": False,
        "name": "Min ROOS: A9 resource-aware",
        "description": "Min ROOS參數；A9只在盤前cash先成瓶頸時以first-improvement改善PASS預留資金",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "resource-aware-binary",
    },
    "C12": {
        "enabled": True,
        "name": "Min ROOS: A9 resource-aware basket",
        "description": "Min ROOS參數；沿用相同cash-binding Gate，每輪評估全部可行PASS promotion並採用最佳改善",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "A9",
        "dl_runtime_mode": "resource-aware-binary-basket",
    },
    "C14": {
        "enabled": False,
        "name": "Min ROOS: Continuous resource-aware",
        "description": "歷史對照；capital-utilization first + MR-11G PASS-only continuous score",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT11G",
        "dl_runtime_mode": "resource-aware-continuous",
    },
    "C15": {
        "enabled": True,
        "name": "Min ROOS: All-event Continuous resource-aware",
        "description": "Min ROOS先維持資本利用；只有cash-binding的DL Selection Mode才使用MR-12A all-event frozen OOS continuous score排序",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12A",
        "dl_runtime_mode": "resource-aware-continuous",
    },
}

# =============================================================================
# 6. 報表差異：逐項用enabled開關
# =============================================================================

STRATEGY_COMPARE_CONTRASTS = {
    "C8-C3": {"enabled": False, "left": "C8", "right": "C3", "description": "Min ROOS下A9 hard-filter效果（既有對照重現）"},
    "C11-C3": {"enabled": False, "left": "C11", "right": "C3", "description": "Min ROOS下A9 resource-aware first-improvement效果"},
    "C12-C3": {"enabled": True, "left": "C12", "right": "C3", "description": "Min ROOS下A9 resource-aware best-improvement效果"},
    "C14-C3": {"enabled": False, "left": "C14", "right": "C3", "description": "Capital-utilization first下MR-11G continuous排序效果"},
    "C14-C12": {"enabled": False, "left": "C14", "right": "C12", "description": "MR-11G Continuous resource-aware相對A9 max-PASS resource-aware效果"},
    "C15-C3": {"enabled": True, "left": "C15", "right": "C3", "description": "Capital-utilization first下MR-12A all-event continuous排序效果"},
    "C15-C12": {"enabled": True, "left": "C15", "right": "C12", "description": "All-event continuous resource-aware相對A9 max-PASS resource-aware效果"},
    "C12-C11": {"enabled": False, "left": "C12", "right": "C11", "description": "Best-improvement相對first-improvement改善"},
    "C11-C8": {"enabled": False, "left": "C11", "right": "C8", "description": "Resource-aware相對A9 hard-filter改善"},
    "C2-C1": {"enabled": False, "left": "C2", "right": "C1", "description": "Full ROOS下TP1 runtime效果"},
    "C7-C1": {"enabled": False, "left": "C7", "right": "C1", "description": "Full ROOS下A9 runtime效果"},
    "C4-C3": {"enabled": False, "left": "C4", "right": "C3", "description": "Min ROOS下TP1 runtime效果"},
    "C6-C5": {"enabled": False, "left": "C6", "right": "C5", "description": "Min-TP1 ROOS下配對DL效果"},
    "C10-C9": {"enabled": False, "left": "C10", "right": "C9", "description": "Min-A9 ROOS下配對DL效果"},
    "C3-C1": {"enabled": False, "left": "C3", "right": "C1", "description": "Min ROOS相對Full ROOS"},
    "C5-C3": {"enabled": False, "left": "C5", "right": "C3", "description": "TP1-aware參數本身效果"},
    "C9-C3": {"enabled": False, "left": "C9", "right": "C3", "description": "A9-aware參數本身效果"},
    "C6-C4": {"enabled": False, "left": "C6", "right": "C4", "description": "TP1-on下參數適應效果"},
    "C10-C8": {"enabled": False, "left": "C10", "right": "C8", "description": "A9-on下參數適應效果"},
    "C6-C1": {"enabled": False, "left": "C6", "right": "C1", "description": "Min-TP1完整方案相對正式基準"},
    "C10-C1": {"enabled": False, "left": "C10", "right": "C1", "description": "Min-A9完整方案相對正式基準"},
}


def _builder(raw) -> StrategyArtifactBuilder | None:
    if raw in (None, {}):
        return None
    return StrategyArtifactBuilder(
        enabled=bool(raw.get("enabled")),
        builder_type=str(raw.get("builder_type") or "").strip(),
        options=dict(raw.get("options") or {}),
    )


def get_strategy_comparison_settings() -> StrategyComparisonSettings:
    preparation = StrategyPreparationPolicy(
        auto_prepare=bool(STRATEGY_COMPARE_PREPARATION.get("auto_prepare")),
        reuse_ready_artifacts=bool(
            STRATEGY_COMPARE_PREPARATION.get("reuse_ready_artifacts")
        ),
        rebuild_stale_artifacts=bool(
            STRATEGY_COMPARE_PREPARATION.get("rebuild_stale_artifacts")
        ),
        resume_parameter_training=bool(
            STRATEGY_COMPARE_PREPARATION.get("resume_parameter_training")
        ),
        require_confirmation=bool(
            STRATEGY_COMPARE_PREPARATION.get("require_confirmation")
        ),
    )
    parameter_sources = {
        source_id: StrategyParameterSource(
            source_id=source_id,
            path_template=raw.get("path_template"),
            description=str(raw.get("description") or "").strip(),
            identity_manifest_path=raw.get("identity_manifest_path"),
            trained_with_dl_id=raw.get("trained_with_dl_id"),
            builder=_builder(raw.get("builder")),
        )
        for source_id, raw in STRATEGY_PARAM_SOURCES.items()
    }
    dl_sources = {
        dl_id: StrategyDLSource(
            dl_id=dl_id,
            filter_id=str(raw.get("filter_id") or "").strip(),
            model_architecture=str(raw.get("model_architecture") or "").strip(),
            experiment_profile=str(raw.get("experiment_profile") or "").strip(),
            threshold=(None if raw.get("threshold") in (None, "") else float(raw.get("threshold"))),
            description=str(raw.get("description") or "").strip(),
            score_source=str(raw.get("score_source") or "canonical_runtime").strip(),
            forward_scores_builder=_builder(raw.get("forward_scores_builder")),
        )
        for dl_id, raw in STRATEGY_DL_SOURCES.items()
    }
    arms = {
        arm_id: StrategyComparisonArm(
            arm_id=arm_id,
            enabled=bool(raw.get("enabled")),
            name=str(raw.get("name") or "").strip(),
            description=str(raw.get("description") or "").strip(),
            param_source=str(raw.get("param_source") or "").strip(),
            rule_policy=str(raw.get("rule_policy") or "").strip(),
            dl_enabled=bool(raw.get("dl_enabled")),
            dl_id=(None if raw.get("dl_id") in (None, "") else str(raw.get("dl_id"))),
            dl_runtime_mode=(
                None
                if raw.get("dl_runtime_mode") in (None, "")
                else str(raw.get("dl_runtime_mode")).strip()
            ),
        )
        for arm_id, raw in STRATEGY_COMPARE_ARMS.items()
    }
    contrasts = {
        contrast_id: StrategyComparisonContrast(
            contrast_id=contrast_id,
            enabled=bool(raw.get("enabled")),
            left=str(raw.get("left") or "").strip(),
            right=str(raw.get("right") or "").strip(),
            description=str(raw.get("description") or "").strip(),
        )
        for contrast_id, raw in STRATEGY_COMPARE_CONTRASTS.items()
    }
    settings = StrategyComparisonSettings(
        schema_version=int(STRATEGY_COMPARE_SCHEMA_VERSION),
        dataset=str(STRATEGY_COMPARE_DATASET).strip(),
        start_date=(None if STRATEGY_COMPARE_START_DATE in (None, "") else str(STRATEGY_COMPARE_START_DATE).strip()),
        end_date=(None if STRATEGY_COMPARE_END_DATE in (None, "") else str(STRATEGY_COMPARE_END_DATE).strip()),
        param_policy=str(STRATEGY_COMPARE_PARAM_POLICY).strip(),
        max_positions=int(STRATEGY_COMPARE_MAX_POSITIONS),
        rotation=str(STRATEGY_COMPARE_ROTATION).strip(),
        output_root=str(STRATEGY_COMPARE_OUTPUT_ROOT).strip(),
        preparation=preparation,
        parameter_sources=parameter_sources,
        dl_sources=dl_sources,
        arms=arms,
        contrasts=contrasts,
    )
    validate_strategy_comparison_settings(settings)
    return settings


__all__ = [
    "STRATEGY_COMPARE_ARMS",
    "STRATEGY_COMPARE_CONTRASTS",
    "STRATEGY_COMPARE_PREPARATION",
    "STRATEGY_DL_SOURCES",
    "STRATEGY_PARAM_SOURCES",
    "get_strategy_comparison_settings",
]
