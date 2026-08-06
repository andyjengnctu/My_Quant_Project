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

STRATEGY_COMPARE_SCHEMA_VERSION = 2

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
            "models/research/breakout_quality/trade_path_label/"
            "a2_teacher_params/p2_dl_off_trained/active_params/{param_filename}"
        ),
        "description": "rule-based filters全關、DL-off訓練的Min ROOS",
        "identity_manifest_path": None,
        "trained_with_dl_id": None,
        "builder": None,
    },
    "min_dl_tp1_roos": {
        "path_template": (
            "models/research/breakout_quality/binary_dl_filter_param_adaptation/"
            "risk_only_rolling/p3_dl_on_trained/active_params/{param_filename}"
        ),
        "description": "rule-based filters全關、指定DL模型訓練的Min-DL ROOS",
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
}

# =============================================================================
# 5. 要比較的對象：逐項用enabled開關
# =============================================================================
# 同一param_source／rule_policy的DL-off與DL-on是canonical controlled pair，
# 必須一起開啟或一起關閉；不同pair及各contrast可獨立開關。

STRATEGY_COMPARE_ARMS = {
    "C1": {
        "enabled": True,
        "name": "Full ROOS × DL-off",
        "description": "原正式基準",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
    },
    "C2": {
        "enabled": True,
        "name": "Full ROOS × DL-TP1",
        "description": "DL套用於原正式參數",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": True,
        "dl_id": "TP1",
    },
    "C3": {
        "enabled": True,
        "name": "Min ROOS × DL-off",
        "description": "最小規則基準",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
    },
    "C4": {
        "enabled": True,
        "name": "Min ROOS × DL-TP1",
        "description": "純DL執行效果",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "TP1",
    },
    "C5": {
        "enabled": True,
        "name": "Min-DL-TP1 ROOS × DL-off",
        "description": "單看DL-aware參數效果",
        "param_source": "min_dl_tp1_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
    },
    "C6": {
        "enabled": True,
        "name": "Min-DL-TP1 ROOS × DL-TP1",
        "description": "DL模型與對應參數完整組合",
        "param_source": "min_dl_tp1_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "TP1",
    },
}

# =============================================================================
# 6. 報表差異：逐項用enabled開關
# =============================================================================

STRATEGY_COMPARE_CONTRASTS = {
    "C2-C1": {"enabled": True, "left": "C2", "right": "C1", "description": "Full ROOS下的DL效果"},
    "C4-C3": {"enabled": True, "left": "C4", "right": "C3", "description": "Min ROOS下的DL效果"},
    "C6-C5": {"enabled": True, "left": "C6", "right": "C5", "description": "Min-DL ROOS下的DL效果"},
    "C3-C1": {"enabled": True, "left": "C3", "right": "C1", "description": "最小規則相對正式基準"},
    "C5-C3": {"enabled": True, "left": "C5", "right": "C3", "description": "DL-aware參數本身效果"},
    "C6-C4": {"enabled": True, "left": "C6", "right": "C4", "description": "完整DL-aware組合的額外效果"},
    "C6-C1": {"enabled": True, "left": "C6", "right": "C1", "description": "完整方案相對正式基準"},
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
            threshold=float(raw.get("threshold")),
            description=str(raw.get("description") or "").strip(),
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
