"""目前正式 Strategy Compare 設定。

本檔只保存目前 Selection PIT／Forward-OOS profiles 需要使用者維護的 active
parameter sources、DL sources、arms 與 contrasts。退役研究定義移至
``config/compatibility/strategy_compare_history.py``，僅於歷史工件解讀／重現時合併使用。
arm／contrast是否啟用仍只由 profile membership 決定。
"""

from __future__ import annotations

from config.breakout_quality import get_breakout_quality_workflow_settings
from config.execution_policy import DEFAULT_FIXED_RISK, DEFAULT_MAX_POSITION_CAP_PCT
from config.compatibility.strategy_compare_history import (
    HISTORICAL_STRATEGY_COMPARE_ARMS,
    HISTORICAL_STRATEGY_COMPARE_CONTRASTS,
    HISTORICAL_STRATEGY_DL_SOURCES,
    HISTORICAL_STRATEGY_PARAM_SOURCES,
)
from config.training_policy import (
    OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
    OPTIMIZER_RANDOM_SEED_DEFAULT,
    OUTER_ROLLING_OOS_HORIZON_MONTHS,
    OUTER_ROLLING_TRAIN_WINDOW_MONTHS,
)
from core.strategy_comparison import (
    StrategyArtifactBuilder,
    StrategyComparisonArm,
    StrategyComparisonContrast,
    StrategyComparisonSettings,
    StrategyMultiSeedRobustnessSettings,
    StrategyDLSource,
    StrategyParameterSource,
    StrategyPreparationPolicy,
    validate_strategy_comparison_settings,
    validate_strategy_multi_seed_robustness_settings,
)

STRATEGY_COMPARE_SCHEMA_VERSION = 20

# =============================================================================
# 1. 常用設定
#    一般 Strategy Compare / robustness 實驗通常只需修改本區。
#    Dataset / param policy / max positions / rotation 不在此複製；它們直接
#    繼承 config/breakout_quality.py 的 BreakoutQualityWorkflowSettings SSOT。
# =============================================================================

STRATEGY_COMPARE_DEFAULT_PROFILE = "forward_oos"
STRATEGY_COMPARE_DEFAULT_ROBUSTNESS_PROFILE = "forward_oos"
STRATEGY_COMPARE_ROBUSTNESS_SEED_COUNT = 8
STRATEGY_COMPARE_ROBUSTNESS_SEED_GENERATOR_SEED = 20260810
STRATEGY_COMPARE_ROBUSTNESS_GPU_TRAIN_WORKERS = 2
STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS = 1
STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED = True
STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE = "compact"
STRATEGY_COMPARE_ROBUSTNESS_PROGRESS_INTERVAL_SECONDS = 60.0
STRATEGY_COMPARE_ROBUSTNESS_YEARLY_REPORT = True
STRATEGY_COMPARE_ROBUSTNESS_KEEP_CHECKPOINTS = False
STRATEGY_COMPARE_ROBUSTNESS_KEEP_SCORES = False
STRATEGY_COMPARE_ROBUSTNESS_KEEP_REPLAY_DETAILS = False

# Current Strategy Compare核心比較名稱的單一真理。
# Selection PIT／Forward-OOS由profile頁首區分，不把研究階段或固定selector語意塞進arm顯示名稱。
STRATEGY_COMPARE_DISPLAY_FULL_ROOS = "Full ROOS"
STRATEGY_COMPARE_DISPLAY_MIN_ROOS = "Min ROOS"
STRATEGY_COMPARE_DISPLAY_MIN_MR12B = "Min MR-12B"
STRATEGY_COMPARE_DISPLAY_MIN_MR13A = "Min MR-13A"

# Strategy Compare以研究階段profile隔離設定與輸出；App只顯示泛化階段名稱，
# arms／contrasts／period／output namespace全部由本檔驅動。
STRATEGY_COMPARE_PROFILES = {
    "selection_pit": {
        "label": "Selection PIT 策略比較",
        "description": "2014～2020 point-in-time策略轉化Gate；核心比較固定為Full／Min baseline與Min MR-12B／MR-13A。",
        "display_alignment_group": "core_strategy_compare",
        "start_date": "2014-01-01",
        "end_date": "2020-12-31",
        "output_root": "outputs/strategy_compare/selection_pit",
        "reuse_output_roots": ("outputs/strategy_compare",),
        "arm_ids": ("C32", "C23", "C25", "C28"),
        "contrast_ids": (
            "C32-C23",
            "C25-C23", "C28-C23", "C28-C25",
        ),
    },
    "forward_oos": {
        "label": "Forward-OOS 策略比較",
        "description": "2021+ frozen Forward-OOS策略Gate；核心比較固定為Full／Min baseline與Min MR-12B／MR-13A。",
        "display_alignment_group": "core_strategy_compare",
        "start_date": None,
        "end_date": None,
        "output_root": "outputs/strategy_compare/forward_oos",
        "reuse_output_roots": ("outputs/strategy_compare",),
        "arm_ids": ("C1", "C3", "C20", "C29"),
        "contrast_ids": (
            "C1-C3",
            "C20-C3", "C29-C3", "C29-C20",
        ),
    },
}

# Multiple-seed robustness各研究階段以獨立config profile呈現於正式選單。
# stochastic/fixed比較對象不在此重列arm ID，而由對應Strategy Compare profile的
# enabled arms + robustness_role動態解析。
STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES = {
    "selection_pit": {
        "label": "Selection PIT Multi-seed robustness",
        "enabled": True,
        "profile_id": "selection_pit",
        "seed_count": STRATEGY_COMPARE_ROBUSTNESS_SEED_COUNT,
        "seed_generator_seed": STRATEGY_COMPARE_ROBUSTNESS_SEED_GENERATOR_SEED,
        "gpu_train_workers": STRATEGY_COMPARE_ROBUSTNESS_GPU_TRAIN_WORKERS,
        "cpu_replay_workers": STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS,
        "reuse_completed": STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED,
        "console_mode": STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE,
        "progress_interval_seconds": STRATEGY_COMPARE_ROBUSTNESS_PROGRESS_INTERVAL_SECONDS,
        "yearly_report": STRATEGY_COMPARE_ROBUSTNESS_YEARLY_REPORT,
        "keep_checkpoints": STRATEGY_COMPARE_ROBUSTNESS_KEEP_CHECKPOINTS,
        "keep_scores": STRATEGY_COMPARE_ROBUSTNESS_KEEP_SCORES,
        "keep_replay_details": STRATEGY_COMPARE_ROBUSTNESS_KEEP_REPLAY_DETAILS,
        "romd_reference_baselines": {
            "min": {"param_source": "selection_min_roos", "rule_policy": "all_off"},
            "full": {"param_source": "selection_full_roos", "rule_policy": "formal"},
        },
        "output_root": "outputs/strategy_compare/robustness/selection_pit",
        "model_work_root": "models/research/breakout_quality/strategy_compare/multi_seed_robustness/selection_pit",
    },
    "forward_oos": {
        "label": "Forward-OOS Multi-seed robustness",
        "enabled": True,
        "profile_id": "forward_oos",
        "seed_count": STRATEGY_COMPARE_ROBUSTNESS_SEED_COUNT,
        "seed_generator_seed": STRATEGY_COMPARE_ROBUSTNESS_SEED_GENERATOR_SEED,
        "gpu_train_workers": STRATEGY_COMPARE_ROBUSTNESS_GPU_TRAIN_WORKERS,
        "cpu_replay_workers": STRATEGY_COMPARE_ROBUSTNESS_CPU_REPLAY_WORKERS,
        "reuse_completed": STRATEGY_COMPARE_ROBUSTNESS_REUSE_COMPLETED,
        "console_mode": STRATEGY_COMPARE_ROBUSTNESS_CONSOLE_MODE,
        "progress_interval_seconds": STRATEGY_COMPARE_ROBUSTNESS_PROGRESS_INTERVAL_SECONDS,
        "yearly_report": STRATEGY_COMPARE_ROBUSTNESS_YEARLY_REPORT,
        "keep_checkpoints": STRATEGY_COMPARE_ROBUSTNESS_KEEP_CHECKPOINTS,
        "keep_scores": STRATEGY_COMPARE_ROBUSTNESS_KEEP_SCORES,
        "keep_replay_details": STRATEGY_COMPARE_ROBUSTNESS_KEEP_REPLAY_DETAILS,
        "romd_reference_baselines": {
            "min": {"param_source": "min_roos", "rule_policy": "all_off"},
            "full": {"param_source": "full_roos", "rule_policy": "formal"},
        },
        "output_root": "outputs/strategy_compare/robustness",
        "model_work_root": "models/research/breakout_quality/strategy_compare/multi_seed_robustness",
    }
}


# =============================================================================
# 2. 前置工件政策
# =============================================================================

STRATEGY_COMPARE_PREPARATION = {
    "auto_prepare": True,
    "reuse_ready_artifacts": True,
    "rebuild_stale_artifacts": True,
    "resume_parameter_training": True,
    "require_confirmation": True,
    # 已完成且replay identity完全相同的pair直接重用歷史正式結果。
    "reuse_completed_results": True,
    # 同一param_source/rule_policy的新pair只執行一次DL-off baseline。
    "reuse_shared_baseline": True,
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

# =============================================================================
# 4. DL工件來源
# =============================================================================

STRATEGY_DL_SOURCES = {
    "CONT12B": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_all_event_pairwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": "MR-12B all-event no-time pairwise ranker frozen OOS score；只允許受控strategy research replay",
        "forward_scores_builder": None,
    },
    "CONT13A": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_no_time_pairwise",
        "threshold": None,
        "score_source": "continuous_ranker_oos",
        "description": (
            "MR-13A Daily Universal frozen Forward-OOS continuous score；"
            "每個盤前決策使用最新已完成交易日資訊，供frozen strategy Gate"
        ),
        "forward_scores_builder": None,
    },
    "CONT12B_PIT": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_all_event_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": (
            "MR-12B Selection point-in-time continuous score；"
            "只供2014～2020無前視策略經濟驗證"
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
    "CONT13A_PIT": {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "daily_universal_no_time_pairwise",
        "threshold": None,
        "score_source": "selection_point_in_time",
        "description": (
            "MR-13A Daily Universal Selection point-in-time score；"
            "每個盤前決策只使用最新已完成交易日資訊，供2014～2020無前視策略經濟驗證"
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

# =============================================================================
# 5. Strategy arm definitions
# =============================================================================
# Arm 是否啟用只由 STRATEGY_COMPARE_PROFILES[*]["arm_ids"] 決定；
# arm definition 本身不再保存第二份 enabled 狀態。

STRATEGY_COMPARE_ARMS = {
    "C1": {
        "name": STRATEGY_COMPARE_DISPLAY_FULL_ROOS,
        "description": "Full optimizer rolling active params；DL-off baseline",
        "param_source": "full_roos",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
    "C3": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_ROOS,
        "description": "只搜尋high_len＋4個ATR；rules全關；DL-off baseline",
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
    "C20": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR12B,
        "description": (
            "與C18使用完全相同K/R0、feasible-ascent與basket內Min ROOS執行順序；"
            "唯一模型差異為DL source改成MR-12B pairwise ranker"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12B",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
        "robustness_role": "stochastic",
    },
    "C23": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_ROOS,
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
    "C25": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR12B,
        "description": (
            "與C23使用完全相同historical P2 Min ROOS params；"
            "使用MR-12B Selection PIT score並完全沿用C18 feasible-ascent selector"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT12B_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
        "robustness_role": "stochastic",
    },
    "C28": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13A,
        "description": (
            "與C25使用完全相同historical P2 Min ROOS params、K/R0與feasible-ascent selector；"
            "唯一DL差異為score source改成MR-13A Daily Universal Selection PIT，"
            "盤前每日依最新已完成交易日score重排"
        ),
        "param_source": "selection_min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13A_PIT",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
        "robustness_role": "stochastic",
    },
    "C29": {
        "name": STRATEGY_COMPARE_DISPLAY_MIN_MR13A,
        "description": (
            "與C20使用相同current Min ROOS、all-off rules與frozen feasible-ascent；"
            "唯一DL source差異為MR-13A Daily Universal Forward-OOS score"
        ),
        "param_source": "min_roos",
        "rule_policy": "all_off",
        "dl_enabled": True,
        "dl_id": "CONT13A",
        "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent",
        "robustness_role": "stochastic",
    },
    "C32": {
        "name": STRATEGY_COMPARE_DISPLAY_FULL_ROOS,
        "description": "2014～2020 historical Full ROOS active params；formal rules；DL-off共同baseline",
        "param_source": "selection_full_roos",
        "rule_policy": "formal",
        "dl_enabled": False,
        "dl_id": None,
        "dl_runtime_mode": None,
        "robustness_role": "fixed_baseline",
    },
}

# =============================================================================
# 6. Contrast definitions
# =============================================================================
# Contrast 是否啟用只由 STRATEGY_COMPARE_PROFILES[*]["contrast_ids"] 決定。

STRATEGY_COMPARE_CONTRASTS = {
    "C20-C3": {"left": "C20", "right": "C3", "description": "current Min ROOS下MR-12B feasible-ascent相對DL-off baseline的Forward-OOS策略效果"},
    "C25-C23": {"left": "C25", "right": "C23", "description": "Selection PIT下固定historical Min ROOS與C18 selector，MR-12B PIT ranking相對DL-off baseline的經濟效果"},
    "C28-C25": {"left": "C28", "right": "C25", "description": "固定historical Min ROOS與feasible-ascent selector，MR-13A daily PIT相對MR-12B event PIT的純DL source效果"},
    "C28-C23": {"left": "C28", "right": "C23", "description": "Selection PIT下MR-13A daily feasible-ascent相對DL-off historical Min ROOS baseline的策略經濟效果"},
    "C29-C3": {"left": "C29", "right": "C3", "description": "current Min ROOS下MR-13A daily feasible-ascent相對DL-off baseline的Forward-OOS策略效果"},
    "C29-C20": {"left": "C29", "right": "C20", "description": "固定current Min ROOS與feasible-ascent，MR-13A daily相對MR-12B event source的純DL Forward-OOS效果"},
    "C1-C3": {"left": "C1", "right": "C3", "description": "Full ROOS相對current Min ROOS的完整策略體系差異；不是單一參數效果"},
    "C32-C23": {"left": "C32", "right": "C23", "description": "Selection Full ROOS相對Selection Min ROOS的完整策略體系差異；不是單一參數效果"},
}


def _merge_compatibility_catalog(
    active: dict[str, dict],
    historical: dict[str, dict],
    *,
    catalog_name: str,
) -> dict[str, dict]:
    overlap = sorted(set(active).intersection(historical))
    if overlap:
        raise ValueError(
            f"Strategy Compare active/historical {catalog_name} ID重複: {overlap}"
        )
    return {**active, **historical}


def _compatibility_catalogs() -> tuple[
    dict[str, dict],
    dict[str, dict],
    dict[str, dict],
    dict[str, dict],
]:
    """Return runtime catalogs including read-only historical compatibility entries."""

    return (
        _merge_compatibility_catalog(
            STRATEGY_PARAM_SOURCES,
            HISTORICAL_STRATEGY_PARAM_SOURCES,
            catalog_name="parameter source",
        ),
        _merge_compatibility_catalog(
            STRATEGY_DL_SOURCES,
            HISTORICAL_STRATEGY_DL_SOURCES,
            catalog_name="DL source",
        ),
        _merge_compatibility_catalog(
            STRATEGY_COMPARE_ARMS,
            HISTORICAL_STRATEGY_COMPARE_ARMS,
            catalog_name="arm",
        ),
        _merge_compatibility_catalog(
            STRATEGY_COMPARE_CONTRASTS,
            HISTORICAL_STRATEGY_COMPARE_CONTRASTS,
            catalog_name="contrast",
        ),
    )


def _builder(raw) -> StrategyArtifactBuilder | None:
    if raw in (None, {}):
        return None
    return StrategyArtifactBuilder(
        enabled=bool(raw.get("enabled")),
        builder_type=str(raw.get("builder_type") or "").strip(),
        options=dict(raw.get("options") or {}),
    )


def get_strategy_multi_seed_robustness_profiles() -> tuple[dict[str, str], ...]:
    return tuple(
        {"robustness_id": str(robustness_id), "label": str(raw.get("label") or robustness_id)}
        for robustness_id, raw in STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES.items()
        if bool(raw.get("enabled", True))
    )


def get_strategy_multi_seed_robustness_settings(
    robustness_id: str | None = None,
) -> StrategyMultiSeedRobustnessSettings:
    selected_id = str(robustness_id or STRATEGY_COMPARE_DEFAULT_ROBUSTNESS_PROFILE).strip()
    if selected_id not in STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES:
        raise ValueError(f"不存在的multi-seed robustness profile: {selected_id}")
    raw = dict(STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES[selected_id])
    settings = StrategyMultiSeedRobustnessSettings(
        robustness_id=selected_id,
        label=str(raw.get("label") or selected_id).strip(),
        enabled=bool(raw.get("enabled", True)),
        profile_id=str(raw.get("profile_id") or "").strip(),
        seed_count=int(raw.get("seed_count", 0) or 0),
        seed_generator_seed=int(raw.get("seed_generator_seed", 0) or 0),
        gpu_train_workers=int(raw.get("gpu_train_workers", STRATEGY_COMPARE_ROBUSTNESS_GPU_TRAIN_WORKERS)),
        cpu_replay_workers=int(raw.get("cpu_replay_workers", 0) or 0),
        reuse_completed=bool(raw.get("reuse_completed", True)),
        console_mode=str(raw.get("console_mode") or "compact").strip(),
        progress_interval_seconds=float(raw.get("progress_interval_seconds", 60.0) or 60.0),
        yearly_report=bool(raw.get("yearly_report", True)),
        keep_checkpoints=bool(raw.get("keep_checkpoints", False)),
        keep_scores=bool(raw.get("keep_scores", False)),
        keep_replay_details=bool(raw.get("keep_replay_details", False)),
        romd_reference_baselines={
            str(key).strip(): {
                "param_source": str(dict(value or {}).get("param_source") or "").strip(),
                "rule_policy": str(dict(value or {}).get("rule_policy") or "").strip(),
            }
            for key, value in dict(raw.get("romd_reference_baselines") or {}).items()
        },
        output_root=str(raw.get("output_root") or "").strip(),
        model_work_root=str(raw.get("model_work_root") or "").strip(),
    )
    validate_strategy_multi_seed_robustness_settings(settings)
    if settings.profile_id not in STRATEGY_COMPARE_PROFILES:
        raise ValueError(
            f"multi-seed robustness引用不存在的Strategy Compare profile: {settings.profile_id}"
        )
    profile_settings = get_strategy_comparison_settings(settings.profile_id)
    fixed = [arm for arm in profile_settings.enabled_arms if arm.robustness_role == "fixed_baseline"]
    stochastic = [arm for arm in profile_settings.enabled_arms if arm.robustness_role == "stochastic"]
    if not fixed:
        raise ValueError("multi-seed robustness至少需要一個fixed_baseline arm")
    if not stochastic:
        raise ValueError("multi-seed robustness至少需要一個stochastic arm")
    score_sources = {
        profile_settings.dl_sources[str(arm.dl_id)].score_source
        for arm in stochastic if arm.dl_id
    }
    expected_score_source = (
        "selection_point_in_time" if settings.profile_id == "selection_pit"
        else "continuous_ranker_oos"
    )
    if score_sources != {expected_score_source}:
        raise ValueError(
            "multi-seed stochastic arms的score source與robustness階段不一致: "
            f"expected={expected_score_source}, actual={sorted(score_sources)}"
        )
    for reference_key, spec in settings.romd_reference_baselines.items():
        matches = [
            arm for arm in fixed
            if arm.param_source == spec["param_source"]
            and arm.rule_policy == spec["rule_policy"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "multi-seed RoMD reference必須唯一對應一個fixed baseline: "
                f"reference={reference_key}, param_source={spec['param_source']}, "
                f"rule_policy={spec['rule_policy']}, matches={len(matches)}"
            )
    return settings


def get_strategy_comparison_profiles() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "profile_id": str(profile_id),
            "label": str(raw["label"]),
            "description": str(raw.get("description") or ""),
        }
        for profile_id, raw in STRATEGY_COMPARE_PROFILES.items()
    )


def get_strategy_comparison_settings(profile_id: str | None = None) -> StrategyComparisonSettings:
    selected_profile_id = str(profile_id or STRATEGY_COMPARE_DEFAULT_PROFILE).strip()
    if selected_profile_id not in STRATEGY_COMPARE_PROFILES:
        raise ValueError(f"未知Strategy Compare profile: {selected_profile_id}")
    profile = dict(STRATEGY_COMPARE_PROFILES[selected_profile_id])
    profile_arm_ids = tuple(str(value) for value in profile.get("arm_ids", ()))
    profile_contrast_ids = tuple(str(value) for value in profile.get("contrast_ids", ()))
    missing_arms = [arm_id for arm_id in profile_arm_ids if arm_id not in STRATEGY_COMPARE_ARMS]
    missing_contrasts = [
        contrast_id
        for contrast_id in profile_contrast_ids
        if contrast_id not in STRATEGY_COMPARE_CONTRASTS
    ]
    if missing_arms or missing_contrasts:
        raise ValueError(
            "Strategy Compare profile引用不存在的設定: "
            f"arms={missing_arms or '-'}, contrasts={missing_contrasts or '-'}"
        )

    (
        parameter_catalog,
        dl_catalog,
        arm_catalog,
        contrast_catalog,
    ) = _compatibility_catalogs()

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
        reuse_completed_results=bool(
            STRATEGY_COMPARE_PREPARATION.get("reuse_completed_results", True)
        ),
        reuse_shared_baseline=bool(
            STRATEGY_COMPARE_PREPARATION.get("reuse_shared_baseline", True)
        ),
    )
    parameter_sources = {
        source_id: StrategyParameterSource(
            source_id=source_id,
            path_template=raw.get("path_template"),
            description=str(raw.get("description") or "").strip(),
            identity_manifest_path=raw.get("identity_manifest_path"),
            trained_with_dl_id=raw.get("trained_with_dl_id"),
            artifact_contract=(
                None
                if raw.get("artifact_contract") in (None, {})
                else dict(raw.get("artifact_contract") or {})
            ),
            builder=_builder(raw.get("builder")),
        )
        for source_id, raw in parameter_catalog.items()
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
        for dl_id, raw in dl_catalog.items()
    }
    ordered_arm_ids = tuple(dict.fromkeys((*profile_arm_ids, *arm_catalog.keys())))
    arms = {
        arm_id: StrategyComparisonArm(
            arm_id=arm_id,
            enabled=arm_id in profile_arm_ids,
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
            dl_runtime_options=(
                None
                if raw.get("dl_runtime_options") in (None, {})
                else dict(raw.get("dl_runtime_options") or {})
            ),
            robustness_role=str(raw.get("robustness_role") or "off").strip(),
        )
        for arm_id in ordered_arm_ids
        for raw in (arm_catalog[arm_id],)
    }
    ordered_contrast_ids = tuple(dict.fromkeys((*profile_contrast_ids, *contrast_catalog.keys())))
    contrasts = {
        contrast_id: StrategyComparisonContrast(
            contrast_id=contrast_id,
            enabled=contrast_id in profile_contrast_ids,
            left=str(raw.get("left") or "").strip(),
            right=str(raw.get("right") or "").strip(),
            description=str(raw.get("description") or "").strip(),
        )
        for contrast_id in ordered_contrast_ids
        for raw in (contrast_catalog[contrast_id],)
    }
    workflow_settings = get_breakout_quality_workflow_settings()
    settings = StrategyComparisonSettings(
        schema_version=int(STRATEGY_COMPARE_SCHEMA_VERSION),
        profile_id=selected_profile_id,
        profile_label=str(profile["label"]),
        dataset=str(workflow_settings.strategy_dataset).strip(),
        start_date=(
            None
            if profile.get("start_date") in (None, "")
            else str(profile.get("start_date")).strip()
        ),
        end_date=(
            None
            if profile.get("end_date") in (None, "")
            else str(profile.get("end_date")).strip()
        ),
        param_policy=str(workflow_settings.strategy_param_policy).strip(),
        max_positions=int(workflow_settings.strategy_max_positions),
        rotation=str(workflow_settings.strategy_rotation).strip(),
        output_root=str(profile["output_root"]).strip(),
        reuse_output_roots=tuple(
            str(value).strip()
            for value in tuple(profile.get("reuse_output_roots") or ())
            if str(value).strip()
        ),
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
    "STRATEGY_COMPARE_PROFILES",
    "STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES",
    "STRATEGY_DL_SOURCES",
    "STRATEGY_PARAM_SOURCES",
    "get_strategy_comparison_profiles",
    "get_strategy_multi_seed_robustness_profiles",
    "get_strategy_multi_seed_robustness_settings",
    "get_strategy_comparison_settings",
]
