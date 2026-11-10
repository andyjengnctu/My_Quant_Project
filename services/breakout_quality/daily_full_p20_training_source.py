"""Seed-scoped Daily trade-path teachers, driven by one declared capability.

Canonical Optimizer search/seed/stitch semantics stay with the Optimizer.
This service validates and composes its signed historical/current outputs.
"""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
from functools import partial
from pathlib import Path
import pandas as pd
from config.execution_policy import (
    DEFAULT_FIXED_RISK,
    DEFAULT_MAX_POSITION_CAP_PCT,
    DEFAULT_PORTFOLIO_MAX_POSITIONS,
    DEFAULT_PORTFOLIO_ROTATION,
)
from core.file_integrity import canonical_json_sha256, load_json_strict
from core.portfolio_param_runtime import build_portfolio_params_signature
from core.strategy_param_artifacts import compute_strategy_param_file_sha256
from filters.breakout_quality.daily_trade40_target import (
    build_formal_min_execution_schedule,
    resolve_formal_min_param_records,
)
from filters.breakout_quality.strategy_compare_sources import (
    PARAM_POLICY_BASE_FINALIST_BEST,
    load_strategy_param_evaluation_view,
)
from filters.breakout_quality.trade_path_label import (
    TRADE_PATH_BASE_FILTER_ID,
    TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
    TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
)
from services.breakout_quality.formal_mintrade40_pit_source import (
    FormalMinTrade40PITBlocked,
    FormalMinTrade40PITSource,
)
from services.breakout_quality.formal_mintrade40_training_source import (
    _current_p16_seed_artifact,
    inspect_formal_mintrade40_p16_teacher,
)
from services.optimizer.strategy_param_service import _canonical_schedule_build_contract
from services.optimizer.optimizer_workspace_recovery import (
    nested_optimizer_workspace_candidate,
    inspect_nested_optimizer_workspace,
    adopt_nested_optimizer_workspace,
    verify_optimizer_runtime_preflight,
)
from services.optimizer.strategy_param_training import (
    prepare_selection_historical_p2_params,
    _build_min_roos_schedule_contract,
    build_min_roos_fold_overrides,
    _runtime_contract,
)
from strategies.daily.search_space import resolve_daily_search_profile
from importlib import import_module
from tempfile import TemporaryDirectory
from services.optimizer.strategy_param_training import (
    _validate_min_roos_params, _min_roos_adaptation_contract,
)


class DailyTradePathTeacher:
    def __init__(self, spec):
        self.spec = spec

    _canonical_schedule_build_contract = staticmethod(
        _canonical_schedule_build_contract
    )
    _current_p16_seed_artifact = staticmethod(_current_p16_seed_artifact)
    build_formal_min_execution_schedule = staticmethod(
        build_formal_min_execution_schedule
    )
    compute_strategy_param_file_sha256 = staticmethod(
        compute_strategy_param_file_sha256
    )
    inspect_formal_mintrade40_p16_teacher = staticmethod(
        inspect_formal_mintrade40_p16_teacher
    )
    load_strategy_param_evaluation_view = staticmethod(
        load_strategy_param_evaluation_view
    )
    prepare_selection_historical_p2_params = staticmethod(
        prepare_selection_historical_p2_params
    )
    resolve_formal_min_param_records = staticmethod(resolve_formal_min_param_records)
    HISTORY_ID = property(lambda self: self.spec["historical_parameter_set"])
    PROFILE = property(lambda self: self.spec["search_profile"])
    BASELINE_ARM = property(lambda self: self.spec["baseline_arm"])
    SOURCE_ID = property(lambda self: self.spec["source_id"])
    ENTRY_POLICY = property(lambda self: self.spec["entry_policy"])

    def historical_params_path(self, project_root: str | Path, seed: int) -> Path:
        if type(seed) is not int or seed <= 0:
            raise FormalMinTrade40PITBlocked(
                "Daily requires an explicit positive teacher seed"
            )
        return (
            Path(project_root).resolve()
            / "outputs"
            / "optimizer"
            / self.spec["history_namespace"]
            / f"seed_{seed}"
            / "p2_dl_off_trained"
            / "active_params"
            / "roos_base_best.json"
        )

    def _expected_historical_preflight(
        self, root: Path, *, validate_path=None
    ) -> dict[str, object]:
        """Derive the expected signed preflight with the canonical Optimizer owner.

        Merely checking a receipt against its own claimed hash is insufficient:
        neither a changed Daily search protocol nor a changed source dataset may
        silently turn a stale historical teacher into a current FIT source.
        """
        policy = self._canonical_schedule_build_contract(
            family=self.spec["param_family"]
        )
        search = resolve_daily_search_profile(self.spec["search_profile"])
        schedule = _build_min_roos_schedule_contract(
            first_oos_date=TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
            last_oos_date=TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
            train_window_months=int(policy["train_window_months"]),
            oos_months=int(policy["oos_horizon_months"]),
        )
        reference = {
            **schedule["reference_payload"],
            "search_fields": list(search["fields"]),
            "entry_strategy": "Daily",
            "entry_universe": "unscored_all_adjacent_quote_stockdays_v1",
            "breakout_required": False,
            "daily_execution_contract": search["execution_contract"],
        }
        schedule = {
            **schedule,
            "reference_type": "canonical_daily_min_roos_fixed_defaults",
            "reference_payload": reference,
            "sha256": canonical_json_sha256(reference),
        }
        args = SimpleNamespace(
            dataset="full",
            trials_per_fold=int(policy["trials_per_fold"]),
            strategy_id="daily",
            daily_search_profile=self.spec["search_profile"],
            filter_id=TRADE_PATH_BASE_FILTER_ID,
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            canonical_strategy_param_family="",
            optimizer_search_protocol="",
        )
        overrides = build_min_roos_fold_overrides(
            baseline_contract=schedule, args=args, training_dl_enabled=False
        )
        if validate_path is not None:
            with TemporaryDirectory(prefix="daily-teacher-verify-") as temporary:
                candidate = Path(temporary) / Path(validate_path).name
                candidate.write_bytes(Path(validate_path).read_bytes())
                original = load_json_strict(candidate)
                verified = _validate_min_roos_params(
                    path=candidate,
                    baseline_contract=schedule,
                    fold_overrides=overrides,
                    args=args,
                    training_dl_enabled=False,
                    arm_id=self.spec["historical_parameter_set"],
                )
                if verified != original:
                    raise FormalMinTrade40PITBlocked(
                        "Historical Daily teacher requires canonical metadata migration"
                    )
        return _runtime_contract(
            root=root,
            args=args,
            baseline_contract=schedule,
            fold_overrides=overrides,
            model_artifact=None,
            arm_id=self.spec["historical_parameter_set"],
            training_dl_enabled=False,
            binary_pit=None,
        )

    def _validated_records(
        self, path: Path, *, historical: bool, project_root: str | Path, seed: int
    ):
        if not path.is_file():
            raise FormalMinTrade40PITBlocked(
                f"Missing signed Daily parameter artifact: {path.name}"
            )
        payload = load_json_strict(path)
        if not isinstance(payload, dict):
            raise FormalMinTrade40PITBlocked(
                "Daily parameter source must be a JSON object"
            )
        if historical:
            adaptation = dict(payload.get("breakout_quality_param_adaptation") or {})
            search = resolve_daily_search_profile(self.spec["search_profile"])
            adaptation_contract = _min_roos_adaptation_contract(
                arm_id=self.HISTORY_ID, training_dl_enabled=False,
                strategy_id="daily", daily_search_profile=self.PROFILE,
            )
            if (
                adaptation.get("mode") != "min_roos_training"
                or adaptation.get("parameter_set")
                != self.spec["historical_parameter_set"]
                or adaptation.get("training_dl_enabled") is not False
                or (adaptation.get("entry_strategy") != "Daily")
                or (
                    adaptation.get("fixed_rule_contract")
                    != adaptation_contract["fixed_rule_contract"]
                )
                or (
                    adaptation.get("daily_execution_contract")
                    != search["execution_contract"]
                )
                or (
                    tuple(adaptation.get("search_fields") or ())
                    != tuple(search["fields"])
                )
            ):
                raise FormalMinTrade40PITBlocked(
                    "Historical teacher differs from the declared Daily search profile"
                )
            expected = self._expected_historical_preflight(
                Path(project_root).resolve(),
                validate_path=path if self.spec.get("validate_history") else None,
            )
            preflight = verify_optimizer_runtime_preflight(
                path.parent.parent / "rolling_preflight.json",
                expected_runtime_identity_sha256=expected["runtime_identity_sha256"],
            )
            summary = load_json_strict(
                path.parent.parent / "rolling_optimizer_summary.json"
            )
            if not isinstance(summary, dict):
                raise FormalMinTrade40PITBlocked(
                    "Missing signed historical Daily Optimizer evidence"
                )
            if (
                preflight.get("runtime_identity_sha256")
                != summary.get("runtime_identity_sha256")
                or preflight.get("runtime_identity_sha256")
                != expected["runtime_identity_sha256"]
                or summary.get("status") != "COMPLETED"
                or (preflight.get("arm_id") != self.spec["historical_parameter_set"])
                or (preflight.get("training_dl_enabled") is not False)
            ):
                raise FormalMinTrade40PITBlocked(
                    "Historical Daily provenance/preflight mismatch"
                )
        records = self.resolve_formal_min_param_records(
            "rolling_active_param_ensemble", payload
        )
        self.validate_teacher_records(records, seed=seed)
        return (payload, records)

    def validate_teacher_records(self, records, *, seed: int):
        """Verify the same-seed, single-member Daily teacher before FIT or Oracle."""
        if type(seed) is not int or seed <= 0 or (not records):
            raise FormalMinTrade40PITBlocked(
                "Daily requires nonempty records and a positive teacher seed"
            )
        for row in records:
            members = tuple(row.get("members") or ())
            if len(members) != 1 or members[0].get("seed") != seed:
                raise FormalMinTrade40PITBlocked(
                    "Daily parameter teacher seed/member mismatch"
                )
            params = row["params_obj"]
            search = resolve_daily_search_profile(self.spec["search_profile"])
            if any(
                getattr(params, key, None) is not value
                for key, value in search.get("fixed_switches", {}).items()
            ):
                raise FormalMinTrade40PITBlocked(
                    "Daily teacher fixed switches differ from declared search profile"
                )
            if any(
                bool(getattr(params, key, False))
                for key in self.spec.get("disabled_filters", ())
            ):
                raise FormalMinTrade40PITBlocked(
                    "Daily teacher contains an undeclared execution filter"
                )
            if (
                bool(params.use_breakout_buy)
                or bool(params.use_breakout_quality_filter)
                or bool(params.use_breakout_quality_ranking)
                or (build_portfolio_params_signature(params) != row["params_signature"])
            ):
                raise FormalMinTrade40PITBlocked(
                    "Daily teacher changed declared execution/score semantics"
                )

    def inspect_history(
        self, *, project_root: str | Path, seed: int
    ) -> dict[str, object]:
        root = Path(project_root).resolve()
        path = self.historical_params_path(root, seed)
        try:
            nested_candidate = nested_optimizer_workspace_candidate(path.parent.parent)
        except ValueError as exc:
            return {
                "path": path,
                "action": "BLOCKED",
                "reason": f"Daily nested history invalid: {exc}",
            }
        if path.is_file():
            if nested_candidate is not None:
                return {
                    "path": path,
                    "action": "BLOCKED",
                    "reason": "Canonical and nested Daily Optimizer artifacts coexist",
                }
            try:
                self._validated_records(
                    path, historical=True, project_root=root, seed=seed
                )
                return {
                    "path": path,
                    "action": "REUSE",
                    "reason": "Verified Daily declared filters historical teacher",
                }
            except (ValueError, TypeError, KeyError, OSError) as exc:
                return {
                    "path": path,
                    "action": "BLOCKED",
                    "reason": f"Daily history invalid: {exc}",
                }
        try:
            if nested_candidate is not None:
                expected = self._expected_historical_preflight(root)
                nested = inspect_nested_optimizer_workspace(
                    canonical_arm_dir=path.parent.parent,
                    expected_runtime_identity_sha256=expected[
                        "runtime_identity_sha256"
                    ],
                )
                if nested is None:
                    raise FormalMinTrade40PITBlocked(
                        "Nested Optimizer workspace disappeared"
                    )
                legacy_params = nested / "active_params" / path.name
                if legacy_params.is_file():
                    self._validated_records(
                        legacy_params, historical=True, project_root=root, seed=seed
                    )
                else:
                    summary_path = nested / "rolling_optimizer_summary.json"
                    if summary_path.is_file():
                        summary = load_json_strict(summary_path)
                        if (
                            not isinstance(summary, dict)
                            or summary.get("status") == "COMPLETED"
                        ):
                            raise FormalMinTrade40PITBlocked(
                                "Completed nested Optimizer summary missing signed parameters"
                            )
                return {
                    "path": path,
                    "action": "RESUME",
                    "reason": "Verified nested Daily Optimizer workspace; relocate without refitting completed folds",
                }
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return {
                "path": path,
                "action": "BLOCKED",
                "reason": f"Daily nested history invalid: {exc}",
            }
        preflight = path.parent.parent / "rolling_preflight.json"
        if preflight.is_file():
            try:
                expected = self._expected_historical_preflight(root)
                verify_optimizer_runtime_preflight(
                    preflight,
                    expected_runtime_identity_sha256=expected[
                        "runtime_identity_sha256"
                    ],
                )
                summary_path = path.parent.parent / "rolling_optimizer_summary.json"
                if summary_path.is_file():
                    summary = load_json_strict(summary_path)
                    if (
                        not isinstance(summary, dict)
                        or summary.get("status") == "COMPLETED"
                    ):
                        raise FormalMinTrade40PITBlocked(
                            "Completed Daily Optimizer summary missing signed parameters"
                        )
            except (ValueError, TypeError, KeyError, OSError) as exc:
                return {
                    "path": path,
                    "action": "BLOCKED",
                    "reason": f"Interrupted Daily history incompatible: {exc}",
                }
        return {
            "path": path,
            "action": "RESUME" if preflight.is_file() else "BUILD",
            "reason": "Independent historical Daily Optimizer teacher required",
        }

    def execute_history(self, *, project_root: str | Path, seed: int):
        root = Path(project_root).resolve()
        state = self.inspect_history(project_root=root, seed=seed)
        if state["action"] == "BLOCKED":
            raise FormalMinTrade40PITBlocked(str(state["reason"]))
        if state["action"] == "REUSE":
            return Path(state["path"])
        path = self.historical_params_path(root, seed)
        if nested_optimizer_workspace_candidate(path.parent.parent) is not None:
            expected = self._expected_historical_preflight(root)
            adopted = adopt_nested_optimizer_workspace(
                canonical_arm_dir=path.parent.parent,
                expected_runtime_identity_sha256=expected["runtime_identity_sha256"],
            )
        else:
            adopted = False
        if adopted:
            state = self.inspect_history(project_root=root, seed=seed)
            if state["action"] == "BLOCKED":
                raise FormalMinTrade40PITBlocked(str(state["reason"]))
            if state["action"] == "REUSE":
                return path
        policy = self._canonical_schedule_build_contract(
            family=self.spec["param_family"]
        )
        result = self.prepare_selection_historical_p2_params(
            project_root=root,
            dataset="full",
            param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
            trials_per_fold=int(policy["trials_per_fold"]),
            max_positions=int(DEFAULT_PORTFOLIO_MAX_POSITIONS),
            rotation=str(DEFAULT_PORTFOLIO_ROTATION),
            fixed_risk=float(DEFAULT_FIXED_RISK),
            max_position_cap_pct=float(DEFAULT_MAX_POSITION_CAP_PCT),
            optimizer_seed=seed,
            resume_parameter_training=True,
            recover_completed_strategy_compare=False,
            first_oos_date=TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
            last_oos_date=TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
            train_window_months=int(policy["train_window_months"]),
            oos_months=int(policy["oos_horizon_months"]),
            output_relative_dir=path.parent.parent.parent,
            parameter_set_identity=self.spec["historical_parameter_set"],
            strategy_id="daily",
            daily_search_profile=self.spec["search_profile"],
            build_context_label="Historical filtered Daily PIT label teacher",
        )
        if Path(result["params_path"]).resolve() != path.resolve():
            raise FormalMinTrade40PITBlocked(
                "Canonical optimizer published unexpected Daily teacher"
            )
        self._validated_records(path, historical=True, project_root=root, seed=seed)
        return path

    def inspect_current(
        self, *, project_root: str | Path, seed: int
    ) -> dict[str, object]:
        return self.inspect_formal_mintrade40_p16_teacher(
            project_root=project_root, seed=seed, baseline_arm=self.spec["baseline_arm"]
        )

    def execute_current(self, *, project_root: str | Path, seed: int):
        return self._current_p16_seed_artifact(
            Path(project_root).resolve(),
            seed,
            authorized_build=True,
            baseline_arm=self.spec["baseline_arm"],
        )

    def load_training_source(
        self, *, project_root: str | Path, seed: int, authorized_build: bool = False
    ) -> FormalMinTrade40PITSource:
        root = Path(project_root).resolve()
        if type(seed) is not int or seed <= 0:
            raise FormalMinTrade40PITBlocked(
                "Daily model FIT needs an explicit DL seed"
            )
        current_path, start, end, _policy = self._current_p16_seed_artifact(
            root,
            seed,
            authorized_build=authorized_build,
            baseline_arm=self.spec["baseline_arm"],
        )
        if authorized_build:
            self.execute_history(project_root=root, seed=seed)
        history_path = self.historical_params_path(root, seed)
        history, historical_records = self._validated_records(
            history_path, historical=True, project_root=root, seed=seed
        )
        current, current_sha = self.load_strategy_param_evaluation_view(
            current_path, evaluation_mode="rolling", start_date=start, end_date=end
        )
        if current.get("kind") != "rolling_active_param_ensemble":
            raise FormalMinTrade40PITBlocked(
                "Daily requires canonical single-member Rolling effective parameters"
            )
        current_records = self.resolve_formal_min_param_records(
            current["kind"], current["payload"]
        )
        self.validate_teacher_records(current_records, seed=seed)
        hist_sha = self.compute_strategy_param_file_sha256(history_path)
        history_periods, _ = self.build_formal_min_execution_schedule(
            historical_records, seed=seed, param_sha256=hist_sha
        )
        current_periods, _ = self.build_formal_min_execution_schedule(
            current_records, seed=seed, param_sha256=current_sha
        )
        if (
            not history_periods
            or not current_periods
            or current_periods[0].start != pd.Timestamp(start).normalize()
            or (history_periods[-1].end is None)
            or (
                history_periods[-1].end + pd.Timedelta(days=1)
                != current_periods[0].start
            )
        ):
            raise FormalMinTrade40PITBlocked(
                "Daily historical/current PIT parameter periods not contiguous"
            )

        def publications(payload, periods):
            folds = tuple(payload.get("folds") or ())
            by_start = {
                pd.Timestamp(
                    f.get("effective_start") or f["oos_start_date"]
                ).normalize(): f
                for f in folds
            }
            if len(by_start) != len(folds) or len(folds) != len(periods):
                raise FormalMinTrade40PITBlocked(
                    "Daily Optimizer fold evidence incomplete"
                )
            pub = []
            for period in periods:
                fold = by_start.get(period.start)
                if fold is None:
                    raise FormalMinTrade40PITBlocked("Daily effective fold absent")
                selected = pd.Timestamp(fold["selection_end_date"]).normalize()
                if selected != period.start - pd.Timedelta(days=1):
                    raise FormalMinTrade40PITBlocked(
                        "Daily training end after effective start"
                    )
                pub.append(str(selected.date()))
            return tuple(pub)

        published = publications(history, history_periods) + publications(
            current["payload"], current_periods
        )
        return FormalMinTrade40PITSource(
            seed=seed,
            source_id=self.spec["source_id"],
            param_sha256=canonical_json_sha256(
                {
                    "contract": self.spec["teacher_contract"],
                    "teacher_seed": seed,
                    "history_sha256": hist_sha,
                    "current_sha256": current_sha,
                    "search_profile": self.spec["search_profile"],
                }
            ),
            schedule=tuple(
                (
                    replace(period, source_id=self.spec["source_id"])
                    for period in (*history_periods, *current_periods)
                )
            ),
            publication_dates=published,
            oos_start_date=str(current_periods[0].start.date()),
        )

    @staticmethod
    def compute_file_sha256(path):
        from core.file_integrity import compute_file_sha256
        return compute_file_sha256(path)

    @staticmethod
    def broker_fingerprint(metric):
        from services.audit.daily_trade40_oracle import truth_producer_fingerprint
        return truth_producer_fingerprint(metric)

    def execution_fingerprint(self):
        root = Path(__file__).resolve().parents[2]
        paths = {path for _, path, _ in self.spec["calculation_owners"]
                 if path.startswith(("core/", "filters/"))}
        return canonical_json_sha256({
            "source_id": self.SOURCE_ID, "entry_policy": self.ENTRY_POLICY,
            "broker_execution": self.broker_fingerprint("net_pnl"),
            "implementation": {p: self.compute_file_sha256(root / p) for p in sorted(paths)},
        })

    def truth_producer_fingerprint(self):
        root = Path(__file__).resolve().parents[2]
        paths = {path for _, path, _ in self.spec["calculation_owners"]}
        return canonical_json_sha256({
            "target_id": self.spec["target_id"], "entry_policy": self.ENTRY_POLICY,
            "source_id": self.SOURCE_ID, "numerical_execution": self.execution_fingerprint(),
            "implementation": {p: self.compute_file_sha256(root / p) for p in sorted(paths)},
        })

    def _validate_schedule(self, schedule, seed):
        self.validate_teacher_records([
            {"members": [{"seed": seed}], "params_obj": p.params,
             "params_signature": p.params_signature} for p in schedule
        ], seed=seed)

    def materialize_labels(self, source, *, expected_seed, **kwargs):
        from services.breakout_quality.daily_full_p20_pit_source import (
            materialize_daily_trade_path_pit_labels,
        )
        self._validate_schedule(source.schedule, expected_seed)
        module, function = self.spec["outcome_builder"]
        labels = materialize_daily_trade_path_pit_labels(
            source, expected_seed=expected_seed,
            spec={"source_id": self.SOURCE_ID, "entry_policy": self.ENTRY_POLICY,
                  **self.spec["pit_contract"], "execution_sha256": self.execution_fingerprint},
            compute_batch=getattr(import_module(module), function), **kwargs,
        )
        return replace(labels, preflight={
            **labels.preflight, "economic_teacher_source_id": self.SOURCE_ID,
        })

    def build_oracle_scores(self, **kwargs):
        from services.audit.daily_filtered_trade_path_truth import build_filtered_daily_oracle_scores
        self._validate_schedule(kwargs["canonical_schedule"], kwargs["seed"])
        return build_filtered_daily_oracle_scores(
            spec=self.spec, fingerprint=self.truth_producer_fingerprint(), **kwargs,
        )

    def load_positive_truth(self, **kwargs):
        from services.audit.daily_filtered_trade_path_truth import load_filtered_positive_truth
        return load_filtered_positive_truth(
            spec=self.spec, fingerprint=self.truth_producer_fingerprint(), **kwargs,
        )

    def resolve_loader_kwargs(
        self, *, project_root: str | Path, seed: int, authorized_build: bool = False
    ) -> dict[str, object]:
        root = Path(project_root).resolve()
        source = self.load_training_source(
            project_root=root, seed=seed, authorized_build=authorized_build
        )
        materializer = (self.materialize_labels if self.spec.get("supervised_source") else
                        getattr(import_module(self.spec["pit_module"]), self.spec["pit_materializer"]))
        return {
            "target_materializer": partial(
                materializer,
                source=source,
                expected_seed=seed,
                cache_project_root=str(root),
            ),
            "target_lineage": {
                "contract": self.spec["teacher_contract"],
                "target_kind": self.spec["target_kind"],
                "target_id": self.spec["target_id"],
                "entry_policy": self.spec["entry_policy"],
                "horizon_mode": "full_trade_cycle",
                "value_field": "raw_net_pnl",
                "seed": seed,
                "optimizer_teacher_seed": seed,
                "maturity_policy": "natural_exit_or_research_terminal_closeout_pit_v1",
                "source_sha256": source.param_sha256,
                "first_effective_date": str(source.schedule[0].start.date()),
                "last_source_period_start": str(source.schedule[-1].start.date()),
                "oos_start_date": source.oos_start_date,
                "teacher_periods": [
                    {
                        "start": str(p.start.date()),
                        "end": None if p.end is None else str(p.end.date()),
                        "params_signature": p.params_signature,
                        "published": published,
                    }
                    for p, published in zip(source.schedule, source.publication_dates)
                ],
            },
        }


from core.daily_trade_path_capabilities import daily_truth_capability

TEACHER = DailyTradePathTeacher(
    daily_truth_capability("daily_trade_path_daily_full_p20_net_pnl")
)
HISTORY_ID, PROFILE, BASELINE_ARM = (
    TEACHER.HISTORY_ID,
    TEACHER.PROFILE,
    TEACHER.BASELINE_ARM,
)
SOURCE_ID, ENTRY_POLICY = TEACHER.SOURCE_ID, TEACHER.ENTRY_POLICY


def historical_params_path(*args, **kwargs):
    return TEACHER.historical_params_path(*args, **kwargs)


_expected_historical_preflight = TEACHER._expected_historical_preflight
_validated_records = TEACHER._validated_records


def validate_teacher_records(*args, **kwargs):
    return TEACHER.validate_teacher_records(*args, **kwargs)


def inspect_history(*args, **kwargs):
    return TEACHER.inspect_history(*args, **kwargs)


def execute_history(*args, **kwargs):
    return TEACHER.execute_history(*args, **kwargs)


def inspect_current(*args, **kwargs):
    return TEACHER.inspect_current(*args, **kwargs)


def execute_current(*args, **kwargs):
    return TEACHER.execute_current(*args, **kwargs)


def load_training_source(*args, **kwargs):
    return TEACHER.load_training_source(*args, **kwargs)


def resolve_loader_kwargs(*args, **kwargs):
    return TEACHER.resolve_loader_kwargs(*args, **kwargs)


_TEACHERS_BY_KIND = {TEACHER.spec["target_kind"]: TEACHER}


def teacher_for_kind(kind):
    """Resolve one declared teacher instance per target contract."""
    from core.daily_trade_path_capabilities import supervised_daily_capabilities
    capability = next((s for s in supervised_daily_capabilities() if s["target_kind"] == kind), None)
    if capability is None:
        if kind == TEACHER.spec["target_kind"]:
            return TEACHER
        raise ValueError(f"Unregistered supervised Daily contract: {kind!r}")
    current = _TEACHERS_BY_KIND.get(kind)
    if current is None or current.spec != capability:
        current = _TEACHERS_BY_KIND[kind] = DailyTradePathTeacher(capability)
    return current


def teacher_for_recipe(recipe):
    from core.daily_trade_path_capabilities import daily_truth_capability
    capability = daily_truth_capability(recipe)
    if capability is None or not capability.get("supervised_source"):
        raise ValueError(f"Undeclared supervised Daily recipe: {recipe!r}")
    return teacher_for_kind(capability["target_kind"])
