"""Behavioral checks for read-only contracts and execution environment isolation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.artifact_inspection import InspectionStatus
from core.file_integrity import canonical_json_sha256
from services.optimizer.execution_settings import (
    OptimizerExecutionSettings, isolate_optimizer_process_environment,
    optimizer_process_environment, temporary_process_environment,
)
from services.optimizer import strategy_param_training as training
from services.optimizer.strategy_param_contracts import build_min_roos_schedule_contract
from strategies.daily.search_space import DAILY_SEARCH_PROFILES


def _fingerprint_in_fresh_worker(resolved):
    from core.training_policy import optimizer_runtime_model_mode_scope
    from core.walk_forward_policy import build_optimizer_effective_policy_fingerprint
    with optimizer_runtime_model_mode_scope("study"):
        return build_optimizer_effective_policy_fingerprint(resolved)


def _policy_member_fixture(mode):
    from services.optimizer import outer_rolling_policy as policies
    selected = "local" if mode == "full" else "base"
    session = SimpleNamespace(
        walk_forward_policy={"model_mode": mode, "train_start_year": 2020,
                             "train_start_date": "2020-01-01", "search_train_end_date": "2021-12-31"},
        strategy_param_scientific_contract={"schema_version": 1, "strategy": {"parameter_policy": selected}},
        train_start_year=2020,
    )
    trial = SimpleNamespace(number=0, params={}, user_attrs={})
    finalist = dict(trial=trial, base_score=2.0, base_rank=1, local_min_score=1.0,
                    local_retention=.5, gate_pass=True)
    members = policies.build_optimizer_policy_members_from_finalists(
        [finalist], objective_mode="full", member_index=1, seed=7, session=session,
    )
    return session, members


def _policy_member_in_fresh_worker(mode):
    import config.training_policy as settings
    from core.training_policy import optimizer_runtime_model_mode_scope
    with (patch.object(settings, "OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED", False),
          patch.object(settings, "OPTIMIZER_FULL_MODE_LOCAL_MIN_REVIEW_ENABLED", True),
          optimizer_runtime_model_mode_scope("oos" if mode == "full" else "full")):
        return _policy_member_fixture(mode)[1]


class OptimizerBoundaryTests(unittest.TestCase):
    def test_policy_members_follow_explicit_mode_across_threads_and_spawn(self):
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
        import config.training_policy as settings
        from core.training_policy import optimizer_runtime_model_mode_scope
        from services.optimizer import outer_rolling_policy as policies
        modes = ("full", "oos", "full", "oos")
        with (patch.object(settings, "OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED", False),
              patch.object(settings, "OPTIMIZER_FULL_MODE_LOCAL_MIN_REVIEW_ENABLED", True),
              optimizer_runtime_model_mode_scope("study")):
            expected = [_policy_member_fixture(mode)[1] for mode in modes]
            self.assertEqual([list(value) for value in expected], [["local"], ["base"], ["local"], ["base"]])
            self.assertIn("local", policies.get_optimizer_paramset_policy_names(model_mode="full"))
            self.assertNotIn("local", policies.get_optimizer_paramset_policy_names(model_mode="oos"))
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual([result[1] for result in pool.map(_policy_member_fixture, modes)], expected)
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
                self.assertEqual(list(pool.map(_policy_member_in_fresh_worker, modes)), expected)
            session, _ = _policy_member_fixture("full")
            session.walk_forward_policy["model_mode"] = "oos"
            with self.assertRaisesRegex(ValueError, "停用"):
                policies.get_optimizer_paramset_policy_names(session=session)

    def test_full_policy_export_paths_and_dashboard_consume_session_policy(self):
        from contextlib import redirect_stdout
        import io
        import config.training_policy as settings
        from services.optimizer import application, static_ensemble_dashboard as dashboard
        from services.optimizer import strategy_param_repository as repository
        with (patch.object(settings, "OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED", False),
              patch.object(settings, "OPTIMIZER_FULL_MODE_LOCAL_MIN_REVIEW_ENABLED", True),
              patch.object(application, "PROJECT_ROOT", self.root),
              patch.object(repository, "refresh_strategy_parameter_manifest")):
            session, members = _policy_member_fixture("full")
            first = members["local"]
            ensemble = [first, {**first, "member_index": 2, "seed": 11, "optimizer_seed": 11}]
            paths, payloads = application._write_static_seed_ensemble_policy_paramsets(
                policy_members_by_policy={"local": ensemble}, seeds=[7, 11], objective_mode="full",
                walk_forward_policy=session.walk_forward_policy, dataset_label="fixture",
                selected_model_mode="full", trials_per_seed=1, models_dir=self.root / "models",
                strategy_params_root=self.root / "strategy_params",
            )
            self.assertEqual(list(paths), ["local"])
            self.assertEqual(list(payloads), ["local"])
            path = Path(paths["local"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payloads["local"])
            self.assertIn(str(path), application._visible_policy_paramset_paths(paths, model_mode="full").values())
            self.assertEqual(application._visible_policy_paramset_paths(paths, model_mode="oos"), {})
            before = path.read_bytes()
            skipped, skipped_payloads = application._write_static_seed_ensemble_policy_paramsets(
                policy_members_by_policy={"local": ensemble}, seeds=[7, 11], objective_mode="full",
                walk_forward_policy={**session.walk_forward_policy, "model_mode": "oos"},
                dataset_label="fixture", selected_model_mode="oos", trials_per_seed=1,
                models_dir=self.root / "models", strategy_params_root=self.root / "strategy_params",
            )
            self.assertEqual((skipped, skipped_payloads), ({}, {}))
            self.assertEqual(path.read_bytes(), before)
            metrics = {"local": {"available": True, "rank_1_oos": 1.25, "rank_1_plain_romd": 1.25}}
            with (patch.object(dashboard, "_build_search_train_dates_for_session", return_value=[]),
                  patch.object(dashboard, "_build_static_policy_rows_from_paramsets", return_value=(metrics, {}))):
                row = dashboard.build_optimizer_static_ensemble_single_fold_oos_row(
                    session, ensemble_payload=payloads["local"], policy_paramsets=payloads,
                )
                self.assertEqual(row["local"], metrics["local"])
                output = io.StringIO()
                with redirect_stdout(output):
                    dashboard.print_optimizer_static_ensemble_rolling_oos_table(
                        session, ensemble_payload=payloads["local"], policy_paramsets=payloads,
                    )
                self.assertIn("local ensemble", output.getvalue())
                self.assertIn("1.25", output.getvalue())
                self.assertNotIn("base ensemble", output.getvalue())

    def test_resolved_policy_fingerprint_survives_spawn(self):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from core.walk_forward_policy import build_optimizer_effective_policy_fingerprint, build_optimizer_runtime_policy
        from services.optimizer import application
        resolved = build_optimizer_runtime_policy(
            application.load_walk_forward_policy(application.PROJECT_ROOT), "full", latest_data_date="2026-11-24",
        )
        expected = build_optimizer_effective_policy_fingerprint(resolved)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            self.assertEqual(pool.submit(_fingerprint_in_fresh_worker, resolved).result(10), expected)

    def test_effective_policy_and_session_ignore_previous_and_concurrent_modes(self):
        from concurrent.futures import ThreadPoolExecutor
        from core import training_policy as policy
        from core.walk_forward_policy import build_optimizer_effective_policy_fingerprint, build_optimizer_runtime_policy
        from services.optimizer import application
        from services.optimizer.scientific_runtime import resolve_session_local_min_review_enabled

        resolved = build_optimizer_runtime_policy(
            application.load_walk_forward_policy(application.PROJECT_ROOT), "full", latest_data_date="2026-11-24",
        )
        session = SimpleNamespace(walk_forward_policy=resolved)
        with (patch.object(policy._settings, "OPTIMIZER_LOCAL_MIN_REVIEW_ENABLED", False),
              patch.object(policy._settings, "OPTIMIZER_FULL_MODE_LOCAL_MIN_REVIEW_ENABLED", True)):
            with policy.optimizer_runtime_model_mode_scope("full"):
                expected = build_optimizer_effective_policy_fingerprint(resolved)
            with policy.optimizer_runtime_model_mode_scope("oos"):
                self.assertEqual(build_optimizer_effective_policy_fingerprint(resolved), expected)
                self.assertTrue(resolve_session_local_min_review_enabled(session))
                with ThreadPoolExecutor(max_workers=1) as pool:
                    self.assertEqual(pool.submit(build_optimizer_effective_policy_fingerprint, resolved).result(3), expected)
                    self.assertTrue(pool.submit(resolve_session_local_min_review_enabled, session).result(3))
                with self.assertRaisesRegex(RuntimeError, "cancel"):
                    with policy.optimizer_runtime_model_mode_scope("full"):
                        policy.set_optimizer_runtime_model_mode("study")
                        raise RuntimeError("cancel")
                self.assertEqual(policy.resolve_optimizer_runtime_model_mode(), "oos")

    def test_application_early_return_and_worker_failure_restore_runtime_mode(self):
        from core import training_policy as policy
        from services.optimizer import application

        with policy.optimizer_runtime_model_mode_scope("oos"):
            with (patch.object(application, "load_walk_forward_policy", return_value={}),
                  patch.object(application, "resolve_dataset_profile_from_cli_env", side_effect=ValueError("invalid dataset"))):
                self.assertEqual(application.main(["optimizer", "--model", "full"], environ={}), 1)
            self.assertEqual(policy.resolve_optimizer_runtime_model_mode(), "oos")
            with (patch.object(application, "configure_optuna_logging", side_effect=KeyboardInterrupt),
                  self.assertRaises(KeyboardInterrupt)):
                application._run_nonrolling_seed_ensemble_member_process_task({"walk_forward_policy": {"model_mode": "full"}})
            self.assertEqual(policy.resolve_optimizer_runtime_model_mode(), "oos")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def schedule(self, **kwargs):
        return build_min_roos_schedule_contract(
            first_oos_date="2021-01-01", last_oos_date="2021-12-31",
            train_window_months=120, oos_months=12, **kwargs,
        )

    def payload(self):
        return {
            "meta": {**self.schedule()["meta"], "trials_per_fold": 3},
            "params_ensemble_by_effective_date": {"2021-01-01": [{"params": {
                **{key: 2 for key in training.MIN_ROOS_SEARCH_FIELDS},
                "tp_percent": training.OPTIMIZER_FIXED_TP_PERCENT,
            }}]},
        }

    def write_payload(self, payload=None):
        path = self.root / "params.json"
        path.write_text(json.dumps(self.payload() if payload is None else payload), encoding="utf-8")
        return path

    def test_all_daily_profiles_share_the_exact_reference_projection(self):
        base = self.schedule()
        for profile, spec in DAILY_SEARCH_PROFILES.items():
            with self.subTest(profile=profile):
                result = self.schedule(strategy_id="daily", daily_search_profile=profile)
                expected = deepcopy(base["reference_payload"])
                expected.update({
                    "search_fields": list(spec["fields"]), "entry_strategy": "Daily",
                    "entry_universe": "unscored_all_adjacent_quote_stockdays_v1",
                    "breakout_required": False, "daily_execution_contract": spec["execution_contract"],
                })
                self.assertEqual(result["reference_payload"], expected)
                self.assertEqual(result["sha256"], canonical_json_sha256(expected))
                self.assertEqual(result["reference_type"], "canonical_daily_min_roos_fixed_defaults")

    def test_daily_profile_names_do_not_define_family_or_fixed_rules(self):
        from config.strategy_params import STRATEGY_PARAM_PROFILES
        from core.strategy_param_policy import get_strategy_param_training_profile
        from services.optimizer.strategy_param_contracts import build_min_roos_adaptation_contract

        for original, descriptor in tuple(DAILY_SEARCH_PROFILES.items()):
            with self.subTest(profile=original):
                alias = "opaque_name" if descriptor["filter_profile"] == "full" else "common_filters_in_an_unrelated_name"
                raw = next(deepcopy(v) for v in STRATEGY_PARAM_PROFILES.values()
                           if v.get("daily_search_profile") == original)
                raw["daily_search_profile"] = alias
                kwargs = dict(arm_id="CONTROL", training_dl_enabled=False, strategy_id="daily")
                expected = build_min_roos_adaptation_contract(**kwargs, daily_search_profile=original)
                with (patch.dict(DAILY_SEARCH_PROFILES, {alias: deepcopy(descriptor)}),
                      patch.dict(STRATEGY_PARAM_PROFILES, {"synthetic_profile": raw})):
                    resolved = get_strategy_param_training_profile("synthetic_profile")
                    self.assertEqual(resolved.filter_profile, descriptor["filter_profile"])
                    actual = build_min_roos_adaptation_contract(**kwargs, daily_search_profile=alias)
                    self.assertEqual(actual, expected)
                    self.assertEqual(self.schedule(strategy_id="daily", daily_search_profile=alias),
                                     self.schedule(strategy_id="daily", daily_search_profile=original))

    def test_daily_profile_requires_declared_family_and_fixed_rules(self):
        from strategies.daily.search_space import resolve_daily_search_profile

        for field in ("filter_profile", "fixed_rule_contract"):
            incomplete = deepcopy(next(iter(DAILY_SEARCH_PROFILES.values())))
            incomplete.pop(field)
            with self.subTest(field=field), patch.dict(DAILY_SEARCH_PROFILES, {"incomplete": incomplete}):
                with self.assertRaises(ValueError):
                    resolve_daily_search_profile("incomplete")

    def test_min_and_full_validation_never_write_even_when_migration_is_needed(self):
        path = self.write_payload()
        before = path.read_bytes(), path.stat().st_mtime_ns
        with patch.object(training, "atomic_write_json", side_effect=AssertionError("inspection wrote")):
            minimum = training.validate_min_roos_params(
                path=path, baseline_contract=self.schedule(),
                fold_overrides={"2021-01-01": {}}, args=SimpleNamespace(trials_per_fold=3),
                training_dl_enabled=False, arm_id="synthetic",
            )
            full = training.validate_selection_full_roos_params(
                path=path, schedule_contract=self.schedule(), args=SimpleNamespace(trials_per_fold=3),
            )
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
        self.assertEqual(minimum["breakout_quality_param_adaptation"]["mode"], "min_roos_training")
        self.assertEqual(full["breakout_quality_param_adaptation"]["mode"], "selection_full_roos_training")
        inspection = training.inspect_min_roos_params(
            path=path, baseline_contract=self.schedule(), fold_overrides={"2021-01-01": {}},
            args=SimpleNamespace(trials_per_fold=3), training_dl_enabled=False, arm_id="synthetic",
        )
        self.assertEqual([item.field for item in inspection.migrations], ["breakout_quality_param_adaptation"])
        training.publish_validated_strategy_params(path, minimum)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), minimum)
        current = training.inspect_min_roos_params(
            path=path, baseline_contract=self.schedule(), fold_overrides={"2021-01-01": {}},
            args=SimpleNamespace(trials_per_fold=3), training_dl_enabled=False, arm_id="synthetic",
        )
        self.assertEqual(current.migrations, ())

    def test_rejected_parameter_evidence_stays_byte_identical(self):
        payload = self.payload()
        payload["meta"]["trials_per_fold"] = 4
        path = self.write_payload(payload)
        before = path.read_bytes(), path.stat().st_mtime_ns
        with self.assertRaises(ValueError), patch.object(training, "atomic_write_json", side_effect=AssertionError("inspection wrote")):
            training.validate_min_roos_params(
                path=path, baseline_contract=self.schedule(), fold_overrides={"2021-01-01": {}},
                args=SimpleNamespace(trials_per_fold=3), training_dl_enabled=False, arm_id="synthetic",
            )
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))

    def test_public_runtime_contract_builders_never_publish_source_witnesses(self):
        args = SimpleNamespace(dataset="full", trials_per_fold=3, max_positions=10,
            rotation="on", fixed_risk=0.01, max_position_cap_pct=0.2)
        inventory = {"fixture": "read-only"}
        with (patch.object(training, "build_source_data_inventory", side_effect=AssertionError("contract published")),
              patch.object(training, "_build_source_data_inventory", return_value=inventory) as read):
            minimum = training.build_min_roos_runtime_contract(
                root=self.root, args=args, baseline_contract=self.schedule(), fold_overrides={},
                model_artifact=None, arm_id="synthetic", training_dl_enabled=False, binary_pit=None,
            )
            full = training.build_selection_full_roos_runtime_contract(
                root=self.root, args=args, schedule_contract=self.schedule(),
            )
        self.assertEqual(read.call_count, 2)
        self.assertEqual(minimum["dataset_identity"], inventory)
        self.assertEqual(full["dataset_identity"], inventory)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_environment_is_snapshot_based_pickle_safe_and_restored_after_failure(self):
        incoming = {"OPTIMIZER_PREP_CACHE_MAX_ITEMS": "7"}
        settings = OptimizerExecutionSettings.from_environ(incoming)
        incoming["OPTIMIZER_PREP_CACHE_MAX_ITEMS"] = "99"
        self.assertEqual(pickle.loads(pickle.dumps(settings)), settings)
        with temporary_process_environment({"OPTIMIZER_PREP_CACHE_MAX_ITEMS": "13", "OPTIMIZER_UNRELATED": "keep"}):
            before = dict(os.environ)
            with self.assertRaisesRegex(RuntimeError, "cancel"):
                with optimizer_process_environment(settings):
                    self.assertEqual(os.environ["OPTIMIZER_PREP_CACHE_MAX_ITEMS"], "7")
                    self.assertNotIn("OPTIMIZER_UNRELATED", os.environ)
                    os.environ["OPTIMIZER_CREATED_INSIDE"] = "1"
                    raise RuntimeError("cancel")
            self.assertEqual(dict(os.environ), before)

    def test_session_and_thread_worker_consume_the_frozen_settings(self):
        from concurrent.futures import ThreadPoolExecutor
        from services.optimizer.session import OptimizerSession

        session = OptimizerSession.__new__(OptimizerSession)
        session.execution_settings = OptimizerExecutionSettings.from_environ({
            "OPTIMIZER_PREP_CACHE_MAX_ITEMS": "7",
            "OPTIMIZER_FULL_EVAL_CACHE_MAX_ITEMS": "9",
        })
        with optimizer_process_environment(session.execution_settings):
            with temporary_process_environment({"OPTIMIZER_PREP_CACHE_MAX_ITEMS": "99"}):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    capacity = executor.submit(session._resolve_prepared_trial_input_cache_max_items).result(timeout=3)
                self.assertEqual(capacity, 7)
                self.assertEqual(session._resolve_full_evaluation_cache_max_items(), 9)

    def test_application_scope_does_not_mutate_caller_or_next_action(self):
        @isolate_optimizer_process_environment
        def run(argv=None, environ=None):
            seen = dict(environ)
            environ["OPTIMIZER_ROLLING_FOLD_WORKERS"] = "4"
            os.environ["OPTIMIZER_ROLLING_FOLD_WORKERS"] = "4"
            return seen

        caller = {"OPTIMIZER_PREP_CACHE_MAX_ITEMS": "5"}
        before = dict(os.environ)
        self.assertEqual(run([], caller), caller)
        self.assertEqual(caller, {"OPTIMIZER_PREP_CACHE_MAX_ITEMS": "5"})
        self.assertEqual(run(environ={}), {})
        self.assertEqual(dict(os.environ), before)

    def test_application_freezes_after_cli_overrides_and_passes_the_snapshot(self):
        from services.optimizer import application

        class CapturedSession(BaseException):
            pass

        captured = {}
        def capture(*, walk_forward_policy, execution_settings):
            captured.update(walk_forward_policy=walk_forward_policy, execution_settings=execution_settings)
            raise CapturedSession()

        incoming = {"OPTIMIZER_PREP_CACHE_MAX_ITEMS": "7"}
        before = dict(os.environ)
        with (patch.object(application, "build_optimizer_session", side_effect=capture),
              patch.object(application, "load_walk_forward_policy", return_value={}),
              patch.object(application, "set_optimizer_runtime_model_mode"),
              patch.object(application, "_mode_needs_latest_data_date", return_value=False),
              patch.object(application, "get_dataset_generation_namespace", return_value=None),
              patch.object(application, "build_optimizer_runtime_policy", return_value={}),
              self.assertRaises(CapturedSession)):
            application.main(
                ["optimizer", "--model", "full", "--timing", "--outer-oos", "--timing-fold-workers", "3"],
                environ=incoming,
            )
        resolved = captured["execution_settings"].as_environ()
        self.assertEqual(resolved["OPTIMIZER_PREP_CACHE_MAX_ITEMS"], "7")
        self.assertEqual(resolved["OPTIMIZER_ROLLING_FOLD_WORKERS"], "3")
        self.assertEqual(resolved["OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS"], "1")
        self.assertNotIn("execution_settings", captured["walk_forward_policy"])
        self.assertEqual(incoming, {"OPTIMIZER_PREP_CACHE_MAX_ITEMS": "7"})
        self.assertEqual(dict(os.environ), before)

    def test_ambient_snapshot_waits_for_the_previous_run_to_restore(self):
        import threading
        from services.optimizer import execution_settings as owner

        entered, release, second_attempted = (threading.Event() for _ in range(3))
        lock = threading.RLock()
        class ObservedLock:
            def __enter__(self):
                if threading.current_thread().name == "settings-second":
                    second_attempted.set()
                lock.acquire()
            def __exit__(self, *args):
                lock.release()

        @isolate_optimizer_process_environment
        def run(environ=None):
            if threading.current_thread().name == "settings-first":
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("second run did not attempt its scope")
            return environ.get("OPTIMIZER_PREP_CACHE_MAX_ITEMS")

        values, errors = {}, []
        def invoke(name, environ):
            try:
                values[name] = run(environ=environ)
            except BaseException as exc:
                errors.append(exc)

        key = "OPTIMIZER_PREP_CACHE_MAX_ITEMS"
        before = os.environ.get(key)
        first = threading.Thread(target=invoke, args=("first", {key: "7"}), name="settings-first")
        second = threading.Thread(target=invoke, args=("second", None), name="settings-second")
        try:
            os.environ[key] = "13"
            with patch.object(owner, "_ENVIRONMENT_LOCK", ObservedLock()):
                first.start()
                self.assertTrue(entered.wait(3))
                second.start()
                self.assertTrue(second_attempted.wait(3))
                release.set()
                first.join(3)
                second.join(3)
                self.assertFalse(first.is_alive() or second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(values, {"first": "7", "second": "13"})
            self.assertEqual(os.environ[key], "13")
        finally:
            release.set()
            if first.ident is not None:
                first.join(3)
            if second.ident is not None:
                second.join(3)
            if before is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = before

    def test_nonrolling_process_task_passes_its_pickled_snapshot_to_the_session(self):
        from services.optimizer import application

        class CapturedSession(BaseException):
            pass

        settings = OptimizerExecutionSettings.from_environ({"OPTIMIZER_PREP_CACHE_MAX_ITEMS": "7"})
        task = pickle.loads(pickle.dumps(dict(
            walk_forward_policy={"model_mode": "full"}, selected_data_dir=str(self.root),
            optimizer_required_min_rows=5, requested_trials=1, seed=7, member_index=1,
            member_count=2, raw_data_loader_path="fixture", execution_settings=settings,
        )))
        captured = {}
        def capture(*, walk_forward_policy, output_dir, execution_settings):
            captured.update(walk_forward_policy=walk_forward_policy, output_dir=output_dir,
                            execution_settings=execution_settings)
            raise CapturedSession()
        with (patch.object(application, "build_optimizer_session", side_effect=capture),
              patch.object(application, "set_optimizer_runtime_model_mode"),
              patch.object(application, "configure_optuna_logging"),
              patch.object(application, "_load_callable_from_import_path"),
              temporary_process_environment({"OPTIMIZER_PREP_CACHE_MAX_ITEMS": "99"}),
              self.assertRaises(CapturedSession)):
            application._run_nonrolling_seed_ensemble_member_process_task(task)
        self.assertEqual(captured["execution_settings"], settings)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_rolling_task_and_spec_pass_frozen_worker_settings_without_environment_writes(self):
        from services.optimizer import outer_rolling_oos as owner, session_factory
        from services.optimizer.outer_rolling_plan import OuterRollingConfig, _build_rolling_folds

        class CapturedSession(BaseException):
            pass

        config = OuterRollingConfig(
            training_start_year=2010, first_oos_year=2021, last_oos_year=2021,
            trials_per_fold=1, raw_universe_required_min_rows=5,
        )
        fold = _build_rolling_folds(config)[0]
        captured = {}
        def capture(*, walk_forward_policy, execution_settings, runtime_cache_identity=None):
            captured.update(walk_forward_policy=walk_forward_policy, execution_settings=execution_settings,
                            runtime_cache_identity=runtime_cache_identity)
            raise CapturedSession()

        for spec in ({}, {"runtime_cache_identity": "fixture"}):
            for prep in ({}, {"OPTIMIZER_PREP_CACHE_MAX_ITEMS": "7"}):
                settings = OptimizerExecutionSettings.from_environ({
                    "OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS": "9", **prep,
                })
                common = dict(
                    project_root=str(self.root), output_dir=str(self.root), base_policy={},
                    selected_data_dir=str(self.root), optimizer_required_min_rows=5,
                    session_ts="fixture", optimizer_seed=7, sampler_kind="random",
                    timing_mode=False, fold_workers=2, optimizer_session_spec=spec,
                    progress_display_contract={}, execution_settings=settings,
                )
                task = owner._build_outer_rolling_fold_task(
                    mode="seed", fold=fold, fold_idx=1, fold_count=1,
                    config_payload=owner._build_outer_rolling_fold_config_payload(config, include_end_cap_date=False),
                    common=common, log_dir=str(self.root),
                )
                task.update(log_path="", seed_ensemble_member=True, mirror_console=True)
                task = pickle.loads(pickle.dumps(task))
                before = dict(os.environ)
                with (patch.object(session_factory, "build_optimizer_session", side_effect=capture),
                      patch.object(session_factory, "configure_optuna_logging"),
                      patch.object(owner, "build_optimizer_runtime_policy", side_effect=lambda policy, mode: policy),
                      patch.object(owner, "_write_parallel_fold_progress_event"),
                      self.assertRaises(CapturedSession)):
                    owner._run_outer_rolling_oos_fold_task(task)
                expected = prep.get("OPTIMIZER_PREP_CACHE_MAX_ITEMS", "9")
                self.assertEqual(captured["execution_settings"].as_environ()["OPTIMIZER_PREP_CACHE_MAX_ITEMS"], expected)
                self.assertEqual(dict(os.environ), before)
                self.assertEqual(common["execution_settings"], settings)
                self.assertEqual(task["optimizer_session_spec"], spec)
                self.assertNotIn("execution_settings", captured["walk_forward_policy"])
                self.assertNotIn("execution_settings", spec)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_completed_inspection_reports_reason_and_does_not_backfill(self):
        from services.research import strategy_multi_seed_robustness as owner

        run = self.root / "fingerprint"
        run.mkdir()
        contract = {"fingerprint": run.name}
        manifest = {"status": "COMPLETED", "contract": contract}
        (run / owner.MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
        (run / owner.SUMMARY_FILENAME).write_text(json.dumps({"contract": contract}), encoding="utf-8")
        (run / owner.REPORT_FILENAME).write_text("existing report", encoding="utf-8")
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir()}
        with (patch.object(owner, "_load_validated_scientific_observations", return_value={
                "stochastic_arms": (), "seeds": (7,), "seed_frame": object(), "seed_yearly_frame": object(),
            }), patch.object(owner, "_validate_summary_seed_aggregates"),
              patch.object(owner, "_write_json", side_effect=AssertionError("inspection wrote"))):
            result = owner.inspect_completed_robustness_run(run, expected_contract=contract)
            self.assertTrue(result.ready, result.reason)
            self.assertIsNotNone(owner._validate_completed_run(run, backfill_integrity=True))
            mismatch = owner.inspect_completed_robustness_run(run, expected_contract={"fingerprint": "other"})
            self.assertEqual(mismatch.status, InspectionStatus.INCOMPATIBLE)
            self.assertIn("fingerprint", mismatch.reason)
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir()})
        (run / owner.MANIFEST_FILENAME).write_text("{broken", encoding="utf-8")
        invalid = owner.inspect_completed_robustness_run(run)
        self.assertEqual(invalid.status, InspectionStatus.INVALID)
        self.assertIn("JSONDecodeError", invalid.reason)


if __name__ == "__main__":
    unittest.main()
