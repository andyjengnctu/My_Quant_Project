"""Reusable publication invariants for Strategy Compare and Robustness.

Called by the existing integration synthetic; not a second runner or test gate.
"""
from __future__ import annotations

import inspect
import io
import json
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


def append_robustness_publication_checks(*, check_true, project_root, robustness_runtime):
    # Rolling Robustness: one publishing process owns both its exact result
    # fingerprint and the shared model workspace until finalization/cleanup.
    # Existing core.file_integrity OS locks are tested across processes; do not
    # introduce a second locking implementation for this workflow.
    import subprocess
    import sys
    from core.file_integrity import sibling_publication_lock_path
    with tempfile.TemporaryDirectory(prefix="robustness_publication_lock_") as temp_dir:
        lock_test_root = Path(temp_dir)
        a = {"fingerprint": "run_a", "model_artifact_fingerprint": "shared_model"}
        b = {"fingerprint": "run_b", "model_artifact_fingerprint": "shared_model"}
        c = {"fingerprint": "run_c", "model_artifact_fingerprint": "different_model"}
        def result_root(value):
            return lock_test_root / "results" / value["fingerprint"]
        def model_root(value):
            return lock_test_root / "models" / value["model_artifact_fingerprint"]

        with patch.object(robustness_runtime, "_run_root", side_effect=result_root), patch.object(
            robustness_runtime, "_model_work_root", side_effect=model_root
        ):
            same_run_blocked = False
            same_model_blocked = False
            independent_run_allowed = False
            outside_process_blocked = False
            with robustness_runtime._exclusive_robustness_run(a):
                try:
                    with robustness_runtime._exclusive_robustness_run(a):
                        pass
                except RuntimeError:
                    same_run_blocked = True
                try:
                    with robustness_runtime._exclusive_robustness_run(b):
                        pass
                except RuntimeError:
                    same_model_blocked = True
                with robustness_runtime._exclusive_robustness_run(c):
                    independent_run_allowed = True
                locked_path = sibling_publication_lock_path(model_root(a))
                # Another interpreter must not acquire this lock while the
                # first process owns it; this also exercises Windows msvcrt.
                child_script = (
                    "from core.file_integrity import exclusive_file_lock\n"
                    "import sys\n"
                    "try:\n"
                    "    with exclusive_file_lock(sys.argv[1]):\n"
                    "        sys.exit(2)\n"
                    "except RuntimeError:\n"
                    "    sys.exit(0)\n"
                )
                child = subprocess.run(
                    [sys.executable, "-c", child_script, str(locked_path)],
                    cwd=str(project_root), capture_output=True, text=True, timeout=8,
                )
                outside_process_blocked = child.returncode == 0
            try:
                with robustness_runtime._exclusive_robustness_run(b):
                    raise RuntimeError("synthetic interrupted robustness process")
            except RuntimeError as exc:
                interrupted_was_observed = "synthetic interrupted" in str(exc)
            with robustness_runtime._exclusive_robustness_run(b):
                released_for_resume = True

            check_true(
                "robustness_results_single_fingerprint_writer_and_shared_model_writer_lock",
                same_run_blocked and same_model_blocked and independent_run_allowed,
            )
            check_true(
                "robustness_shared_model_lock_blocks_other_process_and_releases_for_resume",
                outside_process_blocked and released_for_resume and interrupted_was_observed,
            )
            check_true(
                "robustness_lock_files_outside_mutable_result_and_model_directories",
                not result_root(a).exists() and not model_root(a).exists(),
            )

        resume_root = lock_test_root / "resume" / "same_scientific_run"
        resume_root.mkdir(parents=True)
        current_contract = {"fingerprint": "same_scientific_run"}
        historical_manifest = resume_root / robustness_runtime.MANIFEST_FILENAME
        resume_valid = []
        for state in ("RUNNING", "FAILED"):
            robustness_runtime._write_json(historical_manifest, {
                "status": state, "contract": dict(current_contract),
            })
            robustness_runtime._check_resumable_robustness_manifest(
                resume_root, contract=current_contract,
            )
            resume_valid.append(state)
        rejection_cases = (
            {"status": "COMPLETED", "contract": dict(current_contract)},
            {"status": "RUNNING", "contract": {"fingerprint": "foreign_run"}},
        )
        rejected = []
        for previous_manifest in rejection_cases:
            robustness_runtime._write_json(historical_manifest, previous_manifest)
            try:
                robustness_runtime._check_resumable_robustness_manifest(
                    resume_root, contract=current_contract,
                )
            except RuntimeError:
                rejected.append(True)
            else:
                rejected.append(False)
        historical_manifest.write_text("{not-valid-json", encoding="utf-8")
        try:
            robustness_runtime._check_resumable_robustness_manifest(
                resume_root, contract=current_contract,
            )
        except RuntimeError:
            malformed_rejected = True
        else:
            malformed_rejected = False
        check_true(
            "robustness_interrupted_same_identity_can_resume_without_rewriting_complete",
            resume_valid == ["RUNNING", "FAILED"],
        )
        check_true(
            "robustness_invalid_completed_or_foreign_or_malformed_manifest_is_not_overwritten",
            rejected == [True, True] and malformed_rejected,
        )

    # AI: Viewing a committed result must not silently break the durable SHA
    # evidence used by the canonical REUSE validator.  Exercise the real viewer
    # with isolated files, including historical presentation upgrades and drift.
    with tempfile.TemporaryDirectory(prefix="robustness_latest_readonly_") as temp_dir:
        viewer_root = Path(temp_dir)
        output_root = viewer_root / "results"
        fingerprint = "synthetic_completed_fingerprint"
        run_root = output_root / fingerprint
        run_root.mkdir(parents=True)
        report_path = run_root / robustness_runtime.REPORT_FILENAME
        summary_path = run_root / robustness_runtime.SUMMARY_FILENAME
        manifest_path = run_root / robustness_runtime.MANIFEST_FILENAME
        latest_path = output_root / robustness_runtime.LATEST_FILENAME
        seed_path = run_root / robustness_runtime.SEED_RESULTS_FILENAME
        yearly_path = run_root / robustness_runtime.SEED_YEARLY_RESULTS_FILENAME
        seed_path.write_text("arm_id,seed,direct_selection_r\nA,1,0.2\n", encoding="utf-8")
        yearly_path.write_text("arm_id,seed,year\nA,1,2020\n", encoding="utf-8")
        summary_payload = {
            "schema_version": robustness_runtime.ROBUSTNESS_SCHEMA_VERSION,
            "contract": {"fingerprint": fingerprint},
            "paired_comparisons": [],
            "direct_selection_r_same_seed_comparison": [],
            "selection_r_to_strategy_same_seed_translation": [],
            "common_strategy_report": [],
        }
        summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
        report_path.write_text("preserved completed Markdown\n", encoding="utf-8")
        latest_path.write_text(json.dumps({
            "fingerprint": fingerprint,
            "summary_path": f"results/{fingerprint}/{robustness_runtime.SUMMARY_FILENAME}",
            "report_path": f"results/{fingerprint}/{robustness_runtime.REPORT_FILENAME}",
        }), encoding="utf-8")
        with patch.object(robustness_runtime, "PROJECT_ROOT", viewer_root), patch.object(
            robustness_runtime, "get_strategy_multi_seed_robustness_settings",
            return_value=SimpleNamespace(output_root="results"),
        ):
            def _commit_viewer_fixture():
                artifacts = robustness_runtime._build_durable_result_artifacts(run_root)
                manifest_path.write_text(json.dumps({
                    "status": "COMPLETED",
                    "contract": {"fingerprint": fingerprint},
                    "durable_artifacts": artifacts,
                }), encoding="utf-8")
                return artifacts

            initial_artifacts = _commit_viewer_fixture()
            original_bytes = {
                file.name: file.read_bytes() for file in
                (report_path, summary_path, manifest_path, seed_path, yearly_path)
            }
            with patch.object(robustness_runtime, "_print_report_tables") as printed, patch.object(
                robustness_runtime, "_write_json", side_effect=AssertionError("viewer wrote JSON")
            ), patch.object(
                robustness_runtime, "atomic_write_text", side_effect=AssertionError("viewer wrote Markdown")
            ):
                robustness_runtime.show_latest_multi_seed_robustness_report(
                    robustness_id="synthetic"
                )
            check_true(
                "robustness_latest_view_does_not_mutate_completed_durable_evidence",
                printed.call_count == 1
                and all((run_root / name).read_bytes() == data for name, data in original_bytes.items())
                and robustness_runtime._validate_durable_result_artifacts(run_root, initial_artifacts),
            )

            # An old schema may be normalized for display, but must not be
            # persisted by a read-only menu action (which has no write authority).
            old_summary = {"schema_version": 1, "contract": {"fingerprint": fingerprint}}
            summary_path.write_text(json.dumps(old_summary), encoding="utf-8")
            old_artifacts = _commit_viewer_fixture()
            upgraded_summary = {**old_summary, "display_upgrade_only": True}
            with patch.object(
                robustness_runtime, "_load_seed_results",
                return_value=pd.DataFrame([{"direct_selection_r": 0.2}]),
            ), patch.object(
                robustness_runtime, "_load_seed_yearly_results", return_value=pd.DataFrame()
            ), patch.object(
                robustness_runtime, "_upgrade_derived_report_summary", return_value=upgraded_summary
            ) as upgrader, patch.object(
                robustness_runtime, "_print_report_tables"
            ) as rendered, patch.object(
                robustness_runtime, "_write_json", side_effect=AssertionError("viewer wrote JSON")
            ), patch.object(
                robustness_runtime, "atomic_write_text", side_effect=AssertionError("viewer wrote Markdown")
            ):
                robustness_runtime.show_latest_multi_seed_robustness_report(
                    robustness_id="synthetic"
                )
            check_true(
                "robustness_latest_legacy_upgrade_is_in_memory_only",
                upgrader.call_count == 1
                and rendered.call_args.args[0] == upgraded_summary
                and json.loads(summary_path.read_text(encoding="utf-8")) == old_summary
                and robustness_runtime._validate_durable_result_artifacts(run_root, old_artifacts),
            )

            # A stale pointer, an interrupted run, and a modified report must
            # not be presented as a validated completed result.
            report_path.write_text("modified without updating manifest\n", encoding="utf-8")
            with patch.object(robustness_runtime, "_print_report_tables") as printed:
                try:
                    robustness_runtime.show_latest_multi_seed_robustness_report(robustness_id="synthetic")
                except RuntimeError:
                    corrupted_rejected = True
                else:
                    corrupted_rejected = False
            check_true(
                "robustness_latest_rejects_changed_report_without_manifest_commit",
                corrupted_rejected and not printed.called,
            )
            report_path.write_text("preserved completed Markdown\n", encoding="utf-8")
            _commit_viewer_fixture()
            incomplete = json.loads(manifest_path.read_text(encoding="utf-8"))
            incomplete["status"] = "RUNNING"
            manifest_path.write_text(json.dumps(incomplete), encoding="utf-8")
            try:
                robustness_runtime.show_latest_multi_seed_robustness_report(robustness_id="synthetic")
            except RuntimeError:
                interrupted_rejected = True
            else:
                interrupted_rejected = False
            check_true("robustness_latest_rejects_incomplete_manifest", interrupted_rejected)
            _commit_viewer_fixture()
            latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
            latest_payload["summary_path"] = "results/foreign/robustness_summary.json"
            latest_path.write_text(json.dumps(latest_payload), encoding="utf-8")
            try:
                robustness_runtime.show_latest_multi_seed_robustness_report(robustness_id="synthetic")
            except ValueError:
                wrong_pointer_rejected = True
            else:
                wrong_pointer_rejected = False
            check_true("robustness_latest_rejects_foreign_result_pointer", wrong_pointer_rejected)
            latest_payload["summary_path"] = (
                f"results/{fingerprint}/{robustness_runtime.SUMMARY_FILENAME}"
            )
            latest_path.write_text(json.dumps(latest_payload), encoding="utf-8")
            with robustness_runtime.exclusive_file_lock(
                robustness_runtime.sibling_publication_lock_path(run_root)
            ):
                try:
                    robustness_runtime.show_latest_multi_seed_robustness_report(
                        robustness_id="synthetic"
                    )
                except RuntimeError:
                    publishing_reader_blocked = True
                else:
                    publishing_reader_blocked = False
            check_true(
                "robustness_latest_reader_shares_exact_result_publication_lock",
                publishing_reader_blocked,
            )

    # Completed evidence is immutable even when REUSE is explicitly disabled.
    # A reset/replay knob is not authorization to overwrite an existing run.
    with tempfile.TemporaryDirectory(prefix="robustness_commit_boundaries_") as temp_dir:
        commit_root = Path(temp_dir)
        completed_root = commit_root / "completed" / "run-1"
        completed_root.mkdir(parents=True)
        recorded = {"status": "COMPLETED", "contract": {"fingerprint": "run-1"}}
        completed_manifest = completed_root / robustness_runtime.MANIFEST_FILENAME
        robustness_runtime._write_json(completed_manifest, recorded)
        original_completed = completed_manifest.read_bytes()
        with patch.object(robustness_runtime, "_run_root", return_value=completed_root), patch.object(
            robustness_runtime, "_model_work_root", return_value=commit_root / "models"
        ), patch.object(
            robustness_runtime, "_materialize_registered_strategy_only_scientific_reuse"
        ) as cannot_start_producer:
            try:
                robustness_runtime._execute_robustness_under_lock(
                    cfg=SimpleNamespace(reuse_completed=False),
                    settings=SimpleNamespace(), status={},
                    contract={"fingerprint": "run-1", "resolved_seeds": []},
                    fixed_arms=(), stochastic_arms=(), model_arms=(),
                    strategy_only_arms=(), registered_scientific_reuse={},
                    benchmark_bindings={}, comparison_start="2021-01-01",
                    comparison_end="2021-12-31",
                )
            except RuntimeError:
                completed_protected = True
            else:
                completed_protected = False
        check_true(
            "robustness_reuse_off_does_not_overwrite_completed_or_start_producer",
            completed_protected and not cannot_start_producer.called
            and completed_manifest.read_bytes() == original_completed,
        )
        with patch.object(robustness_runtime, "PROJECT_ROOT", commit_root):
            latest_target = commit_root / "latest.json"
            original_write = robustness_runtime._write_json
            def _reject_latest_only(path, payload):
                if Path(path) == latest_target:
                    raise OSError("injected latest pointer failure")
                return original_write(path, payload)
            with patch.object(robustness_runtime, "_write_json", side_effect=_reject_latest_only):
                try:
                    robustness_runtime._publish_completed_robustness_latest(
                        run_root=completed_root, latest_path=latest_target,
                        contract={"fingerprint": "run-1"},
                        attribution_index_path=None,
                    )
                except OSError:
                    pointer_failure_observed = True
                else:
                    pointer_failure_observed = False
        check_true(
            "robustness_latest_failure_never_demotes_completed_manifest",
            pointer_failure_observed and completed_manifest.read_bytes() == original_completed
            and not latest_target.exists(),
        )

    # The latest pointer may advance to another complete fingerprint before the
    # reader acquires the old run lock. It must not show a stale 'latest'.
    with tempfile.TemporaryDirectory(prefix="robustness_latest_race_") as temp_dir:
        latest_root = Path(temp_dir)
        run_root = latest_root / "results" / "first"
        run_root.mkdir(parents=True)
        latest_path = latest_root / "results" / robustness_runtime.LATEST_FILENAME
        (run_root / robustness_runtime.SUMMARY_FILENAME).write_text(
            json.dumps({"contract": {"fingerprint": "first"}}), encoding="utf-8",
        )
        (run_root / robustness_runtime.REPORT_FILENAME).write_text("report", encoding="utf-8")
        for key, filename in robustness_runtime.DURABLE_RESULT_FILENAMES.items():
            (run_root / filename).touch(exist_ok=True)
        initial_pointer = {
            "fingerprint": "first",
            "summary_path": f"results/first/{robustness_runtime.SUMMARY_FILENAME}",
            "report_path": f"results/first/{robustness_runtime.REPORT_FILENAME}",
        }
        latest_path.write_text(json.dumps(initial_pointer), encoding="utf-8")
        (run_root / robustness_runtime.MANIFEST_FILENAME).write_text(json.dumps({
            "status": "COMPLETED", "contract": {"fingerprint": "first"},
            "durable_artifacts": {},
        }), encoding="utf-8")
        original_reader = robustness_runtime._read_json
        replaced = [False]
        def _advance_during_read(path):
            parsed = original_reader(path)
            if Path(path) == run_root / robustness_runtime.MANIFEST_FILENAME and not replaced[0]:
                replaced[0] = True
                latest_path.write_text(json.dumps({**initial_pointer, "fingerprint": "second"}), encoding="utf-8")
            return parsed
        with patch.object(robustness_runtime, "PROJECT_ROOT", latest_root), patch.object(
            robustness_runtime, "get_strategy_multi_seed_robustness_settings",
            return_value=SimpleNamespace(output_root="results"),
        ), patch.object(robustness_runtime, "_read_json", side_effect=_advance_during_read), patch.object(
            robustness_runtime, "_print_report_tables"
        ) as printed:
            try:
                robustness_runtime.show_latest_multi_seed_robustness_report(robustness_id="test")
            except RuntimeError:
                stale_pointer_rejected = True
            else:
                stale_pointer_rejected = False
        check_true(
            "robustness_latest_reader_rejects_concurrent_pointer_advance",
            stale_pointer_rejected and replaced[0] and not printed.called,
        )


def append_latest_publication_checks(*, check_true):
    # Strategy Compare's historical latest/{md,json,manifest} files are legacy
    # display copies, never the unique scientific run. Prepare all copies and
    # serialize writer commits; on a recoverable fault restore old bytes.
    from services.research import strategy_comparison as comparison_writer
    with tempfile.TemporaryDirectory(prefix="comparison_latest_concurrency_") as temp_dir:
        base = Path(temp_dir)
        newest = base / "new_run"
        newest.mkdir()
        latest = base / "latest"
        latest.mkdir()
        filenames = ("strategy_comparison.md", "strategy_comparison.json", "manifest.json")
        old_bytes = {}
        for filename in filenames:
            (newest / filename).write_bytes(f"new/{filename}".encode("utf-8"))
            data = f"old/{filename}".encode("utf-8")
            (latest / filename).write_bytes(data)
            old_bytes[filename] = data
        from core import file_integrity as latest_integrity
        original_replace = latest_integrity.atomic_replace_with_retry
        rejected_once = [False]
        def _interrupt_manifest(src, dst, **kwargs):
            if Path(dst).name == "manifest.json" and not rejected_once[0]:
                rejected_once[0] = True
                raise OSError("synthetic final manifest publication interrupted")
            return original_replace(src, dst, **kwargs)
        with patch.object(latest_integrity, "atomic_replace_with_retry", side_effect=_interrupt_manifest):
            try:
                comparison_writer._publish_strategy_latest_snapshot(
                    latest, report_path=newest / filenames[0],
                    json_path=newest / filenames[1], manifest_path=newest / filenames[2],
                )
            except OSError:
                interrupted_latest = True
            else:
                interrupted_latest = False
        check_true(
            "strategy_compare_latest_copy_failure_restores_previous_complete_view",
            interrupted_latest and rejected_once[0]
            and all((latest / name).read_bytes() == old_bytes[name] for name in filenames),
        )
        with latest_integrity.exclusive_file_lock(
            latest_integrity.sibling_publication_lock_path(latest)
        ):
            try:
                comparison_writer._publish_strategy_latest_snapshot(
                    latest, report_path=newest / filenames[0],
                    json_path=newest / filenames[1], manifest_path=newest / filenames[2],
                )
            except RuntimeError:
                concurrent_latest_rejected = True
            else:
                concurrent_latest_rejected = False
        check_true(
            "strategy_compare_latest_uses_one_stable_publication_lock",
            concurrent_latest_rejected
            and all((latest / name).read_bytes() == old_bytes[name] for name in filenames),
        )
        comparison_writer._publish_strategy_latest_snapshot(
            latest, report_path=newest / filenames[0],
            json_path=newest / filenames[1], manifest_path=newest / filenames[2],
        )
        check_true(
            "strategy_compare_latest_commits_all_three_original_format_files",
            all((latest / name).read_bytes() == (newest / name).read_bytes() for name in filenames),
        )

    # A binary latest publisher must preserve the original text writer's
    # platform-dependent newline bytes (Windows CRLF, POSIX LF).
    with patch.object(latest_integrity.os, "linesep", "\r\n"):
        simulated_windows_md = latest_integrity.text_publication_bytes("a\nb\n")
        simulated_windows_json = latest_integrity.json_publication_bytes({"a": 1})
    check_true(
        "latest_binary_publisher_preserves_windows_legacy_newline_bytes",
        simulated_windows_md == b"a\r\nb\r\n"
        and b"\r\n" in simulated_windows_json
        and b"\r\r\n" not in simulated_windows_json,
    )

    # Legal publication filenames may themselves end with ".backup".  The
    # shared owner must keep staging and rollback storage separate, otherwise
    # a failed second replace can silently corrupt the first original file.
    with tempfile.TemporaryDirectory(prefix="latest_backup_name_collision_") as temp_dir:
        collision_latest = Path(temp_dir) / "latest"
        collision_latest.mkdir()
        originals = {"foo": b"original-a", "foo.backup": b"original-b"}
        for name, content in originals.items():
            (collision_latest / name).write_bytes(content)
        real_replace = latest_integrity.atomic_replace_with_retry
        def _fail_second_colliding_name(source, target, **kwargs):
            if Path(target).name == "foo.backup":
                raise OSError("synthetic rollback failure injection")
            return real_replace(source, target, **kwargs)
        with patch.object(
            latest_integrity, "atomic_replace_with_retry",
            side_effect=_fail_second_colliding_name,
        ):
            try:
                latest_integrity.publish_latest_snapshot(
                    collision_latest, (("foo", b"new-a"), ("foo.backup", b"new-b")),
                )
            except OSError:
                collision_failure_observed = True
            else:
                collision_failure_observed = False
        check_true(
            "latest_publisher_filename_backup_collision_cannot_corrupt_rollback",
            collision_failure_observed and all(
                (collision_latest / name).read_bytes() == content
                for name, content in originals.items()
            ),
        )
        latest_integrity.publish_latest_snapshot(
            collision_latest, (("foo", b"new-a"), ("foo.backup", b"new-b")),
        )
        check_true(
            "latest_publisher_filename_backup_collision_commits_exact_bytes",
            (collision_latest / "foo").read_bytes() == b"new-a"
            and (collision_latest / "foo.backup").read_bytes() == b"new-b",
        )

    # The same core latest publisher must also own presentation-only Audit,
    # OOS/Rolling consistency, and historical Runtime Promotion Gate copies.
    # Their old JSON/Markdown bytes must remain unchanged.
    from core import file_integrity as latest_integrity
    from services.audit import reusable_report as audit_writer
    from services.research import strategy_compare_application as consistency_writer
    from services.research import runtime_integration_gate as gate_writer
    with tempfile.TemporaryDirectory(prefix="shared_latest_publication_") as temp_dir:
        temp_root = Path(temp_dir)
        audit_def = SimpleNamespace(
            output_subdir="synthetic_audit", report_id="synthetic-audit",
            report_type="read_only",
        )
        with redirect_stdout(io.StringIO()):
            audit_result = audit_writer.persist_reusable_report(
                definition=audit_def, project_root=temp_root,
                payload={"value": 1}, console_text="console", markdown_text="audit report",
            )
        audit_root = temp_root / audit_writer.AUDIT_OUTPUT_ROOT / "synthetic_audit"
        audit_latest = audit_root / "latest"
        audit_run = temp_root / audit_result["artifacts"]["json"]
        check_true(
            "audit_latest_uses_single_owner_and_keeps_historical_json_bytes",
            (audit_latest / "audit.json").read_bytes() == audit_run.read_bytes()
            and (audit_latest / "audit.md").read_bytes() == (audit_run.parent / "audit.md").read_bytes(),
        )
        # Same-second audit reports must not overwrite immutable historical
        # evidence even when two workers share the same timestamp.
        from datetime import datetime, timezone
        fixed_now = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
        collision_def = SimpleNamespace(
            output_subdir="collision_audit", report_id="collision-audit",
            report_type="read_only",
        )
        with patch.object(
            audit_writer, "datetime", SimpleNamespace(now=lambda _tz: fixed_now)
        ), redirect_stdout(io.StringIO()):
            first = audit_writer.persist_reusable_report(
                definition=collision_def, project_root=temp_root, payload={"version": 1},
                console_text="first", markdown_text="first markdown",
            )
            first_path = temp_root / first["artifacts"]["json"]
            first_bytes = first_path.read_bytes()
            second = audit_writer.persist_reusable_report(
                definition=collision_def, project_root=temp_root, payload={"version": 2},
                console_text="second", markdown_text="second markdown",
            )
        second_path = temp_root / second["artifacts"]["json"]
        check_true(
            "audit_same_second_runs_keep_independent_immutable_result_directories",
            first_path != second_path and first_path.read_bytes() == first_bytes
            and json.loads(second_path.read_text(encoding="utf-8"))["version"] == 2
            and (second_path.parent.parent / "latest" / "audit.json").read_bytes()
                == second_path.read_bytes(),
        )

        consistent_payload = {
            "schema_version": 1, "suite_id": "test", "arm_rows": [], "contrast_rows": [],
        }
        with patch.object(
            consistency_writer, "build_strategy_oos_rolling_consistency",
            return_value=consistent_payload,
        ), patch.object(
            consistency_writer, "render_strategy_oos_rolling_consistency",
            return_value="synthetic markdown",
        ), redirect_stdout(io.StringIO()):
            consistency_writer.materialize_strategy_oos_rolling_consistency(
                {}, {}, project_root=temp_root,
            )
        consistency_dir = temp_root / "outputs/strategy_compare/extending_window/consistency/latest"
        check_true(
            "strategy_consistency_latest_shares_one_transaction_and_json_format",
            (consistency_dir / "strategy_consistency.md").read_bytes()
            == latest_integrity.text_publication_bytes("synthetic markdown\n")
            and (consistency_dir / "strategy_consistency.json").read_bytes()
            == latest_integrity.json_publication_bytes(consistent_payload),
        )
        gate_root = temp_root / "gate"
        gate_cfg = SimpleNamespace(output_root="gate")
        gate_payload = {"fingerprint": "gate-case", "status": "NO_GO"}
        with patch.object(
            gate_writer, "get_strategy_runtime_integration_settings", return_value=gate_cfg,
        ), patch.object(
            gate_writer, "collect_runtime_integration_status", return_value=gate_payload,
        ), patch.object(
            gate_writer, "render_runtime_integration_report", return_value="gate markdown",
        ), patch.object(gate_writer, "_print_console_summary"), redirect_stdout(io.StringIO()):
            gate_writer.run_runtime_integration_gate(project_root=temp_root)
            gate_latest = gate_root / "latest"
            gate_original = {
                name: (gate_latest / name).read_bytes()
                for name in (gate_writer.RESULT_FILENAME, gate_writer.REPORT_FILENAME)
            }
            with latest_integrity.exclusive_file_lock(
                latest_integrity.sibling_publication_lock_path(gate_latest)
            ):
                try:
                    gate_writer.show_latest_runtime_integration_report(project_root=temp_root)
                except RuntimeError:
                    gate_reader_rejected = True
                else:
                    gate_reader_rejected = False
            with redirect_stdout(io.StringIO()):
                restored_gate = gate_writer.show_latest_runtime_integration_report(
                    project_root=temp_root
                )
        check_true(
            "runtime_gate_latest_reader_obeys_same_publisher_lock_and_legacy_bytes",
            gate_reader_rejected and restored_gate == gate_payload
            and all((gate_latest / name).read_bytes() == raw
                    for name, raw in gate_original.items())
            and gate_original[gate_writer.REPORT_FILENAME]
                == latest_integrity.text_publication_bytes("gate markdown"),
        )

    # The retained Runtime Promotion latest view also uses the canonical
    # binary publisher. Its completed run artifacts and bytes are immutable.
    from services.breakout_quality import runtime_promotion as promotion_writer
    from services.audit import strategy_compare_source as audit_compare_reader
    with tempfile.TemporaryDirectory(prefix="promotion_latest_contract_") as temp_dir:
        synthetic_root = Path(temp_dir)
        history_parent = synthetic_root / "promotion"
        timestamp = "20261020_142301"
        first_run = latest_integrity.reserve_immutable_output_dir(history_parent, timestamp)
        original_record = first_run / "runtime_promotion.json"
        original_record.write_bytes(b"first\r\n")
        next_run = latest_integrity.reserve_immutable_output_dir(history_parent, timestamp)
        check_true(
            "runtime_promotion_same_second_history_cannot_overwrite_prior_run",
            first_run != next_run and original_record.read_bytes() == b"first\r\n"
            and next_run.parent == history_parent,
        )
        md = next_run / "runtime_promotion.md"
        js = next_run / "runtime_promotion.json"
        md.write_bytes(b"report\r\n")
        js.write_bytes(b"{\r\n  \"status\": \"PROMOTED\"\r\n}\r\n")
        latest = history_parent / "latest"
        latest_integrity.publish_latest_snapshot(latest, (
            (md.name, md.read_bytes()), (js.name, js.read_bytes()),
        ))
        check_true(
            "runtime_promotion_latest_keeps_original_run_bytes_and_shared_owner",
            all((latest / file.name).read_bytes() == file.read_bytes()
                for file in (md, js))
            and "publish_latest_snapshot" in inspect.getsource(promotion_writer.apply_or_refresh_runtime_promotion),
        )
        # A canonical Audit reading mutable latest must use the same lock as
        # Strategy Compare's publisher; immutable pinned runs remain unaffected.
        compare_latest = synthetic_root / "strategy_compare" / "latest"
        compare_latest.mkdir(parents=True)
        with patch.object(
            audit_compare_reader, "get_strategy_comparison_settings",
            return_value=SimpleNamespace(output_root="strategy_compare"),
        ):
            with latest_integrity.exclusive_file_lock(
                latest_integrity.sibling_publication_lock_path(compare_latest)
            ):
                try:
                    audit_compare_reader.load_strategy_compare_source(
                        synthetic_root, profile_id="test"
                    )
                except RuntimeError as exc:
                    audited_latest_is_locked = "發布" in str(exc)
                else:
                    audited_latest_is_locked = False
        check_true(
            "strategy_compare_audit_latest_reader_respects_canonical_writer_lock",
            audited_latest_is_locked,
        )

    # Publication-complete must mean Manifest-last + canonical pair bytes,
    # not just a run JSON claiming status=COMPLETED. This is the same
    # integrity owner used by Strategy Compare REUSE, not a second gate.
    from services.research.strategy_compare_pair_integrity import (
        build_completed_pair_file_integrity,
    )
    from services.research import runtime_integration_gate as gate_evidence
    with tempfile.TemporaryDirectory(prefix="strategy_gate_publication_boundary_") as temp_dir:
        root = Path(temp_dir)
        source_row = {
            "filter_id": "synthetic", "model_architecture": "test",
            "experiment_profile": "test", "threshold": 0.1,
            "score_source": "selection_point_in_time",
        }
        on_fields = {
            "arm_id": "on", "param_source": "p", "rule_policy": "all_off",
            "dl_enabled": True, "dl_id": "q", "dl_runtime_mode": "score",
            "dl_runtime_options": {},
        }
        off_fields = {
            "arm_id": "off", "param_source": "p", "rule_policy": "all_off",
            "dl_enabled": False, "dl_id": None, "dl_runtime_mode": "",
        }
        source = SimpleNamespace(as_dict=lambda: source_row)
        settings = SimpleNamespace(output_root="comparison", dl_sources={"q": source})
        on_arm = SimpleNamespace(arm_id="on", dl_id="q", as_dict=lambda: on_fields)
        off_arm = SimpleNamespace(arm_id="off", as_dict=lambda: off_fields)
        runs = root / "comparison" / "runs"
        run_dir = runs / "fixed_run"
        latest = root / "comparison" / "latest"
        run_dir.mkdir(parents=True)
        latest.mkdir()
        result = {
            "status": "COMPLETED", "config_fingerprint": "0123456789ab",
            "settings": {"arms": {"on": on_fields, "off": off_fields},
                         "dl_sources": {"q": source_row}},
            "pairs": {"group": {"metadata": {"synthetic": True}}},
        }
        run_json = run_dir / "strategy_comparison.json"
        latest_json = latest / "strategy_comparison.json"
        run_json.write_text(json.dumps(result), encoding="utf-8")
        latest_json.write_text(json.dumps(result), encoding="utf-8")

        def _gate_evidence():
            with patch.object(
                gate_evidence, "resolve_strategy_comparison_result_search_roots",
                return_value=("comparison",),
            ):
                return gate_evidence._find_strategy_result_evidence(
                    root, settings, candidate=on_arm, baseline=off_arm,
                )

        try:
            _gate_evidence()
        except FileNotFoundError:
            incomplete_rejected = True
        else:
            incomplete_rejected = False
        check_true(
            "strategy_gate_does_not_trust_uncommitted_completed_json",
            incomplete_rejected,
        )
        pair_dir = run_dir / "pairs" / "group"
        pair_dir.mkdir(parents=True)
        (pair_dir / "strategy_comparison.json").write_text(
            json.dumps(result["pairs"]["group"]), encoding="utf-8"
        )
        (pair_dir / "strategy_comparison.md").write_text(
            "historical pair", encoding="utf-8"
        )
        sidecar = pair_dir / "orderable_candidates.csv"
        sidecar.write_text("ticker,score\n2330,0.2\n", encoding="utf-8")
        record = build_completed_pair_file_integrity({"group": pair_dir})
        result["pair_file_integrity"] = record
        for result_path in (run_json, latest_json):
            result_path.write_text(json.dumps(result), encoding="utf-8")
        manifest = {
            "config_fingerprint": result["config_fingerprint"],
            "run_dir": str(run_dir.relative_to(root)),
            "pair_file_integrity": record,
        }
        for manifest_path in (run_dir / "manifest.json", latest / "manifest.json"):
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        path, _payload = _gate_evidence()
        check_true(
            "strategy_gate_committed_latest_matches_canonical_run_and_pair_integrity",
            path == latest_json,
        )
        with latest_integrity.exclusive_file_lock(
            latest_integrity.sibling_publication_lock_path(latest)
        ):
            try:
                _gate_evidence()
            except RuntimeError:
                latest_guarded = True
            else:
                latest_guarded = False
        check_true(
            "strategy_gate_cannot_read_during_latest_publication",
            latest_guarded,
        )
        latest_json.write_text(json.dumps({**result, "status": "FAILED"}), encoding="utf-8")
        path, _payload = _gate_evidence()
        check_true(
            "strategy_gate_rejects_mixed_latest_and_uses_only_committed_run",
            path == run_json,
        )
        sidecar.write_text("ticker,score\n2330,0.9\n", encoding="utf-8")
        try:
            _gate_evidence()
        except FileNotFoundError:
            modified_pair_rejected = True
        else:
            modified_pair_rejected = False
        check_true(
            "strategy_gate_does_not_trust_changed_completed_pair_sidecars",
            modified_pair_rejected,
        )

    # RUN (not just Audit) must allocate immutable evidence even when the
    # same configuration executes twice within a single wall-clock second.
    with tempfile.TemporaryDirectory(prefix="strategy_same_second_runs_") as temp_dir:
        base = Path(temp_dir)
        fixed_now = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
        settings = SimpleNamespace(
            output_root="output", enabled_arms=(SimpleNamespace(arm_id="test"),),
        )
        with patch.object(comparison_writer, "get_taipei_now", return_value=fixed_now):
            suggested, _latest = comparison_writer._run_directory(
                root=base, settings=settings, fingerprint="0123456789ab",
            )
            first = latest_integrity.reserve_immutable_output_dir(
                suggested.parent, suggested.name,
            )
            second = latest_integrity.reserve_immutable_output_dir(
                suggested.parent, suggested.name,
            )
        check_true(
            "strategy_comparison_same_second_immutable_run_names_are_unique",
            first != second and first.name == suggested.name
            and second.name.startswith(suggested.name + "_")
            and "reserve_immutable_output_dir(run_dir.parent, run_dir.name)"
                in inspect.getsource(comparison_writer.run_strategy_comparison),
        )
        gate_cfg = SimpleNamespace(output_root="gate")
        with patch.object(
            gate_evidence, "get_strategy_runtime_integration_settings",
            return_value=gate_cfg,
        ), patch.object(
            gate_evidence, "collect_runtime_integration_status",
            return_value={"fingerprint": "0123456789ab", "status": "BLOCKED"},
        ), patch.object(
            gate_evidence, "render_runtime_integration_report", return_value="report",
        ), patch.object(gate_evidence, "_print_console_summary"), patch.object(
            gate_evidence, "datetime"
        ) as fake_clock, redirect_stdout(io.StringIO()):
            fake_clock.now.return_value.astimezone.return_value.strftime.return_value = "20261021_120000"
            gate_evidence.run_runtime_integration_gate(project_root=base)
            gate_evidence.run_runtime_integration_gate(project_root=base)
        check_true(
            "runtime_gate_same_second_immutable_reports_are_not_overwritten",
            len(list((base / "gate" / "runs").iterdir())) == 2,
        )
