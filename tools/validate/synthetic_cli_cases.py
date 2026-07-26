from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import importlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

from config.breakout_quality_experiments import (
    ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
    TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
)

from .checks import add_check


def _capture_stdout(func, *args, **kwargs):
    buf = StringIO()
    with redirect_stdout(buf):
        rc = func(*args, **kwargs)
    return rc, buf.getvalue()


def _capture_stderr(func, *args, **kwargs):
    buf = StringIO()
    with redirect_stderr(buf):
        rc = func(*args, **kwargs)
    return rc, buf.getvalue()


def _assert_value_error(results, category, case_id, metric_name, func, expected_substring):
    try:
        func()
    except ValueError as exc:
        add_check(results, category, case_id, metric_name, True, expected_substring in str(exc))
    else:
        add_check(results, category, case_id, metric_name, True, False)


def validate_dataset_cli_contract_case(_base_params):
    case_id = "CLI_DATASET_WRAPPER_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    app_breakout_quality = importlib.import_module("apps.breakout_quality")
    moment_contract_module = importlib.import_module(
        "filters.breakout_quality.moment_contract"
    )
    app_ml_optimizer = importlib.import_module("apps.ml_optimizer")
    app_portfolio_sim = importlib.import_module("apps.portfolio_sim")
    app_vip_scanner = importlib.import_module("apps.vip_scanner")
    scanner_main_module = importlib.import_module("tools.scanner.main")
    validate_cli_module = importlib.import_module("tools.validate.cli")

    rc, help_text = _capture_stdout(
        app_breakout_quality.main,
        ["apps/breakout_quality.py", "--help"],
    )
    add_check(results, "cli_contract", case_id, "breakout_quality_app_help_rc", 0, rc)
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_app_help_usage",
        True,
        "用法: python apps/breakout_quality.py [menu|workflow|<command>] [options]" in help_text,
    )
    for command in ("menu", "workflow", "build-dataset", "build-pretrain-dataset", "pretrain", "train", "export-scores", "report", "evaluate"):
        add_check(
            results,
            "cli_contract",
            case_id,
            f"breakout_quality_app_help_lists_{command.replace('-', '_')}",
            True,
            command in help_text,
        )

    with patch("apps.breakout_quality.is_interactive_console", return_value=False):
        rc, no_arg_text = _capture_stdout(
            app_breakout_quality.main,
            ["apps/breakout_quality.py"],
        )
    add_check(results, "cli_contract", case_id, "breakout_quality_noninteractive_no_arg_help_rc", 0, rc)
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_noninteractive_no_arg_help",
        True,
        "Breakout quality filter" in no_arg_text,
    )

    with (
        patch("apps.breakout_quality.is_interactive_console", return_value=True),
        patch("apps.breakout_quality._run_interactive_menu", return_value=31) as mocked_menu,
    ):
        rc = app_breakout_quality.main(["apps/breakout_quality.py"])
    add_check(results, "cli_contract", case_id, "breakout_quality_interactive_no_arg_menu_rc", 31, rc)
    add_check(results, "cli_contract", case_id, "breakout_quality_interactive_no_arg_menu_called", 1, mocked_menu.call_count)

    with (
        patch("apps.breakout_quality.is_interactive_console", return_value=True),
        patch("apps.breakout_quality._run_interactive_menu", return_value=32) as mocked_menu,
    ):
        rc = app_breakout_quality.main(["apps/breakout_quality.py", "menu"])
    add_check(results, "cli_contract", case_id, "breakout_quality_explicit_menu_rc", 32, rc)
    add_check(results, "cli_contract", case_id, "breakout_quality_explicit_menu_called", 1, mocked_menu.call_count)

    fake_workflow_args = SimpleNamespace(filter_id="synthetic_quality")
    with (
        patch("apps.breakout_quality._parse_workflow_args", return_value=fake_workflow_args) as mocked_parse,
        patch("apps.breakout_quality._run_workflow", return_value=33) as mocked_workflow,
    ):
        rc = app_breakout_quality.main(
            ["apps/breakout_quality.py", "workflow", "--filter-id", "synthetic_quality"]
        )
    add_check(results, "cli_contract", case_id, "breakout_quality_workflow_rc", 33, rc)
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_workflow_parse_argv",
        ["--filter-id", "synthetic_quality"],
        mocked_parse.call_args.args[0],
    )
    add_check(results, "cli_contract", case_id, "breakout_quality_workflow_called", 1, mocked_workflow.call_count)

    workflow_args = SimpleNamespace(
        filter_id="synthetic_quality",
        dataset="full",
        max_tickers=0,
        rebuild_dataset=False,
        epochs=3,
        batch_size=32,
        evaluation_batch_size=128,
        evaluation_workers=4,
        parallel_split_evaluation=True,
        train_prefetch_batches=0,
        preload_feature_bank=True,
        experiment_profile=ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
        lr=0.001,
        weight_decay=0.0001,
        gradient_clip_norm=1.0,
        final_refit_mode="matched_optimizer_steps",
        class_weight_mode="none",
        time_weight_mode="none",
        seed=42,
        fixed_threshold=0.5,
        inner_validation_months=24,
        early_stopping_patience=5,
        early_stopping_min_delta=0.0,
        use_inner_validation=True,
        device="cpu",
        mixed_precision=False,
        mixed_precision_dtype="float16",
        deterministic_algorithms=True,
        allow_tf32=False,
        evaluate_oos=True,
    )
    workflow_calls = []
    fake_train_module = SimpleNamespace(
        parse_args=lambda argv: SimpleNamespace(argv=list(argv)),
        validate_training_args=lambda parsed: None,
    )

    def _fake_run_command(command, args, *, program_name):
        workflow_calls.append((command, list(args), program_name))
        return 0

    workflow_output = StringIO()
    with (
        patch("apps.breakout_quality.BREAKOUT_QUALITY_MODEL_ARCHITECTURE", "ts2vec_frozen_linear_v1"),
        patch("apps.breakout_quality._load_command_module", return_value=fake_train_module),
        patch("apps.breakout_quality._dataset_refresh_plan", return_value=("none", [])),
        patch(
            "apps.breakout_quality._pretraining_refresh_plan",
            return_value=(True, True, ["synthetic pretraining refresh"]),
        ),
        patch("apps.breakout_quality._run_command", side_effect=_fake_run_command),
        redirect_stdout(workflow_output),
    ):
        workflow_rc = app_breakout_quality._run_workflow(
            workflow_args,
            program_name="apps/breakout_quality.py",
        )
    workflow_console = workflow_output.getvalue()
    workflow_calls_by_command = {
        command: argv for command, argv, _program_name in workflow_calls
    }
    add_check(results, "cli_contract", case_id, "breakout_quality_workflow_report_rc", 0, workflow_rc)
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_workflow_uses_report_not_verbose_evaluate",
        ["build-pretrain-dataset", "pretrain", "train", "export-scores", "report"],
        [call[0] for call in workflow_calls],
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_workflow_report_includes_oos_flag",
        True,
        "--include-oos" in workflow_calls_by_command["report"],
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_workflow_propagates_experiment_to_all_model_stages",
        True,
        all(
            "--experiment-profile" in workflow_calls_by_command[command]
            and ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE
            in workflow_calls_by_command[command]
            for command in ("pretrain", "train", "export-scores", "report")
        ),
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_workflow_propagates_parallel_split_evaluation",
        True,
        "--parallel-split-evaluation" in workflow_calls_by_command["train"],
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_workflow_propagates_torch_execution_contract",
        True,
        (
            all(
                "--device" in workflow_calls_by_command[command]
                and "cpu" in workflow_calls_by_command[command]
                and "--no-mixed-precision" in workflow_calls_by_command[command]
                and "--mixed-precision-dtype" in workflow_calls_by_command[command]
                and "float16" in workflow_calls_by_command[command]
                and "--deterministic-algorithms" in workflow_calls_by_command[command]
                and "--no-allow-tf32" in workflow_calls_by_command[command]
                for command in ("pretrain", "train", "export-scores")
            )
        ),
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_workflow_propagates_experiment_and_regularization",
        True,
        (
            "--experiment-profile" in workflow_calls_by_command["train"]
            and ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE
            in workflow_calls_by_command["train"]
            and "--weight-decay" in workflow_calls_by_command["train"]
            and "0.0001" in workflow_calls_by_command["train"]
            and "--gradient-clip-norm" in workflow_calls_by_command["train"]
            and "1.0" in workflow_calls_by_command["train"]
        ),
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_workflow_propagates_refit_and_weight_modes",
        True,
        (
            "--final-refit-mode" in workflow_calls_by_command["train"]
            and "matched_optimizer_steps" in workflow_calls_by_command["train"]
            and "--class-weight-mode" in workflow_calls_by_command["train"]
            and "none" in workflow_calls_by_command["train"]
            and "--time-weight-mode" in workflow_calls_by_command["train"]
        ),
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_workflow_shows_each_stage_elapsed",
        5,
        workflow_console.count("[完成]") if "總耗時" in workflow_console else -1,
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_redirected_workflow_has_no_ansi",
        False,
        "\x1b[" in workflow_console,
    )

    mantis_workflow_calls = []

    def _fake_run_mantis_command(command, args, *, program_name):
        mantis_workflow_calls.append((command, list(args), program_name))
        return 0

    mantis_workflow_output = StringIO()
    with (
        patch(
            "apps.breakout_quality.BREAKOUT_QUALITY_MODEL_ARCHITECTURE",
            "mantis_v2_frozen_linear_v1",
        ),
        patch("apps.breakout_quality.require_mantis_v2_class", return_value=object),
        patch("apps.breakout_quality._load_command_module", return_value=fake_train_module),
        patch("apps.breakout_quality._dataset_refresh_plan", return_value=("none", [])),
        patch("apps.breakout_quality._pretraining_refresh_plan") as mocked_mantis_pretraining_plan,
        patch("apps.breakout_quality._run_command", side_effect=_fake_run_mantis_command),
        redirect_stdout(mantis_workflow_output),
    ):
        mantis_workflow_rc = app_breakout_quality._run_workflow(
            workflow_args,
            program_name="apps/breakout_quality.py",
        )
    mantis_commands = [call[0] for call in mantis_workflow_calls]
    add_check(
        results,
        "cli_contract",
        case_id,
        "mantis_workflow_skips_project_pretraining_and_runs_supervised_stages",
        (0, ["train", "export-scores", "report"], 0, 3),
        (
            mantis_workflow_rc,
            mantis_commands,
            mocked_mantis_pretraining_plan.call_count,
            mantis_workflow_output.getvalue().count("[完成]"),
        ),
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "mantis_workflow_prints_pinned_external_encoder_contract",
        True,
        (
            "external_encoder=repository=paris-noah/MantisV2"
            in mantis_workflow_output.getvalue()
            and "project_pretraining=False" in mantis_workflow_output.getvalue()
            and "encoder_fine_tuning=False" in mantis_workflow_output.getvalue()
        ),
    )

    moment_workflow_calls = []

    def _fake_run_moment_command(command, args, *, program_name):
        moment_workflow_calls.append((command, list(args), program_name))
        return 0

    moment_workflow_output = StringIO()
    with (
        patch(
            "apps.breakout_quality.BREAKOUT_QUALITY_MODEL_ARCHITECTURE",
            "moment_1_base_frozen_linear_v1",
        ),
        patch("apps.breakout_quality.require_moment_pipeline_class", return_value=object),
        patch("apps.breakout_quality._load_command_module", return_value=fake_train_module),
        patch("apps.breakout_quality._dataset_refresh_plan", return_value=("none", [])),
        patch("apps.breakout_quality._pretraining_refresh_plan") as mocked_moment_pretraining_plan,
        patch("apps.breakout_quality._run_command", side_effect=_fake_run_moment_command),
        redirect_stdout(moment_workflow_output),
    ):
        moment_workflow_rc = app_breakout_quality._run_workflow(
            workflow_args,
            program_name="apps/breakout_quality.py",
        )
    moment_commands = [call[0] for call in moment_workflow_calls]
    add_check(
        results,
        "cli_contract",
        case_id,
        "moment_workflow_skips_project_pretraining_and_runs_supervised_stages",
        (0, ["train", "export-scores", "report"], 0, 3),
        (
            moment_workflow_rc,
            moment_commands,
            mocked_moment_pretraining_plan.call_count,
            moment_workflow_output.getvalue().count("[完成]"),
        ),
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "moment_workflow_prints_pinned_external_encoder_contract",
        True,
        (
            "external_encoder=repository=AutonLab/MOMENT-1-base"
            in moment_workflow_output.getvalue()
            and "runtime=momentfm-0.1.4/transformers-5.5.0"
            in moment_workflow_output.getvalue()
            and "project_pretraining=False" in moment_workflow_output.getvalue()
            and "encoder_fine_tuning=False" in moment_workflow_output.getvalue()
        ),
    )

    def _valid_moment_runtime_version(package_name):
        return {
            "momentfm": "0.1.4",
            "transformers": "5.5.0",
        }[package_name]

    with (
        patch.object(
            moment_contract_module,
            "version",
            side_effect=_valid_moment_runtime_version,
        ),
        patch.dict(
            sys.modules,
            {"momentfm": SimpleNamespace(MOMENTPipeline=object)},
        ),
    ):
        valid_moment_pipeline = moment_contract_module.require_moment_pipeline_class()
    add_check(
        results,
        "cli_contract",
        case_id,
        "moment_runtime_accepts_exact_package_versions",
        object,
        valid_moment_pipeline,
    )

    def _stale_transformers_version(package_name):
        return {
            "momentfm": "0.1.4",
            "transformers": "5.4.0",
        }[package_name]

    try:
        with (
            patch.object(
                moment_contract_module,
                "version",
                side_effect=_stale_transformers_version,
            ),
            patch.dict(
                sys.modules,
                {"momentfm": SimpleNamespace(MOMENTPipeline=object)},
            ),
        ):
            moment_contract_module.require_moment_pipeline_class()
    except RuntimeError as exc:
        stale_transformers_rejected = (
            "package=transformers" in str(exc)
            and "expected=5.5.0" in str(exc)
            and "actual=5.4.0" in str(exc)
        )
    else:
        stale_transformers_rejected = False
    add_check(
        results,
        "cli_contract",
        case_id,
        "moment_runtime_rejects_stale_transformers",
        True,
        stale_transformers_rejected,
    )

    pretrain_module = importlib.import_module("tools.filters.breakout_quality.pretrain")
    parsed_pretrain = pretrain_module.parse_args(
        [
            "--filter-id",
            "synthetic_quality",
            "--experiment-profile",
            ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
        ]
    )
    resolved_pretrain_profile, resolved_pretrain_paths = (
        pretrain_module._resolve_pretrained_encoder_output(parsed_pretrain)
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_pretrain_uses_selected_experiment_profile",
        True,
        (
            resolved_pretrain_profile == ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE
            and resolved_pretrain_paths.output_dir.parent.name
            == ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE
        ),
    )

    train_module = importlib.import_module("tools.filters.breakout_quality.train")
    epoch_line = train_module._render_epoch_selection_progress(
        epoch=2,
        max_epochs=20,
        train_loss=0.66784,
        validation_loss=0.69397,
        elapsed_sec=65.4,
        improved=True,
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_train_epoch_output_is_concise",
        "  Epoch  2/20 | Train Loss 0.667840 | Val Loss 0.693970 | 耗時 01:05.4 | ★ 新最佳",
        epoch_line,
    )
    full_refit_line = train_module._render_full_selection_progress(
        epoch=2,
        epochs=2,
        train_loss=0.674436,
        elapsed_sec=3723.2,
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_full_refit_output_labels_train_loss",
        "  Epoch  2/2 | Train Loss 0.674436 | 耗時 01:02:03.2",
        full_refit_line,
    )
    colored_epoch_line = train_module._render_epoch_selection_progress(
        epoch=1,
        max_epochs=20,
        train_loss=0.7,
        validation_loss=0.8,
        elapsed_sec=1.2,
        improved=False,
        color=True,
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_epoch_elapsed_value_is_cyan_only",
        True,
        "耗時 \x1b[96m00:01.2\x1b[0m" in colored_epoch_line
        and "Train Loss \x1b[" not in colored_epoch_line,
    )
    build_module = importlib.import_module("tools.filters.breakout_quality.build_dataset")
    build_progress_line = build_module._render_full_build_progress(
        index=12,
        total=100,
        ticker="2330",
        event_count=12345,
        group_count=678,
        elapsed_sec=65.4,
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_dataset_progress_is_single_line_summary",
        "Dataset 重建  12/100 ( 12.0%) | 2330 | 累計 events=12,345 groups=678 | 耗時 01:05.4",
        build_progress_line,
    )
    compact_summary = train_module._render_training_summary(
        split_report={
            "selection_train_row_count": 538887,
            "selection_train_group_count": 16832,
            "selection_train_date_range": {"start": "2011-01-03", "end": "2018-11-05"},
            "inner_validation_row_count": 187316,
            "inner_validation_group_count": 6065,
            "inner_validation_date_range": {"start": "2019-01-02", "end": "2020-11-05"},
            "final_refit_row_count": 729654,
            "final_refit_group_count": 23072,
            "final_refit_date_range": {"start": "2011-01-03", "end": "2020-11-05"},
            "oos_evaluable_row_count": 591679,
            "oos_group_count": 17346,
            "oos_date_range": {"start": "2021-01-04", "end": "2025-12-22"},
        },
        use_inner_validation=True,
        final_train_loss=0.674436,
        training_sampling_mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        time_weight_mode="none",
        training_weight_reduction="batch_weight_sum",
        inner_train_sampling_summary={
            "source_row_count": 538887,
            "sampled_row_count": 16832,
        },
        final_refit_sampling_summary={
            "source_row_count": 729654,
            "sampled_row_count": 23072,
        },
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_train_summary_keeps_key_counts",
        True,
        (
            "Training Sampling：unique_ticker_date；Inner 538,887→16,832；"
            "Final 729,654→23,072" in compact_summary
            and "Final Refit：729,654 rows / 23,072 groups" in compact_summary
            and "OOS（未參與訓練）：591,679 rows / 17,346 groups" in compact_summary
            and "Final Loss：0.674436" in compact_summary
        ),
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_train_summary_omits_raw_dict",
        False,
        "split_report=" in compact_summary or "{'" in compact_summary,
    )

    policy_filter_id = app_breakout_quality._policy_filter_id()
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_interactive_filter_id_comes_from_policy",
        app_breakout_quality.normalize_filter_id(
            app_breakout_quality.BREAKOUT_QUALITY_DEFAULT_FILTER_ID
        ),
        policy_filter_id,
    )
    policy_train_settings = app_breakout_quality._policy_train_settings(policy_filter_id)
    parsed_train_defaults = app_breakout_quality._train_defaults()
    policy_fields = (
        "epochs",
        "batch_size",
        "evaluation_batch_size",
        "evaluation_workers",
        "parallel_split_evaluation",
        "train_prefetch_batches",
        "preload_feature_bank",
        "lr",
        "weight_decay",
        "gradient_clip_norm",
        "seed",
        "fixed_threshold",
        "use_inner_validation",
        "inner_validation_months",
        "early_stopping_patience",
        "early_stopping_min_delta",
    )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_interactive_training_settings_come_from_policy",
        tuple(getattr(parsed_train_defaults, name) for name in policy_fields),
        tuple(getattr(policy_train_settings, name) for name in policy_fields),
    )

    interactive_policy_settings = SimpleNamespace(
        filter_id="synthetic_quality",
        epochs=7,
        batch_size=64,
        evaluation_batch_size=256,
        evaluation_workers=3,
        parallel_split_evaluation=True,
        train_prefetch_batches=0,
        preload_feature_bank=True,
        lr=0.002,
        weight_decay=0.0002,
        gradient_clip_norm=0.8,
        seed=17,
        fixed_threshold=0.55,
        use_inner_validation=True,
        inner_validation_months=18,
        early_stopping_patience=4,
        early_stopping_min_delta=0.001,
        device="cpu",
        mixed_precision=False,
        mixed_precision_dtype="float16",
        deterministic_algorithms=True,
        allow_tf32=False,
    )
    workflow_prompt_labels = []
    workflow_bool_answers = iter((False, False))

    def _fake_workflow_bool(label, default):
        workflow_prompt_labels.append((label, default))
        return next(workflow_bool_answers)

    with (
        patch("apps.breakout_quality._policy_filter_id", return_value="synthetic_quality"),
        patch(
            "apps.breakout_quality._policy_train_settings",
            return_value=interactive_policy_settings,
        ),
        patch("apps.breakout_quality._print_policy_defaults") as mocked_policy_print,
        patch(
            "apps.breakout_quality._prompt_choice",
            side_effect=AssertionError("unexpected dataset profile prompt"),
        ) as mocked_choice,
        patch(
            "apps.breakout_quality._prompt_int",
            side_effect=AssertionError("unexpected ticker coverage prompt"),
        ) as mocked_int,
        patch("apps.breakout_quality._dataset_refresh_plan", return_value=("none", [])),
        patch("apps.breakout_quality._prompt_bool", side_effect=_fake_workflow_bool),
        patch("apps.breakout_quality._run_workflow") as mocked_interactive_workflow,
    ):
        interactive_workflow_rc = app_breakout_quality._interactive_workflow(
            "apps/breakout_quality.py"
        )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_workflow_uses_full_all_oos_without_prompts",
        (
            0,
            [(
                "是否強制完整重建 dataset（即使目前不需要）",
                False,
            ), (
                "確認開始",
                True,
            )],
            1,
            0,
            0,
            0,
        ),
        (
            interactive_workflow_rc,
            workflow_prompt_labels,
            mocked_policy_print.call_count,
            mocked_choice.call_count,
            mocked_int.call_count,
            mocked_interactive_workflow.call_count,
        ),
    )

    stale_prompt_labels = []
    stale_bool_answers = iter((True,))

    def _fake_stale_workflow_bool(label, default):
        stale_prompt_labels.append((label, default))
        return next(stale_bool_answers)

    with (
        patch("apps.breakout_quality._policy_filter_id", return_value="synthetic_quality"),
        patch(
            "apps.breakout_quality._policy_train_settings",
            return_value=interactive_policy_settings,
        ),
        patch("apps.breakout_quality._print_policy_defaults"),
        patch(
            "apps.breakout_quality._prompt_choice",
            side_effect=AssertionError("unexpected dataset profile prompt"),
        ),
        patch(
            "apps.breakout_quality._prompt_int",
            side_effect=AssertionError("unexpected ticker coverage prompt"),
        ),
        patch(
            "apps.breakout_quality._dataset_refresh_plan",
            return_value=("rebuild", ["來源 CSV inventory 已變更"]),
        ),
        patch("apps.breakout_quality._prompt_bool", side_effect=_fake_stale_workflow_bool),
        patch("apps.breakout_quality._run_workflow", return_value=0) as mocked_stale_workflow,
    ):
        stale_workflow_rc = app_breakout_quality._interactive_workflow(
            "apps/breakout_quality.py"
        )
    stale_request = mocked_stale_workflow.call_args.args[0]
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_stale_dataset_auto_rebuilds_with_fixed_full_all_oos",
        (
            0,
            [(
                "確認開始",
                True,
            )],
            False,
            "full",
            0,
            True,
            1,
        ),
        (
            stale_workflow_rc,
            stale_prompt_labels,
            bool(stale_request.rebuild_dataset),
            stale_request.dataset,
            int(stale_request.max_tickers),
            bool(stale_request.evaluate_oos),
            mocked_stale_workflow.call_count,
        ),
    )

    build_dataset_calls = []

    def _fake_build_dataset_run(command, args, *, program_name):
        build_dataset_calls.append((command, list(args), program_name))
        return 0

    with (
        patch("apps.breakout_quality._policy_filter_id", return_value="synthetic_quality"),
        patch("apps.breakout_quality._print_policy_defaults"),
        patch(
            "apps.breakout_quality._prompt_choice",
            side_effect=AssertionError("unexpected dataset profile prompt"),
        ) as mocked_build_choice,
        patch(
            "apps.breakout_quality._prompt_int",
            side_effect=AssertionError("unexpected ticker coverage prompt"),
        ) as mocked_build_int,
        patch("apps.breakout_quality._prompt_bool", return_value=True),
        patch("apps.breakout_quality._run_command", side_effect=_fake_build_dataset_run),
    ):
        build_dataset_rc = app_breakout_quality._interactive_build_dataset(
            "apps/breakout_quality.py"
        )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_build_dataset_menu_uses_full_all_without_prompts",
        (
            0,
            0,
            0,
            "build-dataset",
            ["--dataset", "full", "--filter-id", "synthetic_quality"],
        ),
        (
            build_dataset_rc,
            mocked_build_choice.call_count,
            mocked_build_int.call_count,
            build_dataset_calls[-1][0],
            build_dataset_calls[-1][1],
        ),
    )


    with (
        patch("apps.breakout_quality._policy_filter_id", return_value="synthetic_quality"),
        patch(
            "apps.breakout_quality._policy_train_settings",
            return_value=interactive_policy_settings,
        ),
        patch("apps.breakout_quality._print_policy_defaults") as mocked_policy_print,
        patch("apps.breakout_quality._prompt_bool", return_value=False) as mocked_confirm,
        patch("builtins.input", side_effect=AssertionError("unexpected policy prompt")),
        patch("apps.breakout_quality._run_command") as mocked_train_command,
    ):
        interactive_train_rc = app_breakout_quality._interactive_train(
            "apps/breakout_quality.py"
        )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_train_skips_policy_backed_questions",
        (0, 1, 1, 0),
        (
            interactive_train_rc,
            mocked_policy_print.call_count,
            mocked_confirm.call_count,
            mocked_train_command.call_count,
        ),
    )

    report_calls = []

    def _fake_report_run(command, args, *, program_name):
        report_calls.append((command, list(args), program_name))
        return 0

    with (
        patch("apps.breakout_quality._policy_filter_id", return_value="synthetic_quality"),
        patch(
            "apps.breakout_quality.BREAKOUT_QUALITY_EXPERIMENT_PROFILE",
            "baseline",
        ),
        patch(
            "apps.breakout_quality._prompt_bool",
            side_effect=AssertionError("interactive report must not prompt for OOS"),
        ) as mocked_prompt,
        patch("apps.breakout_quality._run_command", side_effect=_fake_report_run),
    ):
        report_with_oos_rc = app_breakout_quality._interactive_report(
            "apps/breakout_quality.py"
        )
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_report_fixed_oos_without_prompt",
        (
            0,
            0,
            "report",
            [
                "--filter-id",
                "synthetic_quality",
                "--experiment-profile",
                "baseline",
                "--include-oos",
            ],
        ),
        (
            report_with_oos_rc,
            mocked_prompt.call_count,
            report_calls[-1][0],
            report_calls[-1][1],
        ),
    )
    report_module = importlib.import_module("tools.filters.breakout_quality.report")
    add_check(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_report_cli_defaults_to_oos",
        True,
        report_module.parse_args([]).include_oos,
    )

    with TemporaryDirectory(prefix="breakout_quality_rebuild_detection_") as temp_dir:
        temp_root = Path(temp_dir)
        source_dir = temp_root / "data" / "tw_stock_data_vip_reduced"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_csv = source_dir / "2330.csv"
        benchmark_csv = source_dir / "0050.csv"
        source_csv.write_text("Date,Open\n2026-01-01,100\n", encoding="utf-8")
        benchmark_csv.write_text("Date,Open\n2026-01-01,50\n", encoding="utf-8")

        with patch.object(app_breakout_quality, "PROJECT_ROOT", temp_root):
            dataset_paths = app_breakout_quality._dataset_paths("synthetic_quality")
            for name, path in dataset_paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == "summary":
                    continue
                if name == "events_csv":
                    path.write_text("ticker,date\n", encoding="utf-8")
                else:
                    path.write_bytes(b"npy")
            inventory = app_breakout_quality.build_source_data_inventory(temp_root, "reduced")
            summary_payload = {
                "filter_id": "synthetic_quality",
                "dataset": "reduced",
                "dataset_storage_schema_version": app_breakout_quality.DATASET_STORAGE_SCHEMA_VERSION,
                "dataset_storage_format": app_breakout_quality.DATASET_STORAGE_FORMAT,
                "source_data_inventory": inventory,
                "source_selection": {"requested_max_tickers": 0},
                "policy": app_breakout_quality.DEFAULT_LABEL_POLICY.as_manifest_payload(),
                "feature_cache_policy": app_breakout_quality.DEFAULT_LABEL_POLICY.feature_cache_manifest_payload(),
                "label_policy": app_breakout_quality.DEFAULT_LABEL_POLICY.label_manifest_payload(),
                "feature_columns": list(app_breakout_quality.FEATURE_COLUMNS),
                "context_columns": list(app_breakout_quality.CONTEXT_COLUMNS),
                "dataset_artifacts": {
                    name: {
                        "filename": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": "synthetic-not-used-by-refresh-plan",
                    }
                    for name, path in dataset_paths.items()
                    if name != "summary"
                },
            }
            dataset_paths["summary"].write_text(
                json.dumps(summary_payload),
                encoding="utf-8",
            )
            unchanged_reasons = app_breakout_quality._dataset_rebuild_reasons(
                "synthetic_quality",
                "reduced",
                max_tickers=0,
            )
            relabel_summary = dict(summary_payload)
            relabel_summary["label_policy"] = {
                **summary_payload["label_policy"],
                "min_mfe_return": 0.99,
            }
            relabel_summary["policy"] = {
                **summary_payload["policy"],
                "min_mfe_return": 0.99,
            }
            dataset_paths["summary"].write_text(
                json.dumps(relabel_summary),
                encoding="utf-8",
            )
            relabel_plan = app_breakout_quality._dataset_refresh_plan(
                "synthetic_quality",
                "reduced",
                max_tickers=0,
            )
            dataset_paths["summary"].write_text(
                json.dumps(summary_payload),
                encoding="utf-8",
            )
            source_csv.write_text(
                "Date,Open\n2026-01-01,100\n2026-01-02,101\n",
                encoding="utf-8",
            )
            changed_reasons = app_breakout_quality._dataset_rebuild_reasons(
                "synthetic_quality",
                "reduced",
                max_tickers=0,
            )
            coverage_reasons = app_breakout_quality._dataset_rebuild_reasons(
                "synthetic_quality",
                "reduced",
                max_tickers=30,
            )

        add_check(
            results,
            "cli_contract",
            case_id,
            "breakout_quality_unchanged_source_skips_rebuild",
            [],
            unchanged_reasons,
        )
        add_check(
            results,
            "cli_contract",
            case_id,
            "breakout_quality_label_policy_change_uses_fast_relabel",
            "relabel",
            relabel_plan[0],
        )
        add_check(
            results,
            "cli_contract",
            case_id,
            "breakout_quality_updated_source_requires_rebuild",
            True,
            any("來源 CSV inventory 已更新" in reason for reason in changed_reasons),
        )
        add_check(
            results,
            "cli_contract",
            case_id,
            "breakout_quality_ticker_coverage_change_requires_rebuild",
            True,
            any("ticker coverage 不符" in reason for reason in coverage_reasons),
        )

    command_modules = {
        "build-dataset": "tools.filters.breakout_quality.build_dataset",
        "train": "tools.filters.breakout_quality.train",
        "export-scores": "tools.filters.breakout_quality.export_scores",
        "report": "tools.filters.breakout_quality.report",
        "evaluate": "tools.filters.breakout_quality.evaluate",
    }
    for command, expected_module in command_modules.items():
        received_argv = []

        def _fake_command_main(argv, received_argv=received_argv):
            received_argv.append(list(argv))
            return 23

        fake_module = SimpleNamespace(main=_fake_command_main)
        with patch("apps.breakout_quality.importlib.import_module", return_value=fake_module) as mocked_import:
            rc = app_breakout_quality.main(
                ["apps/breakout_quality.py", command, "--filter-id", "synthetic_quality"]
            )
        metric_command = command.replace("-", "_")
        add_check(results, "cli_contract", case_id, f"breakout_quality_app_{metric_command}_rc", 23, rc)
        add_check(
            results,
            "cli_contract",
            case_id,
            f"breakout_quality_app_{metric_command}_module",
            expected_module,
            mocked_import.call_args.args[0],
        )
        add_check(
            results,
            "cli_contract",
            case_id,
            f"breakout_quality_app_{metric_command}_argv",
            [["--filter-id", "synthetic_quality"]],
            received_argv,
        )

    _assert_value_error(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_app_unknown_command_rejected",
        lambda: app_breakout_quality.main(["apps/breakout_quality.py", "unknown"]),
        "不支援的 breakout quality command",
    )
    _assert_value_error(
        results,
        "cli_contract",
        case_id,
        "breakout_quality_app_unknown_top_level_flag_rejected",
        lambda: app_breakout_quality.main(["apps/breakout_quality.py", "--bad"]),
        "不支援的參數",
    )

    wrapper_cases = [
        {
            "program": "apps/ml_optimizer.py",
            "module": app_ml_optimizer,
            "patch_target": "tools.optimizer.main",
            "env_kw": "environ",
        },
        {
            "program": "apps/vip_scanner.py",
            "module": app_vip_scanner,
            "patch_target": "tools.scanner.main",
            "env_kw": "env",
        },
        {
            "program": "apps/portfolio_sim.py",
            "module": app_portfolio_sim,
            "patch_target": "tools.portfolio_sim.main",
            "env_kw": "env",
            "extra_kw": {"env": {"V16_AUTO_OPEN_BROWSER": "0"}},
        },
        {
            "program": "tools/scanner/main.py",
            "module": scanner_main_module,
            "patch_target": "tools.scanner.scan_runner.main",
            "env_kw": "env",
        },
        {
            "program": "tools/validate/cli.py",
            "module": validate_cli_module,
            "patch_target": "tools.validate.main",
            "env_kw": "environ",
        },
    ]

    for case in wrapper_cases:
        program = case["program"]
        module = case["module"]
        kwargs = dict(case.get("extra_kw", {}))
        if case.get("env_kw") and case["env_kw"] not in kwargs:
            kwargs[case["env_kw"]] = {"TEST": "1"}

        rc, help_text = _capture_stdout(module.main, [program, "--help"], **kwargs)
        metric_prefix = program.replace("/", "_").replace(".", "_")
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_rc", 0, rc)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_usage", True, f"用法: python {program}" in help_text)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_dataset_choices", True, "reduced|full" in help_text)

        sentinel = 17
        with patch(case["patch_target"], return_value=sentinel) as mocked:
            rc = module.main([program], **kwargs)
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_default_passthrough_rc", sentinel, rc)
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_default_passthrough_called", 1, mocked.call_count)
            called_argv = mocked.call_args.kwargs.get("argv")
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_default_passthrough_argv", [program], called_argv)

        with patch(case["patch_target"], return_value=sentinel) as mocked:
            rc = module.main([program, "--dataset", "reduced"], **kwargs)
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_reduced_passthrough_rc", sentinel, rc)
            called_argv = mocked.call_args.kwargs.get("argv")
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_reduced_passthrough_argv", [program, "--dataset", "reduced"], called_argv)

        with patch(case["patch_target"], return_value=sentinel) as mocked:
            rc = module.main([program, "--dataset=full"], **kwargs)
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_inline_full_passthrough_rc", sentinel, rc)
            called_argv = mocked.call_args.kwargs.get("argv")
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_inline_full_passthrough_argv", [program, "--dataset=full"], called_argv)

        _assert_value_error(
            results,
            "cli_contract",
            case_id,
            f"{metric_prefix}_invalid_flag_rejected",
            lambda module=module, kwargs=kwargs: module.main([program, "--bad"], **kwargs),
            "不支援的參數",
        )
        _assert_value_error(
            results,
            "cli_contract",
            case_id,
            f"{metric_prefix}_missing_dataset_value_rejected",
            lambda module=module, kwargs=kwargs: module.main([program, "--dataset"], **kwargs),
            "缺少值",
        )
        _assert_value_error(
            results,
            "cli_contract",
            case_id,
            f"{metric_prefix}_empty_dataset_value_rejected",
            lambda module=module, kwargs=kwargs: module.main([program, "--dataset="], **kwargs),
            "不能為空",
        )
        _assert_value_error(
            results,
            "cli_contract",
            case_id,
            f"{metric_prefix}_positional_arg_rejected",
            lambda module=module, kwargs=kwargs: module.main([program, "extra"], **kwargs),
            "不支援的位置參數",
        )

    summary["wrapper_count"] = len(wrapper_cases)
    return results, summary


def validate_local_regression_cli_contract_case(_base_params):
    case_id = "CLI_LOCAL_REGRESSION_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    app_package_zip = importlib.import_module("apps.package_zip")
    app_smart_downloader = importlib.import_module("apps.smart_downloader")
    export_requirements_lock = importlib.import_module("requirements.export_requirements_lock")
    downloader_main_module = importlib.import_module("tools.downloader.main")
    run_all = importlib.import_module("tools.local_regression.run_all")
    run_chain_checks = importlib.import_module("tools.local_regression.run_chain_checks")
    run_meta_quality = importlib.import_module("tools.local_regression.run_meta_quality")
    run_ml_smoke = importlib.import_module("tools.local_regression.run_ml_smoke")
    run_quick_gate = importlib.import_module("tools.local_regression.run_quick_gate")
    preflight_env = importlib.import_module("tools.validate.preflight_env")

    rc, help_text = _capture_stdout(run_all.main, ["tools/local_regression/run_all.py", "--help"])
    add_check(results, "cli_contract", case_id, "run_all_help_rc", 0, rc)
    add_check(results, "cli_contract", case_id, "run_all_help_mentions_meta_quality", True, "meta_quality" in help_text)
    add_check(results, "cli_contract", case_id, "run_all_help_mentions_only", True, "--only" in help_text)

    parsed_default = run_all._parse_cli_args(["tools/local_regression/run_all.py"])
    add_check(results, "cli_contract", case_id, "run_all_default_only_steps_none", None, parsed_default.get("only_steps"))
    parsed_dedup = run_all._parse_cli_args(["tools/local_regression/run_all.py", "--only", "quick_gate,quick_gate,meta_quality"])
    add_check(results, "cli_contract", case_id, "run_all_only_dedup_normalized", ["quick_gate", "meta_quality"], parsed_dedup.get("only_steps"))
    parsed_inline = run_all._parse_cli_args(["tools/local_regression/run_all.py", "--only=ml_smoke,meta_quality"])
    add_check(results, "cli_contract", case_id, "run_all_only_inline_normalized", ["ml_smoke", "meta_quality"], parsed_inline.get("only_steps"))

    _assert_value_error(results, "cli_contract", case_id, "run_all_only_missing_value_rejected", lambda: run_all._parse_cli_args(["tools/local_regression/run_all.py", "--only"]), "缺少值")
    _assert_value_error(results, "cli_contract", case_id, "run_all_only_empty_value_rejected", lambda: run_all._parse_cli_args(["tools/local_regression/run_all.py", "--only="]), "不可為空")
    _assert_value_error(results, "cli_contract", case_id, "run_all_unknown_flag_rejected", lambda: run_all._parse_cli_args(["tools/local_regression/run_all.py", "--bad"]), "不支援的參數")

    rc, help_text = _capture_stdout(preflight_env.main, ["tools/validate/preflight_env.py", "--help"])
    add_check(results, "cli_contract", case_id, "preflight_help_rc", 0, rc)
    add_check(results, "cli_contract", case_id, "preflight_help_mentions_meta_quality", True, "meta_quality" in help_text)
    parsed_steps_none = preflight_env._parse_cli_steps(["tools/validate/preflight_env.py"])
    add_check(results, "cli_contract", case_id, "preflight_default_steps_none", None, parsed_steps_none)
    parsed_steps = preflight_env._parse_cli_steps(["tools/validate/preflight_env.py", "--steps", "quick_gate,meta_quality"])
    add_check(results, "cli_contract", case_id, "preflight_steps_parse", ["quick_gate", "meta_quality"], parsed_steps)
    normalized_steps = preflight_env._normalize_local_regression_steps(["quick_gate", "quick_gate", "meta_quality"])
    add_check(results, "cli_contract", case_id, "preflight_steps_dedup_normalized", ["quick_gate", "meta_quality"], normalized_steps)
    _assert_value_error(results, "cli_contract", case_id, "preflight_invalid_step_rejected", lambda: preflight_env._normalize_local_regression_steps(["bad"]), "只接受")

    no_arg_cases = [
        ("apps/package_zip.py", app_package_zip.main),
        ("requirements/export_requirements_lock.py", export_requirements_lock.main),
        ("tools/local_regression/run_chain_checks.py", run_chain_checks.main),
        ("tools/local_regression/run_ml_smoke.py", run_ml_smoke.main),
        ("tools/local_regression/run_meta_quality.py", run_meta_quality.main),
        ("tools/local_regression/run_quick_gate.py", run_quick_gate.main),
        ("apps/smart_downloader.py", app_smart_downloader.main),
        ("tools/downloader/main.py", downloader_main_module.main),
    ]
    for program, main_func in no_arg_cases:
        metric_prefix = program.replace("/", "_").replace(".", "_")
        rc, help_text = _capture_stdout(main_func, [program, "--help"])
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_rc", 0, rc)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_usage", True, f"用法: python {program}" in help_text)
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_unknown_flag_rejected", lambda main_func=main_func, program=program: main_func([program, "--bad"]), "不支援的參數")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_positional_arg_rejected", lambda main_func=main_func, program=program: main_func([program, "extra"]), "不支援的位置參數")

    summary["no_arg_case_count"] = len(no_arg_cases)
    return results, summary


def validate_run_all_cli_error_usage_contract_case(_base_params):
    case_id = "RUN_ALL_CLI_ERROR_USAGE_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    run_all = importlib.import_module("tools.local_regression.run_all")

    rc, stderr_text = _capture_stderr(run_all.main, ["tools/local_regression/run_all.py", "--bad"])
    add_check(results, "cli_contract", case_id, "run_all_invalid_flag_main_rc", 2, rc)
    add_check(results, "cli_contract", case_id, "run_all_invalid_flag_error_usage_mentions_meta_quality", True, "meta_quality" in stderr_text)
    add_check(results, "cli_contract", case_id, "run_all_invalid_flag_error_usage_mentions_only", True, "--only" in stderr_text)

    rc, stderr_text = _capture_stderr(run_all.main, ["tools/local_regression/run_all.py", "--only"])
    add_check(results, "cli_contract", case_id, "run_all_missing_only_value_rc", 2, rc)
    add_check(results, "cli_contract", case_id, "run_all_missing_only_value_usage_mentions_meta_quality", True, "meta_quality" in stderr_text)

    return results, summary


def validate_package_zip_runtime_contract_case(_base_params):
    case_id = "PACKAGE_ZIP_RUNTIME_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    app_package_zip = importlib.import_module("apps.package_zip")

    with TemporaryDirectory(prefix="package_zip_contract_") as tmp_dir:
        project_root = Path(tmp_dir)
        (project_root / "apps").mkdir(parents=True)
        (project_root / "arch").mkdir()
        (project_root / "pkg").mkdir()
        (project_root / "pkg" / "module.py").write_text("print('ok')\n", encoding="utf-8")
        (project_root / "README.md").write_text("demo\n", encoding="utf-8")
        (project_root / "pkg" / "__pycache__").mkdir()
        (project_root / "pkg" / "__pycache__" / "module.cpython-312.pyc").write_bytes(b"cache")
        (project_root / "orphan.pyc").write_bytes(b"orphan-cache")
        (project_root / "main_20250101_deadbeef.zip").write_bytes(b"main-old")
        (project_root / "other_branch_20250102_cafebabe.zip").write_bytes(b"other-old")
        (project_root / "to_chatgpt_bundle_20250103_deadbeef.zip").write_bytes(b"bundle-old")

        fake_now = SimpleNamespace(strftime=lambda fmt: "20260404_123456")

        def _fake_run_git(*args):
            if args == ("rev-parse", "--abbrev-ref", "HEAD"):
                return SimpleNamespace(stdout="feature/runtime-contract\n")
            if args == ("rev-parse", "--short", "HEAD"):
                return SimpleNamespace(stdout="abc1234\n")
            if args == ("ls-files", "--cached", "--others", "--exclude-standard", "-z"):
                return SimpleNamespace(stdout="pkg/module.py\0README.md\0pkg/__pycache__/module.cpython-312.pyc\0orphan.pyc\0")
            raise AssertionError(f"unexpected git args: {args}")

        with patch.object(app_package_zip, "PROJECT_ROOT", project_root),              patch.object(app_package_zip, "get_taipei_now", return_value=fake_now),              patch.object(app_package_zip, "_run_git", side_effect=_fake_run_git):
            rc, stdout_text = _capture_stdout(app_package_zip.main, ["apps/package_zip.py"])

        new_zip_path = project_root / "feature-runtime-contract_20260404_123456_abc1234.zip"
        archived_root_zips = sorted(path.name for path in (project_root / "arch").glob("*.zip"))
        add_check(results, "cli_contract", case_id, "package_zip_main_rc", 0, rc)
        add_check(results, "cli_contract", case_id, "package_zip_output_exists", True, new_zip_path.exists())
        add_check(results, "cli_contract", case_id, "package_zip_archives_non_bundle_root_zips_only", ["main_20250101_deadbeef.zip", "other_branch_20250102_cafebabe.zip"], archived_root_zips)
        add_check(results, "cli_contract", case_id, "package_zip_root_old_zip_removed", False, (project_root / "other_branch_20250102_cafebabe.zip").exists())
        add_check(results, "cli_contract", case_id, "package_zip_root_bundle_preserved", True, (project_root / "to_chatgpt_bundle_20250103_deadbeef.zip").exists())
        add_check(results, "cli_contract", case_id, "package_zip_root_bundle_not_archived", False, (project_root / "arch" / "to_chatgpt_bundle_20250103_deadbeef.zip").exists())
        add_check(results, "cli_contract", case_id, "package_zip_cache_dir_removed", False, (project_root / "pkg" / "__pycache__").exists())
        add_check(results, "cli_contract", case_id, "package_zip_orphan_pyc_removed", False, (project_root / "orphan.pyc").exists())
        add_check(results, "cli_contract", case_id, "package_zip_stdout_reports_archived_count", True, "[package_zip] archived old root zips=2" in stdout_text)

        with zipfile.ZipFile(new_zip_path) as zf:
            member_names = sorted(zf.namelist())
        add_check(results, "cli_contract", case_id, "package_zip_zip_members", ["README.md", "pkg/module.py"], member_names)

    summary["checks"] = len(results)
    return results, summary



def validate_package_zip_commit_test_suite_orchestration_case(_base_params):
    case_id = "PACKAGE_ZIP_COMMIT_TEST_SUITE_ORCHESTRATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    app_package_zip = importlib.import_module("apps.package_zip")

    with TemporaryDirectory(prefix="package_zip_orchestration_") as tmp_dir:
        project_root = Path(tmp_dir)
        (project_root / "apps").mkdir(parents=True)
        (project_root / "pkg").mkdir()
        (project_root / "pkg" / "module.py").write_text("print('ok')\n", encoding="utf-8")
        (project_root / "README.md").write_text("demo\n", encoding="utf-8")
        (project_root / "legacy_20250101_deadbeef.zip").write_bytes(b"legacy")
        (project_root / "to_chatgpt_bundle_20250103_deadbeef.zip").write_bytes(b"bundle-old")

        fake_now = SimpleNamespace(strftime=lambda fmt: "20260405_120000")
        git_commands = []
        python_commands = []

        def _fake_run_git(*args):
            git_commands.append(args)
            if args == ("status", "--porcelain", "--untracked-files=all"):
                return SimpleNamespace(stdout=" M apps/package_zip.py\n?? README.md\n")
            if args == ("add", "-A"):
                return SimpleNamespace(stdout="")
            if args == ("commit", "-m", "feat: package workflow"):
                return SimpleNamespace(stdout="[feature/workflow fedcba9] feat: package workflow\n")
            if args == ("rev-parse", "--abbrev-ref", "HEAD"):
                return SimpleNamespace(stdout="feature/workflow\n")
            if args == ("rev-parse", "--short", "HEAD"):
                return SimpleNamespace(stdout="fedcba9\n")
            if args == ("ls-files", "--cached", "--others", "--exclude-standard", "-z"):
                return SimpleNamespace(stdout="README.md\0pkg/module.py\0")
            raise AssertionError(f"unexpected git args: {args}")

        def _fake_run_python_command(command):
            python_commands.append(command)
            return SimpleNamespace(returncode=0)

        with patch.object(app_package_zip, "PROJECT_ROOT", project_root),              patch.object(app_package_zip, "get_taipei_now", return_value=fake_now),              patch.object(app_package_zip, "_run_git", side_effect=_fake_run_git),              patch.object(app_package_zip, "_run_python_command", side_effect=_fake_run_python_command):
            rc, stdout_text = _capture_stdout(
                app_package_zip.main,
                ["apps/package_zip.py", "--commit-message", "feat: package workflow", "--run-test-suite"],
            )

        new_zip_path = project_root / "feature-workflow_20260405_120000_fedcba9.zip"
        commit_idx = git_commands.index(("commit", "-m", "feat: package workflow"))
        ls_files_idx = git_commands.index(("ls-files", "--cached", "--others", "--exclude-standard", "-z"))
        add_check(results, "cli_contract", case_id, "package_zip_orchestration_main_rc", 0, rc)
        add_check(results, "cli_contract", case_id, "package_zip_orchestration_add_called", True, ("add", "-A") in git_commands)
        add_check(results, "cli_contract", case_id, "package_zip_orchestration_commit_called", True, ("commit", "-m", "feat: package workflow") in git_commands)
        add_check(results, "cli_contract", case_id, "package_zip_orchestration_commit_precedes_zip_snapshot", True, commit_idx < ls_files_idx)
        add_check(results, "cli_contract", case_id, "package_zip_orchestration_output_uses_post_commit_sha", True, new_zip_path.exists())
        add_check(results, "cli_contract", case_id, "package_zip_orchestration_test_suite_called", [[sys.executable, "apps/test_suite.py"]], python_commands)
        add_check(results, "cli_contract", case_id, "package_zip_orchestration_test_suite_after_zip", True, stdout_text.index(f"[package_zip] output={new_zip_path}") < stdout_text.index("[package_zip] test_suite=pass"))
        add_check(results, "cli_contract", case_id, "package_zip_orchestration_commit_headline_reported", True, "[package_zip] commit=[feature/workflow fedcba9] feat: package workflow" in stdout_text)
        add_check(results, "cli_contract", case_id, "package_zip_orchestration_bundle_preserved", True, (project_root / "to_chatgpt_bundle_20250103_deadbeef.zip").exists())

    summary["checks"] = len(results)
    return results, summary


def validate_extended_tool_cli_contract_case(_base_params):
    case_id = "CLI_EXTENDED_TOOL_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    app_test_suite = importlib.import_module("apps.test_suite")
    app_workbench = importlib.import_module("apps.workbench")
    optimizer_main = importlib.import_module("tools.optimizer.main")
    portfolio_sim_main = importlib.import_module("tools.portfolio_sim.main")
    scan_runner = importlib.import_module("tools.scanner.scan_runner")
    validate_main = importlib.import_module("tools.validate.main")

    dataset_cases = [
        ("tools/optimizer/main.py", optimizer_main.main, "environ", {}),
        ("tools/portfolio_sim/main.py", portfolio_sim_main.main, "env", {"V16_AUTO_OPEN_BROWSER": "0"}),
        ("tools/scanner/scan_runner.py", scan_runner.main, "env", {"TEST": "1"}),
        ("tools/validate/main.py", validate_main.main, "environ", {}),
    ]

    for program, main_func, env_kw, env_value in dataset_cases:
        metric_prefix = program.replace("/", "_").replace(".", "_")
        kwargs = {env_kw: env_value}
        rc, help_text = _capture_stdout(main_func, [program, "--help"], **kwargs)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_rc", 0, rc)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_usage", True, f"用法: python {program}" in help_text)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_dataset_choices", True, "reduced|full" in help_text)
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_invalid_flag_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "--bad"], **kwargs), "不支援的參數")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_missing_dataset_value_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "--dataset"], **kwargs), "缺少值")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_empty_dataset_value_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "--dataset="], **kwargs), "不能為空")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_positional_arg_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "extra"], **kwargs), "不支援的位置參數")

    no_dataset_cases = [
        ("apps/test_suite.py", app_test_suite.main, {}),
        ("apps/workbench.py", app_workbench.main, {}),
    ]

    for program, main_func, kwargs in no_dataset_cases:
        metric_prefix = program.replace("/", "_").replace(".", "_")
        rc, help_text = _capture_stdout(main_func, [program, "--help"], **kwargs)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_rc", 0, rc)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_usage", True, f"用法: python {program}" in help_text)
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_invalid_flag_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "--bad"], **kwargs), "不支援的參數")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_positional_arg_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "extra"], **kwargs), "不支援的位置參數")

    summary["extended_case_count"] = len(dataset_cases) + len(no_dataset_cases)
    return results, summary
