"""Registry-driven ranking wiring checks, independent of current experiments."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import patch


def append_ranking_policy_descriptor_checks(check):
    from core import buy_sort, portfolio_entry_selection as selection, ranking_policy
    from filters.breakout_quality.strategy_compare_dl_artifacts import (
        resolve_arm_artifact_dl_source_ids,
    )

    specs = ranking_policy.RANKING_POLICY_SPECS
    check("ranking_policy_unique_descriptor_ids", len(specs), len({s.policy_id for s in specs}))
    missing = [s.policy_id for s in specs if s.resource_aware and (
        s.selector_key not in selection._SELECTOR_HANDLERS
        or (s.score_transform_key is not None and s.score_transform_key not in selection._SCORE_TRANSFORMS)
    )]
    check("ranking_policy_all_primitives_are_bound", [], missing)
    sources = {name: SimpleNamespace(score_source="selection_point_in_time")
               for name in ("primary", "safety", "fit_r", "fit_excess")}
    role_options = {
        "safety": ("safety_dl_id", "safety"),
        "expected_r_calibration": ("expected_r_fit_dl_id", "fit_r"),
        "expected_excess_r_calibration": ("expected_excess_r_fit_dl_id", "fit_excess"),
    }
    failures = []
    for spec in specs:
        options = {role_options[role][0]: role_options[role][1] for role in spec.dependency_roles}
        arm = SimpleNamespace(arm_id="fixture", dl_enabled=True, dl_id="primary",
                              dl_runtime_mode=spec.policy_id, dl_runtime_options=options)
        actual = resolve_arm_artifact_dl_source_ids(SimpleNamespace(dl_sources=sources), arm)
        expected = ("primary", *(role_options[role][1] for role in spec.dependency_roles))
        if actual != expected:
            failures.append((spec.policy_id, expected, actual))
        if buy_sort.build_breakout_quality_score_ranking_order(spec.policy_id) != list(spec.score_order):
            failures.append((spec.policy_id, "report score order"))
        contract = buy_sort.build_breakout_quality_capital_aware_ranking_contract(
            spec.policy_id, ranking_options=options,
        )
        if bool(contract is not None) != spec.capital_contract_enabled:
            failures.append((spec.policy_id, "capital report contract availability"))
        for option_name in options:
            missing_arm = SimpleNamespace(**{**vars(arm), "dl_runtime_options": {
                key: value for key, value in options.items() if key != option_name
            }})
            try:
                resolve_arm_artifact_dl_source_ids(SimpleNamespace(dl_sources=sources), missing_arm)
            except ValueError:
                continue
            failures.append((spec.policy_id, "missing dependency accepted", option_name))
    check("ranking_policy_report_and_dependency_capabilities_are_complete", [], failures)

    # An alias combining existing capabilities must work without adding its ID
    # to the dependency resolver or the Portfolio selector's dispatch tables.
    source = next(spec for spec in specs if spec.dynamic_options_key == "safety_percentile")
    alias = replace(source, policy_id="synthetic-composed-ranking-policy")
    try:
        alias.selector_key = "different"
        immutable = False
    except FrozenInstanceError:
        immutable = True
    check("ranking_policy_descriptor_is_immutable", True, immutable)
    registry = {spec.policy_id: spec for spec in specs}
    registry[alias.policy_id] = alias
    rows = [{"ticker": "sample", "breakout_quality_ranking_options": {"safety_percentile_cutoff": 0.75}}]
    routed = []

    def handler(candidates, **kwargs):
        routed.append(kwargs)
        return candidates, kwargs["default_diag"]

    with patch.object(ranking_policy, "_POLICY_BY_ID", registry), \
         patch.object(selection, "_resource_aware_quality_policy", return_value=alias.policy_id), \
         patch.object(selection, "_resource_aware_default_diag", return_value={"selector": alias.selector_label}), \
         patch.object(selection, "_simulate_reserved_candidate_order", return_value={"selected_count": 1}), \
         patch.dict(selection._SELECTOR_HANDLERS, {alias.selector_key: handler}), \
         patch.dict(selection._SCORE_TRANSFORMS, {alias.score_transform_key: lambda values: (values, {})}):
        order, diagnostic = selection.reorder_candidates_for_resource_aware_quality(
            rows, available_cash=1000, sizing_equity=1000, pre_market_occupied=0,
            max_positions=1, params=SimpleNamespace(),
        )
        dependencies = resolve_arm_artifact_dl_source_ids(
            SimpleNamespace(dl_sources=sources),
            SimpleNamespace(arm_id="fixture", dl_enabled=True, dl_id="primary",
                            dl_runtime_mode=alias.policy_id, dl_runtime_options={"safety_dl_id": "safety"}),
        )
    check("ranking_policy_new_composition_routes_by_capability", (rows, ("primary", "safety"), 0.75),
          (order, dependencies, routed[0]["safety_percentile_cutoff"]))
    check("ranking_policy_new_composition_keeps_declared_selector_label",
          alias.selector_label, diagnostic["selector"])
