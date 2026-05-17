from __future__ import annotations

from typing import Dict, List, Tuple

from core.model_paths import discover_model_param_sources, resolve_default_primary_param_source_record, resolve_run_best_params_path

DEFAULT_PARAM_SOURCE_LABEL = "run_best_params.json"
LEGACY_PARAM_SOURCE_LABEL = "run_best_params.json"


def build_workbench_param_source_options(project_root: str, *, include_rolling_oos: bool = False, include_active_param_ensemble: bool = False) -> Tuple[List[str], Dict[str, str], Dict[str, str], str]:
    records = discover_model_param_sources(project_root, include_rolling_oos=include_rolling_oos, include_active_param_ensemble=include_active_param_ensemble)
    labels = [str(record["label"]) for record in records]
    path_by_label = {str(record["label"]): str(record["path"]) for record in records}
    key_by_label = {str(record["label"]): str(record["key"]) for record in records}

    if not labels:
        # Keep the Workbench usable enough to show the existing strict-load error
        # when models/ has not been populated yet.
        fallback_path = resolve_run_best_params_path(project_root)
        return [LEGACY_PARAM_SOURCE_LABEL], {LEGACY_PARAM_SOURCE_LABEL: fallback_path}, {LEGACY_PARAM_SOURCE_LABEL: "run_best"}, LEGACY_PARAM_SOURCE_LABEL

    default_record = resolve_default_primary_param_source_record(project_root)
    default_label = str(default_record.get("label") or DEFAULT_PARAM_SOURCE_LABEL)
    if default_label not in path_by_label:
        default_label = LEGACY_PARAM_SOURCE_LABEL if LEGACY_PARAM_SOURCE_LABEL in path_by_label else labels[0]
    return labels, path_by_label, key_by_label, default_label
