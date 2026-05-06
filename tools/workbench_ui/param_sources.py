from __future__ import annotations

from typing import Dict, List, Tuple

from core.model_paths import discover_model_param_sources, resolve_run_best_params_path

DEFAULT_PARAM_SOURCE_LABEL = "run_best | 目前參數"


def build_workbench_param_source_options(project_root: str) -> Tuple[List[str], Dict[str, str], Dict[str, str], str]:
    records = discover_model_param_sources(project_root)
    labels = [str(record["label"]) for record in records]
    path_by_label = {str(record["label"]): str(record["path"]) for record in records}
    key_by_label = {str(record["label"]): str(record["key"]) for record in records}

    if not labels:
        # Keep the Workbench usable enough to show the existing strict-load error
        # when models/ has not been populated yet.
        fallback_path = resolve_run_best_params_path(project_root)
        return [DEFAULT_PARAM_SOURCE_LABEL], {DEFAULT_PARAM_SOURCE_LABEL: fallback_path}, {DEFAULT_PARAM_SOURCE_LABEL: "run_best"}, DEFAULT_PARAM_SOURCE_LABEL

    default_label = DEFAULT_PARAM_SOURCE_LABEL if DEFAULT_PARAM_SOURCE_LABEL in path_by_label else labels[0]
    return labels, path_by_label, key_by_label, default_label
