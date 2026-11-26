"""Lightweight declarative contract, independent of portfolio/config imports."""
POLICY = "same_day_raw_score_permutation_v1"
REPEATS = 99
OPTION = "candidate_score_permutation"


def permutation_spec(options):
    spec = dict(options or {}).get(OPTION)
    if spec is None:
        return None
    if (not isinstance(spec, dict) or set(spec) != {"policy", "repeats", "control_arm_id"}
            or spec.get("policy") != POLICY or type(spec.get("repeats")) is not int
            or spec["repeats"] != REPEATS or not isinstance(spec.get("control_arm_id"), str)
            or not spec["control_arm_id"]):
        raise ValueError("Invalid fixed 99-repeat candidate-score permutation contract")
    return dict(spec)
