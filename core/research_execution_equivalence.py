"""Content-pinned proof for reviewed execution-only preparation changes.

Each pair is an exact current implementation and its unchanged numerical
predecessor, not a scientific version or a general source-hash bypass. Unknown
code keeps its own identity. Prices, inputs, teachers, samples and all other
execution dependencies remain independently verified by their original owners.
"""
from __future__ import annotations

# Baseline ZIP SHA256: 3aae4d710c451d519245c2c1d204ffc235bcc453497f607b4078751659c21c6e
_REVIEWED = {
    "filters/breakout_quality/trade_path_label.py": {
        "file_sha256": "a9da0254197bc8bae01378b237d226710707647ed60f52406c0ad2f7c422e724",
        "original_file_sha256": "c64d7d40e5ac7a26364a2be1ccaf34a756a6013d239e5eda151094d48faca202",
        "calculations": {
            "15b24fead87d243ce3af92da7715b03d04c8b56992c9a27d21cc012c042663ae": "e62c85ac5f2b0e109c2258063fec76d81f8679091f31e5b4c2b618c8300e0e4a"
        }
    },
    "filters/breakout_quality/daily_trade40_target.py": {
        "file_sha256": "4e49ee3ea59315237ff2971861c71993a888f1bea3a816a2a94454f03a77cf37",
        "original_file_sha256": "4df90ba6d95a22cd0a78845141dc6bfb91683e7fa2ecc612cafa6f66e6bd3370",
        "calculations": {
            "89c63e6560669ef7edfdbd67dd18351da1ba4112f423ca324739987c6cffa8f4": "5bff63da3d8d1d7f04c06dddfd3c19735224019f7f7d5635f86de64c501c07ab",
            "536e268739713cb59124741bc6084ee118be1481730ba7117c6b8e8742697dc4": "06ceb265c4004fba0dfd5af63ee3f1fe8b9fd8f1f3c4c88eed3391e9f2defcbc"
        }
    },
    # Audited additions of TrueDaily-only model profiles/target policies.
    # Existing Oracle price-provider contracts, eligibility and runtime
    # configuration projections are unchanged from the prior baseline.
    # Exact source bytes are pinned: any subsequent change hashes normally.
    "core/breakout_quality_registry.py": {
        "file_sha256": "b06df905611f7d10db03683cacd3b69ec11a13127fdb3cff0582b3e6d27e8a33",
        "original_file_sha256": "917b0e84a7506a578109d110b239645e14c9c3c14a6daafe3523f673c23d015e",
        "calculations": {},
    },
    "core/breakout_quality_runtime.py": {
        "file_sha256": "3ba08563d94a60a0ad93e44387dbd67c0380c42a3450cd258a84102354d4a630",
        "original_file_sha256": "0095a9a8d8d726cce05213e0237cb401ffb073fe4b6f84c1b3dfb89bc4f1131f",
        "calculations": {},
    },
    "config/breakout_quality.py": {
        "file_sha256": "e66bd41bab7a60cbe4d9b96c8a11f9b146b261616d96a15f3921df278472d391",
        "original_file_sha256": "3c0323d35cf71c71b70211f9b65c24e158251353f0a39b6f5b3067e38451f004",
        "calculations": {},
    },
    "filters/breakout_quality/daily_ranker_data.py": {
        "file_sha256": "32a6c04a2e7024fb128e459b9d87e5ba9f3c58db73034173103cdada72b0a20e",
        "original_file_sha256": "65338ef9511e2739fccda5db90bbdc65893f63d5b97c968ef3e50fa51963c2c6",
        "calculations": {}
    }
}

# Newly imported solely to bound execution resources. Neither module supplies
# a feature, sample, Target, fee, order, seed or fitting parameter.
RESOURCE_ONLY_DEPENDENCIES = frozenset({
    "core/research_execution_resources.py", "config/runtime.py",
})


# A research-menu model switch is not a price/feature/Target change.
# Proof anchor: test-branch-1_20261107_072649_85be9915.zip
# (SHA256 d36e32296daed6bee8f4a0b0893fd00de2a3c1c4fd3e4316ecfbc73ab999e6e7).
# The AST of that source and this retirement is identical after replacing ONLY
# the literal (model ID, executable profile ID) research selection tuple.
# Any other executable config edit falls through to its real physical SHA.
_RESEARCH_SELECTION_ONLY_CONFIG_AST_SHA256 = (
    "c3c118fe43687589e554a52bc5eed404ca93c7750abe48787b013e889a0b5105"
)
_RESEARCH_SELECTION_ONLY_CONFIG_BASELINE_SHA256 = (
    "a5368e77feca79108465c8a25131f3c030433ed3fcf40a82c7c481c792f33380"
)


def _certified_model_selection_only_config_sha256(relative: str, actual: str) -> str:
    """Preserve the price source identity for an AST-proven model-menu choice.

    No arbitrary model-selection expressions can receive this equivalence:
    the sole exempt assignment must be a side-effect-free literal string pair.
    The whole remaining module AST, including price-related settings, is pinned.
    """
    if relative != "config/breakout_quality.py":
        return actual
    import ast
    from hashlib import sha256
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / relative
    if not source.is_file() or sha256(source.read_bytes()).hexdigest() != actual:
        return actual
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    count = 0
    for statement in tree.body:
        if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or statement.targets[0].id != "BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE"):
            continue
        value = statement.value
        if (not isinstance(value, ast.Tuple) or len(value.elts) != 2
                or not all(isinstance(item, ast.Constant) and isinstance(item.value, str)
                           and item.value for item in value.elts)):
            return actual
        statement.value = ast.Tuple(
            elts=[ast.Constant(value="__SELECTED_MODEL_ID__"),
                  ast.Constant(value="__SELECTED_PROFILE_ID__")],
            ctx=ast.Load(),
        )
        count += 1
    if count != 1:
        return actual
    selection_free_ast = ast.dump(tree, annotate_fields=True, include_attributes=False)
    if sha256(selection_free_ast.encode("utf-8")).hexdigest() == _RESEARCH_SELECTION_ONLY_CONFIG_AST_SHA256:
        return _RESEARCH_SELECTION_ONLY_CONFIG_BASELINE_SHA256
    return actual


def reviewed_execution_source_sha256(relative: str, actual: str) -> str:
    normalized = _certified_model_selection_only_config_sha256(relative, actual)
    if normalized != actual:
        return normalized
    proof = _REVIEWED.get(relative)
    if proof is not None and actual == proof["file_sha256"]:
        return proof["original_file_sha256"]
    return actual


def reviewed_calculation_sha256(relative: str, file_sha256: str, syntax_sha256: str) -> str:
    proof = _REVIEWED.get(relative)
    if proof is not None and file_sha256 == proof["file_sha256"]:
        return proof["calculations"].get(syntax_sha256, syntax_sha256)
    return syntax_sha256
