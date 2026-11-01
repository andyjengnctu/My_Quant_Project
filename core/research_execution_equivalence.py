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


def reviewed_execution_source_sha256(relative: str, actual: str) -> str:
    proof = _REVIEWED.get(relative)
    if proof is not None and actual == proof["file_sha256"]:
        return proof["original_file_sha256"]
    return actual


def reviewed_calculation_sha256(relative: str, file_sha256: str, syntax_sha256: str) -> str:
    proof = _REVIEWED.get(relative)
    if proof is not None and file_sha256 == proof["file_sha256"]:
        return proof["calculations"].get(syntax_sha256, syntax_sha256)
    return syntax_sha256
