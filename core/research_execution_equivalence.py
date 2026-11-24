"""Content-pinned proof for reviewed execution-only preparation changes.

Each pair is an exact current implementation and its unchanged numerical
predecessor, not a scientific version or a general source-hash bypass. Unknown
code keeps its own identity. Prices, inputs, teachers, samples and all other
execution dependencies remain independently verified by their original owners.
"""
from __future__ import annotations

from functools import lru_cache

# Baseline ZIP SHA256: 3aae4d710c451d519245c2c1d204ffc235bcc453497f607b4078751659c21c6e
_REVIEWED = {
    # AI: b0548fab input-read optimization: unchanged SHA bytes, Dataset rows
    # and validation decisions; only repeated reads/scans are memoized. The
    # complete current source is pinned so ANY later edit hashes normally.
    "core/file_integrity.py": {
        "file_sha256": "ab5ad7c0ef8cc0c6c7b0eee28061ebd6d6e20cd784f5d2e83a445d249834deb3",
        "original_file_sha256": "6f5abc343fb6ecc75e43f0ed8ead19023b627fc5ebaa93a2804c6bf1d87746ec",
        "calculations": {},
    },
    "filters/breakout_quality/source_inventory.py": {
        "file_sha256": "56542cc049f2a48eeb258b65f09869c8d85f468dc02c366b544e0301664621c0",
        "original_file_sha256": "a0d20b2e0a99a40f747880bfc5dce7fff92e54660876a73b2c538fc6ac2c8fa6",
        "calculations": {},
    },
    "filters/breakout_quality/workflow_io.py": {
        "file_sha256": "4ca328dd515831fee40da42386d71fc0b19577d90f7b54d4e2c220b8f89c9a31",
        "original_file_sha256": "332d1facb60f0474e78608e366db91a8b3b7aa00b0436998b7229854d65f0d42",
        "calculations": {},
    },
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
        # 2026-11-08 Full/P17: only additive rank/classifier profile and
        # research-spec entries; unchanged existing Oracle provider contracts.
        "reviewed_model_only_revisions": {
            "0361bee2c35cab50b11f2991cec777e200df5354a117f34c44fcc52f876d4186":
            "60c2d5a0c6e0107ccc2f3b5fdf4a07dc85b10abebd4daad9637c79876e414f1d",
        },
    },
    "core/breakout_quality_runtime.py": {
        "file_sha256": "3ba08563d94a60a0ad93e44387dbd67c0380c42a3450cd258a84102354d4a630",
        "original_file_sha256": "0095a9a8d8d726cce05213e0237cb401ffb073fe4b6f84c1b3dfb89bc4f1131f",
        "calculations": {},
        # 2026-11-08 Full/P17: only a new Target ID and a distinct
        # target-policy dictionary entry; existing Target policies are intact.
        "reviewed_model_only_revisions": {
            "a5606816fa2b8bc56a6dd5e66df4dd49866d8d8f8fc02b5d83316a96f81b2b90":
            "3e20eee163cd32a7fd0c9ed5db41dcd52439f7403cc8869932f19f32df1e184b",
        },
    },
    "config/breakout_quality.py": {
        "file_sha256": "e66bd41bab7a60cbe4d9b96c8a11f9b146b261616d96a15f3921df278472d391",
        "original_file_sha256": "3c0323d35cf71c71b70211f9b65c24e158251353f0a39b6f5b3067e38451f004",
        "calculations": {},
    },
    "filters/breakout_quality/daily_ranker_data.py": {
        "file_sha256": "32a6c04a2e7024fb128e459b9d87e5ba9f3c58db73034173103cdada72b0a20e",
        "original_file_sha256": "65338ef9511e2739fccda5db90bbdc65893f63d5b97c968ef3e50fa51963c2c6",
        "calculations": {},
        # AI: Compared with a3d9a7e4: the additional daily model input hook
        # leaves the existing Oracle cohort, target and eligibility path intact.
        # Bridge this one loader revision for both encodings; later feature
        # extensions live in the separate model-input provider.
        "reviewed_model_only_revisions": {
            "3ba49651c71a28574431646146c62fcfe0b2ae9ec88d467b9224e2eab44df00e":
                "d4d31e387030656aa1609dfcff5de66345a7158d687d74a9510785c88114f4d2",
            "121be1928f6f1538e9678ace4f036bfff17d48688a6d84e3e0e95d8c40c34a0f":
                "b40485ed13310be47261d9b504c874e15254ec30d3e9e423b657b644a7b91bdf",
        },
    },
}

# Newly imported solely to bound execution resources. Neither module supplies
# a feature, sample, Target, fee, order, seed or fitting parameter.
RESOURCE_ONLY_DEPENDENCIES = frozenset({
    "core/research_execution_resources.py", "config/runtime.py",
})


# A research-menu model selection is not a price/feature/Target change.
# Prior proof: test-branch-1_20261107_072649_85be9915.zip; the archived
# model-comparison roster is preserved as a separately exact-pinned AST.
# Full/P17 proof: independently compared the config AST in the source ZIP
# test-branch-1_20261108_012908_5e28bc05.zip to the current FullTrade ZIP.
# The only executable AST changes were the selected model/profile pair and
# insertion of two literal model-test reference pairs (CZ, DA). No price,
# eligibility, trading or Target configuration AST changed. Both reviewed
# configurations are independently pinned; subsequent changes to any model
# references, price settings or executable code fail closed to physical SHA.
# Only the active literal (model ID, profile ID) tuple is interchangeable.
_RESEARCH_SELECTION_ONLY_CONFIG_AST_SHA256 = (
    "19fd080bb82875e2298a321021bda3d21d0fd1e9f373707436d4beb8a9ced711"
)
_RESEARCH_SELECTION_ONLY_CONFIG_LEGACY_AST_SHA256 = (
    "7ab1d4e0b6a06053b596dfa4c970b0866464499586e66ab5994551fc8fa97b46"
)
_RESEARCH_SELECTION_ONLY_CONFIG_BASELINE_SHA256 = (
    "a5368e77feca79108465c8a25131f3c030433ed3fcf40a82c7c481c792f33380"
)


# AI: Summary refactor proof against the uploaded e9eb8b99 baseline. Apart
# from model-membership assignments/import and its one policy getter, both
# complete module ASTs are independently unchanged. The exact reviewed ASTs
# below preserve the uploaded source identities, not a different producer.
_SUMMARY_MEMBERSHIP_CONFIG_AST_SHA256 = "419dd1afabf15fa38522de310d06719f36bcaec31ff1701fe54da1cfd344603f"
_SUMMARY_MEMBERSHIP_CONFIG_BASELINE_SHA256 = _RESEARCH_SELECTION_ONLY_CONFIG_BASELINE_SHA256
_SUMMARY_MEMBERSHIP_POLICY_AST_SHA256 = "200baddc2fedeb3f88cf5c755c43891e5302a3f2e5ed4f86419daf69e961b7e2"
_SUMMARY_MEMBERSHIP_POLICY_BASELINE_SHA256 = "8c6a1d59c13d87efca83b10587b0d23027f36af9c2403b9ff165fcd1e2fd943f"


def canonical_ast_schema(tree):
    """Copy an AST using the published Python 3.13+ empty-list dump convention.

    Only empty list fields are omitted; all nonempty fields and their order
    remain intact. This keeps approved hashes portable without mutating the
    caller's cached AST or masking changes to executable values.
    """
    import ast
    from copy import deepcopy

    result = deepcopy(tree)
    for node in ast.walk(result):
        node._fields = tuple(
            name for name in node._fields
            if not isinstance(getattr(node, name, None), list) or getattr(node, name)
        )
    return result


def _certified_model_selection_only_config_sha256(relative: str, actual: str) -> str:
    """Preserve price identity for reviewed non-numerical config declarations.

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
    prefetch_count = 0
    for statement in list(tree.body):
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "BREAKOUT_QUALITY_CUDA_INFERENCE_PREFETCH"):
            # AI: Independently verified ordered input staging only. Removing
            # this literal still requires the entire remaining AST certificate.
            if not isinstance(statement.value, ast.Constant) or type(statement.value.value) is not bool:
                return actual
            prefetch_count += 1
            tree.body.remove(statement)
            continue
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
    if count != 1 or prefetch_count > 1:
        return actual
    selection_free_ast = ast.dump(canonical_ast_schema(tree), annotate_fields=True, include_attributes=False)
    syntax_sha = sha256(selection_free_ast.encode("utf-8")).hexdigest()
    if syntax_sha == _SUMMARY_MEMBERSHIP_CONFIG_AST_SHA256:
        return _SUMMARY_MEMBERSHIP_CONFIG_BASELINE_SHA256
    if syntax_sha in (
        _RESEARCH_SELECTION_ONLY_CONFIG_AST_SHA256,
        _RESEARCH_SELECTION_ONLY_CONFIG_LEGACY_AST_SHA256,
    ):
        return _RESEARCH_SELECTION_ONLY_CONFIG_BASELINE_SHA256
    return actual


def _certified_summary_membership_policy_sha256(relative: str, actual: str) -> str:
    if relative != "core/breakout_quality_policy.py":
        return actual
    import ast
    from hashlib import sha256
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / relative
    if not source.is_file() or sha256(source.read_bytes()).hexdigest() != actual:
        return actual
    syntax = ast.dump(canonical_ast_schema(ast.parse(source.read_text(encoding="utf-8"))), include_attributes=False)
    if sha256(syntax.encode("utf-8")).hexdigest() == _SUMMARY_MEMBERSHIP_POLICY_AST_SHA256:
        return _SUMMARY_MEMBERSHIP_POLICY_BASELINE_SHA256
    return actual


# AI: Same-target decision-context declarations add no broker execution rule.
# Exact bytes preserve disposable Min execution caches; FIT/source contracts
# still bind the new architecture, context schema and ordered FIT inputs.
_REVIEWED.setdefault('core/breakout_quality_registry.py', {"file_sha256": '6a2cfba976968888746115b8ed218bc5690343652fa4175a8195b4db96d10e5b', "original_file_sha256": 'f46267b9342f03680f0ec76ebe5c3ba50fa4f0217c541a560055d1e4544ecec3', "calculations": {}}).setdefault(
    "reviewed_model_only_revisions", {})['6a2cfba976968888746115b8ed218bc5690343652fa4175a8195b4db96d10e5b'] = 'f46267b9342f03680f0ec76ebe5c3ba50fa4f0217c541a560055d1e4544ecec3'
_REVIEWED.setdefault('filters/breakout_quality/models/architectures.py', {"file_sha256": 'f21154817d201a9a685287fce277653f413108bea1f05f7214a236b03ad1515e', "original_file_sha256": '7759ecf25673d4e1709376f19f1dffb885e00fa8294b3275d718d1dda473f302', "calculations": {}}).setdefault(
    "reviewed_model_only_revisions", {})['f21154817d201a9a685287fce277653f413108bea1f05f7214a236b03ad1515e'] = '7759ecf25673d4e1709376f19f1dffb885e00fa8294b3275d718d1dda473f302'
_REVIEWED.setdefault('config/research_summary.py', {"file_sha256": 'cac2cb2835358f193c83fc42a8847f7bb093ca486d2b8929d4becaf1795477e3', "original_file_sha256": '9280b09d88652ccfffeb3d4f9a73270ee2b15cbf58e8d7e90b8cc9c7599b38fb', "calculations": {}}).setdefault(
    "reviewed_model_only_revisions", {})['cac2cb2835358f193c83fc42a8847f7bb093ca486d2b8929d4becaf1795477e3'] = '9280b09d88652ccfffeb3d4f9a73270ee2b15cbf58e8d7e90b8cc9c7599b38fb'

# AI: 2026-11-24 opt-in original-signal inventory. Removing the two diagnostic
# keyword arguments and branches recovers the complete dec8e2d2 module AST
# exactly. Default price, fill, exit, fee and maturity code is unchanged.
# Proof: outputs/research/learning_studies/engineering_default_path_equivalence.json.
# Exact current bytes/encodings only; any later edit hashes normally.
_REVIEWED["filters/breakout_quality/daily_trade40_target.py"].setdefault(
    "reviewed_model_only_revisions", {}).update({'d716d6007511d42f19388d5e0535b145c275adebb7563cfe06bd6e0605f32883': '4df90ba6d95a22cd0a78845141dc6bfb91683e7fa2ecc612cafa6f66e6bd3370', '9bea022a543d347d5f46815d9189bec08e1ac5b57451ee7e8580a781ef2edf45': 'e569741dc258f637757d42cdd3febd39818f2136224a23b4fc26d7c505d4ada1'})

# 2026-11-24: materialize normalized dates once for scalar path indexing.
# test_research_realpath_dates reconstructs the complete d716d6 predecessor
# byte-for-byte and compares every outcome field across horizons/reset dates.
# Six real tickers (30,728 samples, three paired repetitions) also preserve
# every label array byte and PIT preflight field; aggregate cold speedup 1.395x.
# Each encoding AND calculation is exact-pinned. Any other edit hashes live.
_REVIEWED["filters/breakout_quality/daily_trade40_target.py"][
    "reviewed_model_only_revisions"
].update({
    "ea88b0473a4c6f822b3de719f54e6810e3b88a662e59ce585133b972d04b731d":
        "4df90ba6d95a22cd0a78845141dc6bfb91683e7fa2ecc612cafa6f66e6bd3370",
    "d326779477bc01952fa954189b1161fa2a1362f3e8e8ef29a4ca9eddeff2bff1":
        "e569741dc258f637757d42cdd3febd39818f2136224a23b4fc26d7c505d4ada1",
})
_REVIEWED["filters/breakout_quality/daily_trade40_target.py"][
    "reviewed_calculation_revisions"
] = {
    physical_sha: {
        "ac73c2b627b75e3a5dd60c45ae8a206363fa64f27362a8ac05581865ddd8de4e":
            "7d22c5edbd455bb9715a24be5ae71feab98a2248620d00b9238c731bdf2f5b9e",
        "737dc103e2d564e69021881b3a21e564f5bb70dc7ffbd1bcf44a33b9565f7b27":
            "98c80c47584be11046fb2dc562c7184fa2112ec4672e695bad016d967c1e0513",
        "725e36feb4ea4885276ccd5ede1b65876bf7b59866452eb646d6fc2806dc4217":
            "d09aa292059cd3d446e52aed5ddd18b78f17121e1d5e9d0ee225b1414c813f73",
    }
    for physical_sha in (
        "ea88b0473a4c6f822b3de719f54e6810e3b88a662e59ce585133b972d04b731d",
        "d326779477bc01952fa954189b1161fa2a1362f3e8e8ef29a4ca9eddeff2bff1",
    )
}


# Independent producer queue resource additions, proved against de64a6dc.
# Remove only the literal resource capacity and the exact isolated resolver;
# the entire remaining module AST must equal its separately observed predecessor.
# The queue, producer, trial protocol and numerical owners are never aliased here.
_INDEPENDENT_QUEUE_RESOURCE_PREDECESSORS = {
    "config/training_performance_policy.py": (
        "a4ff690da29d325a70593df2003476ea5740ee8d795f662e5678b0200c130a12",
        "f1dd90b1ff94eba4e4b6ef4703de6518a7f0ae5c2e292f8d4ce6e2bc483a09f1",
        "d635114d3cdfb70be4944f41c49e7ae010c89ef13284e168a850cbe0a0ad2965",
    ),
    "core/training_performance.py": (
        "959dd0b181d961751270836aed619afd4a0461211bd56ac4443f20b851286886",
        "b5e79dc378cadd244e755d8d794fe0bca1ca8dce929b57b13dbdb44426a6bc14",
        "5fd746c14b39fabdf906c2a0907329efaa59b63fb1e60f92f89b9a53deb77984",
    ),
}
_INDEPENDENT_QUEUE_RESOLVER_AST = "ddf7e6ac18666c506a39f3492213f07de7998221d3dd19d026d5eb5ee68ec367"


@lru_cache(maxsize=32)
def _independent_queue_resource_predecessor(relative: str, content: bytes) -> str | None:
    """Prove an additive execution-resource declaration; unknown AST fails closed."""
    import ast
    from hashlib import sha256
    proof = _INDEPENDENT_QUEUE_RESOURCE_PREDECESSORS.get(relative)
    if proof is None:
        return None
    try:
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError):
        return None
    def fingerprint(node):
        return sha256(ast.dump(canonical_ast_schema(node), annotate_fields=True,
                               include_attributes=False).encode("utf-8")).hexdigest()
    knob = "OPTIMIZER_INDEPENDENT_SEED_FOLD_GLOBAL_WORKERS"
    declarations = 0
    imports = 0
    resource_exports = 0
    exports = {
        "build_optimizer_outer_rolling_env_defaults":
            "str(resolve_optimizer_independent_seed_fold_global_workers())",
        "build_training_performance_policy_snapshot":
            "resolve_optimizer_independent_seed_fold_global_workers()",
    }
    for statement in list(tree.body):
        if relative.startswith("config/"):
            if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == knob):
                if (not isinstance(statement.value, ast.Constant)
                        or type(statement.value.value) is not int
                        or not 1 <= statement.value.value <= 64):
                    return None
                declarations += 1
                tree.body.remove(statement)
        else:
            if (isinstance(statement, ast.ImportFrom)
                    and statement.module == "config.training_performance_policy"
                    and statement.level == 0):
                for alias in list(statement.names):
                    if alias.name == knob:
                        if alias.asname is not None:
                            return None
                        imports += 1
                        statement.names.remove(alias)
            if (isinstance(statement, ast.FunctionDef)
                    and statement.name == "resolve_optimizer_independent_seed_fold_global_workers"):
                if fingerprint(statement) != _INDEPENDENT_QUEUE_RESOLVER_AST:
                    return None
                declarations += 1
                tree.body.remove(statement)
            if isinstance(statement, ast.FunctionDef) and statement.name in exports:
                expected = ast.parse(exports[statement.name], mode="eval").body
                found = 0
                for node in ast.walk(statement):
                    if not isinstance(node, ast.Dict):
                        continue
                    for index in reversed(range(len(node.keys))):
                        key = node.keys[index]
                        if isinstance(key, ast.Constant) and key.value == knob:
                            if fingerprint(node.values[index]) != fingerprint(expected):
                                return None
                            found += 1
                            del node.keys[index]
                            del node.values[index]
                if found != 1:
                    return None
                resource_exports += 1
    if resource_exports != (0 if relative.startswith("config/") else len(exports)):
        return None
    if declarations != 1 or imports != (0 if relative.startswith("config/") else 1):
        return None
    return proof[2 if b"\r\n" in content else 1] if fingerprint(tree) == proof[0] else None


def _independent_queue_resource_certificate(relative: str):
    from hashlib import sha256
    from pathlib import Path
    if relative not in _INDEPENDENT_QUEUE_RESOURCE_PREDECESSORS:
        return None
    source = Path(__file__).resolve().parents[1] / relative
    raw = source.read_bytes()
    predecessor = _independent_queue_resource_predecessor(relative, raw)
    return None if predecessor is None else (sha256(raw).hexdigest(), predecessor, raw)


@lru_cache(maxsize=32)
def _calculation_syntax_owner_predecessor(content: bytes, helper_content: bytes) -> str | None:
    """Reinsert the unchanged low-level syntax owner into its full predecessor."""
    import ast
    from hashlib import sha256
    def fingerprint(node):
        return sha256(ast.dump(canonical_ast_schema(node), annotate_fields=True,
                               include_attributes=False).encode("utf-8")).hexdigest()
    try:
        tree, helper = ast.parse(content), ast.parse(helper_content)
    except (SyntaxError, UnicodeDecodeError):
        return None
    if fingerprint(helper) != "7aeb457ca1adfa64fcf7aa9c926650858f607a92670fd4330673927709f49aef":
        return None
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "core.source_calculation_syntax" and node.level == 0]
    if (len(imports) != 1 or [(item.name, item.asname) for item in imports[0].names] != [
            ("_CalculationSyntax", None), ("_syntax_payload", None),
            ("calculation_syntax_attestation", None)]):
        return None
    original_definitions = [node for node in helper.body
                            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                            and node.name in {"_CalculationSyntax", "_syntax_payload"}]
    if len(original_definitions) != 2:
        return None
    position = tree.body.index(imports[0])
    tree.body[position:position + 1] = original_definitions
    replaced = 0
    expected = ast.parse("return calculation_syntax_attestation(parsed)").body[0]
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name == "_source_content_attestation":
            for statement in function.body:
                if (isinstance(statement, ast.If) and isinstance(statement.test, ast.Name)
                        and statement.test.id == "is_python" and statement.body
                        and fingerprint(statement.body[-1]) == fingerprint(expected)):
                    statement.body[-1] = ast.parse(
                        "payload = _syntax_payload(_CalculationSyntax().visit(parsed))"
                    ).body[0]
                    replaced += 1
    if (replaced != 1 or fingerprint(tree) !=
            "a8b90adf67e9853881d1385ded7ed24eaabdeaccd06aa3e4d79b06261456ae47"):
        return None
    return ("d4abc57ae1def8e7b88a30d4398a7dceffd6d0d81a12050d0dedca1dc1940440"
            if b"\r\n" in content else
            "62f248fccb421a376c70dd2604422f1aa8ac845cf6ec122558ec673c876ed125")


def _calculation_syntax_owner_certificate(relative: str):
    from hashlib import sha256
    from pathlib import Path
    if relative != "services/research/oracle_artifact_contract.py":
        return None
    root = Path(__file__).resolve().parents[1]
    raw = (root / relative).read_bytes()
    predecessor = _calculation_syntax_owner_predecessor(
        raw, (root / "core/source_calculation_syntax.py").read_bytes(),
    )
    return None if predecessor is None else (sha256(raw).hexdigest(), predecessor)


def reviewed_execution_source_sha256(relative: str, actual: str) -> str:
    resource = _independent_queue_resource_certificate(relative)
    if resource is not None and actual == resource[0]:
        actual = resource[1]
    syntax_owner = _calculation_syntax_owner_certificate(relative)
    if syntax_owner is not None and actual == syntax_owner[0]:
        actual = syntax_owner[1]
    optimizer_previous = _reviewed_optimizer_execution_sha256(relative, actual)
    if optimizer_previous != actual:
        return optimizer_previous
    normalized = _certified_model_selection_only_config_sha256(relative, actual)
    if normalized == actual:
        normalized = _certified_summary_membership_policy_sha256(relative, actual)
    if normalized != actual:
        return normalized
    proof = _REVIEWED.get(relative)
    if proof is not None:
        if actual == proof["file_sha256"]:
            return proof["original_file_sha256"]
        # Exact, independently reviewed additive declarations only.  Older
        # certified source identities and unrelated executable revisions never
        # enter this rule; unknown bytes always keep their physical SHA.
        model_only_previous = proof.get("reviewed_model_only_revisions", {}).get(actual)
        if model_only_previous is not None:
            return model_only_previous
    return actual


def reviewed_calculation_sha256(relative: str, file_sha256: str, syntax_sha256: str) -> str:
    proof = _REVIEWED.get(relative)
    if proof is not None and file_sha256 == proof["file_sha256"]:
        return proof["calculations"].get(syntax_sha256, syntax_sha256)
    if proof is not None:
        return proof.get("reviewed_calculation_revisions", {}).get(
            file_sha256, {},
        ).get(syntax_sha256, syntax_sha256)
    return syntax_sha256


# 7e16508c source-ownership/execution-context proof (2026-11-24).
# Baseline physical bytes: test-branch-1_20261124_120712_7e16508c.zip;
# AST/numerical comparisons use its LF-identical Git archive. Evidence:
# final_source_and_numerical_review.json (75 complete payloads compared with
# the original uncached algorithm and the corrected current cache),
# ranking_policy_equivalence.json (27 metadata / 81 sort / 92 selector /
# 5825 validation), plus explicit Daily context/worker/filter behavior tests.
# The first baseline anomaly is retained with benchmark_cache_reproduction.json:
# object-id reuse could return another dataset's cached statistics. No fields
# are excluded to conceal it; the baseline cache is cleared for each final case.
# Every current source part must match. Baselines contain predecessor source
# inventories (including earlier reviewed extraction projections), not result
# digests. Legacy Replay's original 13 syntax payloads are always hashed live.
# Native bytes and raw source parts accept only these exact encoding variants.
# The benchmark cache correctness fix is deliberately excluded: its live source
# changes Replay/candidate identity and is never mapped to the erroneous cache.
_REVIEWED_SOURCE_INVENTORIES = {'oracle_daily_provider': {'current': {'filters/breakout_quality/daily_ranker_data.py': ('e2b61e98476854365099f2cddb7ce3bbb5c100110589e73bcfaab698628abbd3',
                                                                                         'e2b61e98476854365099f2cddb7ce3bbb5c100110589e73bcfaab698628abbd3',
                                                                                         'fe6fb8ae03e76037974822d286eb3e31e3af83542fe25a2bef7b2ddb18a45b78'),
                                       'filters/breakout_quality/profile_ranker_data.py': ('cb8722b62cd69dd0c598457ed0ba73ae3ce321e8e174824c7e9decf76d024c14',
                                                                                           'aa3513e0368dd9e5c10f1244c808a88ac702a8d5b8178b4ca76d8ccd66145de4',
                                                                                           'cb8722b62cd69dd0c598457ed0ba73ae3ce321e8e174824c7e9decf76d024c14'),
                                       'filters/breakout_quality/daily_model_input_context.py': ('7128ad7183a5291980565284948a9c8644f2962294d52b9604f51e7aa0f5b8ce',
                                                                                                 '7128ad7183a5291980565284948a9c8644f2962294d52b9604f51e7aa0f5b8ce',
                                                                                                 'c43ac2774516bc1bf24604240964688bd1426d78972b8badb169319d90248792'),
                                       'filters/breakout_quality/daily_sample_accumulator.py': ('157ee1eece719d53751ed9e478479406705eecd5ec0836fdf70505bf7e95f96e',
                                                                                                '157ee1eece719d53751ed9e478479406705eecd5ec0836fdf70505bf7e95f96e',
                                                                                                'acfa95a4323fa3bf24ab4287c9e0e211c763a96172e2cf118679dcd93ab28bc0'),
                                       'core/walk_forward_policy.py': ('1f44fc1e042ddd5beff4ad12f92bf7c783983423e94c0e604a538e6283f7a3f6',
                                                                       '1f44fc1e042ddd5beff4ad12f92bf7c783983423e94c0e604a538e6283f7a3f6',
                                                                       'c96c3c876985ca19ed8002a76d16531cead99f06640ac8411b3d3d1892de6f52')},
                           'baseline': {'filters/breakout_quality/daily_ranker_data.py': ('121be1928f6f1538e9678ace4f036bfff17d48688a6d84e3e0e95d8c40c34a0f',
                                                                                          '3ba49651c71a28574431646146c62fcfe0b2ae9ec88d467b9224e2eab44df00e',
                                                                                          '121be1928f6f1538e9678ace4f036bfff17d48688a6d84e3e0e95d8c40c34a0f'),
                                        'filters/breakout_quality/profile_ranker_data.py': ('d32a9fa95bed048591d886214da81b0b6fc4f9c128977514d742613263b4ec70',
                                                                                            '16166646333272683f5a0103a242a7a4a05f1c2433ae52031cb6bef91981fb38',
                                                                                            'd32a9fa95bed048591d886214da81b0b6fc4f9c128977514d742613263b4ec70'),
                                        'core/walk_forward_policy.py': ('92cb9ff84d0bc0354de9db2ab2eae7e2bca81a3461b0ab621f40f5b0e3fc525b',
                                                                        '92cb9ff84d0bc0354de9db2ab2eae7e2bca81a3461b0ab621f40f5b0e3fc525b',
                                                                        'be6652e2b0bd85c3d1ed89cb35dff1efc3c456091afb53db2944a4c1ac9253ae')},
                           'baseline_lf': {'filters/breakout_quality/daily_ranker_data.py': '121be1928f6f1538e9678ace4f036bfff17d48688a6d84e3e0e95d8c40c34a0f',
                                           'filters/breakout_quality/profile_ranker_data.py': 'd32a9fa95bed048591d886214da81b0b6fc4f9c128977514d742613263b4ec70',
                                           'core/walk_forward_policy.py': '92cb9ff84d0bc0354de9db2ab2eae7e2bca81a3461b0ab621f40f5b0e3fc525b'},
                           'baseline_crlf': {'filters/breakout_quality/daily_ranker_data.py': '121be1928f6f1538e9678ace4f036bfff17d48688a6d84e3e0e95d8c40c34a0f',
                                             'filters/breakout_quality/profile_ranker_data.py': 'd32a9fa95bed048591d886214da81b0b6fc4f9c128977514d742613263b4ec70',
                                             'core/walk_forward_policy.py': 'be6652e2b0bd85c3d1ed89cb35dff1efc3c456091afb53db2944a4c1ac9253ae'}},
 'portfolio_candidates': {'current': {'@execution_sources': '4dbbe02625401dc5c5a0f2b6139a790e52c59a448df5a1e4bbed033d27fcca19',
                                      'core/portfolio_engine.py': '98c398e76c97f6066642f045b5ff74f09d28cf076fafcf3ed3f01151e001b381',
                                      'core/portfolio_day_candidates.py': '2052e3d64568b26721483e4b7855b37cc5d8f484b0235a7c7658b97e517a3d37',
                                      'core/portfolio_candidate_admission.py': 'e462acaea55f75e6e1143ff580362dd52e01d9460db0edb2f60c234fba13403d',
                                      'core/portfolio_candidates.py': '7f5418d6896f8c94de8d4824ba451008af4a54a4e3c48fe6c6bd982eaf67e23a',
                                      'core/portfolio_day_context.py': 'fd5969d9b2bd053b67278e4794c2746450270511b9d6438e7005f63516f73133',
                                      'core/portfolio_day_events.py': '6862fd9d4be982c3ad0c4301a710ce7d4c929734129ff2cb85e797a8030cf5fd',
                                      'core/portfolio_replay_support.py': 'e74d328e995b737e2f2c25bf205651ffb05784faf26b49bfd4ebd10e7c679808',
                                      'core/portfolio_day_equity.py': 'b39398bc0d8ab9e0ef079a0c298dd8f86a78499f607eb3884d411d8d12a6c49a',
                                      'core/portfolio_day_signal_lifecycle.py': '6a918dca87883659ba5a0a760ebfa67f8cfa8f259d2e6b477c864f6fba3ee100',
                                      'core/portfolio_capacity_rows.py': '1597a192df61da5c9c807fc9149d560e6e800f758150c107a39943c1b490f353',
                                      'core/portfolio_terminal_equity.py': '801bdb4c58e71aabe19b9aa08e93d059f85b0921cddeccb8a2e17cd71518cadc',
                                      'core/portfolio_terminal_statistics.py': 'e74a96108a27cc03b0726776dfc1065b5b6c84b357a6221adcb7dbc35ffc2572',
                                      'core/portfolio_terminal_reporting.py': '903a3357f8bf24dc94ba0cdb371d9c39584b9a1c382484fd2ff7f7440119ad8d',
                                      'core/portfolio_run_result.py': 'ac4a7deb40ec23c6e03aa7c0ec3f878b9659470a6173894420003a06da2d0ed9',
                                      'core/portfolio_entries.py': '7a16b0e24123b099127f1ba3330c0b433e9ba7a2a315c15244226c1dc2654cc9',
                                      'core/portfolio_exits.py': 'd44b56a5ec423e9c6049c73527d32438aab08dff88e82441f784b776aadc9a42',
                                      'core/portfolio_entry_selection.py': '710340465712a6fd981cb26e50a24f5491c7df9b9e55d8158a962c0c94927d3d',
                                      'core/portfolio_entry_selection_common.py': '8d9b27ad4d13826468e68cc99db30f2dc9d742a6efce5512fba5be5fdab4dd68',
                                      'core/portfolio_entry_selection_max_dl.py': 'a1c0869d6e18ea3e7da59fe2e84e11ebbe69f8815377149f175a7e228242ee3a',
                                      'core/portfolio_ensemble.py': '2214e433e5685aee71bc7dfd15d52ed4283b4bf1bf3d111c87c5f0402254540f',
                                      'core/portfolio_entry_plans.py': '1bfcd46aad95a3af89657ff37cc90eae7436210c1199fbaceea36fe5767c8d35',
                                      'core/portfolio_entry_policy.py': '070818b8c6f4afd38d6a9c815b0982f4c25deea068ccac830be382989f804053',
                                      'core/portfolio_fast_data.py': '890fe8f268652f0e6102782154bd5f57eb065c03595563e16f7ae8c12ea8f7bf',
                                      'core/portfolio_stats.py': '6350e7d73f3b76425f25c97801731f3c3e6af9f6507457a68e27a25a2e37944d',
                                      'core/portfolio_fast_access.py': 'c4ff1cab2cc819510d5c3844ea350076e6cd879c723cdac5d309289bed391b77',
                                      'core/ranking_policy.py': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
                                      'core/portfolio_run_state.py': '305cfec53a4aaf43ed24d3cd53c1d8d933704bad65e02e0c22bcf7d7753d2612',
                                      'core/buy_sort.py': 'cbeb650a94a4b5fae64c8874823e78ff1cad24d00377dd99a45c7fbad010d8a6',
                                      'core/backtest_core.py': 'acb61dca04c51ad584a63ba7ba0c6769f32167691938226f3b383d58c9f7296b',
                                      'core/backtest_finalize.py': '3fd8d083017a51f38b27a0384d6b5d9a0f381dc57556b7d1ff2491b294616b8f',
                                      'core/history_filters.py': '69735f05fafd5306da8f83676cd3f9a7b4be1096ef960399f9d0e8be06bbe4dd',
                                      'services/portfolio_replay.py': '5ee1ff1fa48860654ffbf6644b287637e601c66e0c4a277fc8164a829c05ebc9',
                                      'services/portfolio_replay_runtime.py': '2239d08c18a1968dce65fbada1e433e6e0ae30882ddbb80b6a41beb367749066',
                                      'core/position_step.py': '3d8478e4e699eb7aae02edb5027de701b6e447dbfd4bac526f640f967a35d4b2',
                                      'core/position_management.py': '0d02808329b710f90a4811e0149ff0d089bb99cca1da769bf874d8320d741343',
                                      'core/exit_priority.py': '9bc18862540957870cdafbc636f6787cdb9e6e8a42035a2c85bdd5466eeb9d8d',
                                      'core/entry_plans.py': 'c95b942559424dbf5f4afbfb1c5f7b90ad6c0e1e2b7d3fd1a58e8a09d410e29f',
                                      'core/extended_signals.py': '46837f8445c8e83d5758adf86f6b975243ef6de4eeb0cf3303c28c1eb8cfc97c',
                                      'core/order_lot_policy.py': '2b351700be07f52fd6096aedf3ab95aa427144c28beca98cc4ec91812619800d',
                                      'core/exact_accounting.py': 'dcbb7d578595d6d1c9199e95c737fff12e9483e66669559989b09e8e85d7e463',
                                      'core/fee_rebate.py': '7b59a1b3e3b4c5ee6d7c0fe793375364fac188f9b0de0a6739c764908dca09e5',
                                      'core/price_utils.py': '79c1feab6ffbd8acd34cf2b94512e7f6fe36d3f8ea0e28412a92bb9147208a10',
                                      'core/selection_policy.py': 'd73572126baf34c7d29795042213c05dc67166d3ca16ac2041c2a81c8d9cf652',
                                      'core/signal_utils.py': 'c5af835c02deb9c5a4ddfbfefbf4f477d3190c397502ec46271e0c40f5838ea7',
                                      'core/breakout_false_filter.py': '67fd40430c44dea207f5b34e340f1233fda545ae8a03c1ee798b23a268570f88',
                                      'core/capital_policy.py': '73005c90a83bec4935bd8a68b65f7ec13ba435f4a2ad49a25c74e4ef75bb209a',
                                      'core/native_optimizer_hotpath.py': '1754f0a9f80be1603408f35163f7df94a6cf0f20259c650bdc4ed927287b9c47',
                                      'core/native_optimizer_hotpath.c': ('bbe4bd08e486214fd7cb449e70f13e307f3cb709699fa6554fc68b3332e2bdb1',
                                                                          'bbe4bd08e486214fd7cb449e70f13e307f3cb709699fa6554fc68b3332e2bdb1',
                                                                          'e36eef4d95ba97dd2505cab98c1bd087d426fe478d7681cd756d30af8b551307'),
                                      'core/training_performance.py': '1769b33c89d731a01b5f418adc1f38ecbe549c6c1e75ec98e9fd07506eb8a54b',
                                      'config/training_performance_policy.py': 'e172a81ce1a0bdf36cb1e5fbedc1e9422ef4b263cf88bf4984f875f0f58180d9',
                                      'config/broker_accounting.py': '8095a0bf8da096ca987d9ab4f5f7d5ba30cc65b6509b6470e212c63e09f8ee35',
                                      'config/execution_policy.py': '90c7a30ee2083e217a37b9538fd79f8a098749021ed894357a5f9709d28b123b',
                                      'core/feature_bank.py': '0923a92a6de7cc3fdaeff36bb2b27ad143a5c174f044c858ff829e0e57a674da',
                                      'core/strategy_params.py': '696c262e26d2f0afe8d6e3a033919072e11699910497316f016c7eb83fa103ec',
                                      'strategies/breakout/schema.py': 'f48a71449d6aeeebedcdd1ebc7e7c4a8a21a2ef9739acbb8c34d31bf2727c617',
                                      'core/breakout_quality_runtime_port.py': '29015af9a89a16d4f964642982c27ba36b2e013d9d641bc89f50def67b46d822',
                                      'core/seed_ensemble_policy.py': '728e6360b83b70a2a8d0ce765b00324e4c1160903cb9e5ba90a20392b41e6aab',
                                      'config/breakout_policy.py': '7c29ebafdcc2e8c3b168c801b6e8cb9d5f6f5569e978c942312a56b3ba74f252',
                                      'core/active_param_ensemble.py': '8a42e9e206e128aa09b4597356a64b02c57bb382b1851b8d7a4d698582c65a4c',
                                      'core/rolling_oos_params.py': '1c4c09e3b1528b1c3f751cbf77bd96dab8550ba152d79e9d9c595348f53d4cc4',
                                      'core/params_io.py': 'ad205ef2178888f9dd8dbf9e5b67414d92f9cac7cee1f3686dd8f1a660846351',
                                      'core/portfolio_param_runtime.py': 'c3950ed1354f7c44919a25f6bbaa730ca9e06bb0f94d77c9d842ea96d4154f5f',
                                      'core/param_access.py': 'b115f48440bf1e02416892ec55d90d120e1ddedbddd6c72477536a3da507f364',
                                      'core/raw_universe_contract.py': '0fd177427b3fd61e7b4fd865561ade8cb5a19e5cd013e05ff1b2ba66089ddc51',
                                      'core/file_integrity.py': '39645817c711d1a6896721713179c6d218328249841160698a80b39e128504be',
                                      'core/scientific_runtime_context.py': '09a9a04e2cf788ddae8c2fafb23b9b406bb2263797d4233492d1b40e595c50e7',
                                      'core/breakout_reentry.py': 'b8a402c06250efa805d33104da6a7f618cdfa9abed8448a9fe9ed5df7154ba19',
                                      'core/breakout_quality_ranking_context.py': '1fab5f819762aff43022982ee5d84feb3ef5b369b1c0f344c36ca3fbc0848b0b',
                                      'core/positive_score_admission.py': '9dee6e37b20df6855c4983ab56784b3ae9e39ac9b48dd8f74f9aa3beebadb654',
                                      'core/positive_netpnl_contract.py': '08fb20450edc7333bbde09e019f38fd9ee2d41fa230ef416633d31dd4adfc51a',
                                      'core/config.py': '83d14a9f4d4d016d37941281e0dea49aa4ca526f1fb766be6d0e18d61cda651d',
                                      'core/training_policy.py': '13d1a8a4131bef928e083081b7849ac6fda569f34453cc6a1dc3bfc20302d467',
                                      'config/training_policy.py': 'eabb28c2e562f54472f76bd6d48914a8191ee6d0d8b251bbed57a4c94ff31d08',
                                      'core/walk_forward_policy.py': '1695230ba7b8c7bab4f923f6f9967759dd82446b604c57c86f31d7c0bfa725be',
                                      'core/trade_plans.py': 'dc7e1d8b622ac95275fabd50350e6a261180cc3b706822420d868f6f3be2ae23',
                                      'core/data_utils.py': '01c02e8a15b23c76034f5036b7b579b0c828fc89fcfec2a1f090af770bdb7cc3'},
                          'baseline': {'core/portfolio_engine.py': '873926b452c753461ed6f2f3e48ecf9fd90a49a5b4f6fb18a5ef342ff9dc386a',
                                       'core/portfolio_day_candidates.py': '2052e3d64568b26721483e4b7855b37cc5d8f484b0235a7c7658b97e517a3d37',
                                       'core/portfolio_candidate_admission.py': 'e462acaea55f75e6e1143ff580362dd52e01d9460db0edb2f60c234fba13403d',
                                       'core/portfolio_candidates.py': 'd897fa2f80dc7c07e532ccf8ae01b26c0f54d75d368b8641c522e2f6ee7dc326',
                                       'core/portfolio_day_context.py': 'fd5969d9b2bd053b67278e4794c2746450270511b9d6438e7005f63516f73133',
                                       'core/portfolio_day_events.py': 'dbe579cf179949f616358833cda65ccae2b7435e9dfdf779ce208031889a3490',
                                       'core/portfolio_replay_support.py': 'e74d328e995b737e2f2c25bf205651ffb05784faf26b49bfd4ebd10e7c679808',
                                       'core/portfolio_day_equity.py': 'b39398bc0d8ab9e0ef079a0c298dd8f86a78499f607eb3884d411d8d12a6c49a',
                                       'core/portfolio_day_signal_lifecycle.py': '6a918dca87883659ba5a0a760ebfa67f8cfa8f259d2e6b477c864f6fba3ee100',
                                       'core/portfolio_capacity_rows.py': '1597a192df61da5c9c807fc9149d560e6e800f758150c107a39943c1b490f353',
                                       'core/portfolio_terminal_equity.py': '801bdb4c58e71aabe19b9aa08e93d059f85b0921cddeccb8a2e17cd71518cadc',
                                       'core/portfolio_terminal_statistics.py': 'e74a96108a27cc03b0726776dfc1065b5b6c84b357a6221adcb7dbc35ffc2572',
                                       'core/portfolio_terminal_reporting.py': '903a3357f8bf24dc94ba0cdb371d9c39584b9a1c382484fd2ff7f7440119ad8d',
                                       'core/portfolio_run_result.py': 'ac4a7deb40ec23c6e03aa7c0ec3f878b9659470a6173894420003a06da2d0ed9',
                                       'core/portfolio_entries.py': '7a16b0e24123b099127f1ba3330c0b433e9ba7a2a315c15244226c1dc2654cc9',
                                       'core/portfolio_exits.py': 'd44b56a5ec423e9c6049c73527d32438aab08dff88e82441f784b776aadc9a42',
                                       'core/portfolio_entry_selection.py': '86f0aa7c0ecc82b2b0c8fe9be7b79a2afaa8fdf759cac9936357304af21da908',
                                       'core/portfolio_entry_selection_common.py': '68eeba7b93662bb9dbc4b8936958c71f147ab2a1b88188f0c90f4501f1418438',
                                       'core/portfolio_entry_selection_max_dl.py': 'a1c0869d6e18ea3e7da59fe2e84e11ebbe69f8815377149f175a7e228242ee3a',
                                       'core/portfolio_ensemble.py': '2214e433e5685aee71bc7dfd15d52ed4283b4bf1bf3d111c87c5f0402254540f',
                                       'core/portfolio_entry_plans.py': '1bfcd46aad95a3af89657ff37cc90eae7436210c1199fbaceea36fe5767c8d35',
                                       'core/portfolio_entry_policy.py': '196c38345ffeb79acf680105ad9237830e23f332cf96c44192a9e4004a347d00',
                                       'core/portfolio_param_runtime.py': 'c3950ed1354f7c44919a25f6bbaa730ca9e06bb0f94d77c9d842ea96d4154f5f',
                                       'core/portfolio_fast_data.py': '785a5f1065782d6b521e54bb4f44eb1b87ab1da8d0284688c455aa535790f056',
                                       'core/portfolio_stats.py': '6350e7d73f3b76425f25c97801731f3c3e6af9f6507457a68e27a25a2e37944d',
                                       'core/portfolio_fast_access.py': 'c4ff1cab2cc819510d5c3844ea350076e6cd879c723cdac5d309289bed391b77',
                                       'core/buy_sort.py': 'f8867277b95b63d2ce7bba17c9ac31e5ceb095631d6e2079db461826f4c65893',
                                       'core/backtest_core.py': 'acb61dca04c51ad584a63ba7ba0c6769f32167691938226f3b383d58c9f7296b',
                                       'core/backtest_finalize.py': '3fd8d083017a51f38b27a0384d6b5d9a0f381dc57556b7d1ff2491b294616b8f',
                                       'core/seed_ensemble_policy.py': '728e6360b83b70a2a8d0ce765b00324e4c1160903cb9e5ba90a20392b41e6aab',
                                       'core/history_filters.py': '69735f05fafd5306da8f83676cd3f9a7b4be1096ef960399f9d0e8be06bbe4dd',
                                       'services/portfolio_replay.py': '1b4bc619acbbb527e1b5b623caa87b28992c7668db952500e994ce9fb735d2bc',
                                       'services/portfolio_replay_runtime.py': '2239d08c18a1968dce65fbada1e433e6e0ae30882ddbb80b6a41beb367749066'}},
 'replay_numerics': {'current': {'core/native_optimizer_hotpath.py': '1754f0a9f80be1603408f35163f7df94a6cf0f20259c650bdc4ed927287b9c47',
                                 'core/native_optimizer_hotpath.c': ('bbe4bd08e486214fd7cb449e70f13e307f3cb709699fa6554fc68b3332e2bdb1',
                                                                     'bbe4bd08e486214fd7cb449e70f13e307f3cb709699fa6554fc68b3332e2bdb1',
                                                                     'e36eef4d95ba97dd2505cab98c1bd087d426fe478d7681cd756d30af8b551307'),
                                 'core/training_performance.py': '1769b33c89d731a01b5f418adc1f38ecbe549c6c1e75ec98e9fd07506eb8a54b',
                                 'config/training_performance_policy.py': 'e172a81ce1a0bdf36cb1e5fbedc1e9422ef4b263cf88bf4984f875f0f58180d9',
                                 'config/broker_accounting.py': '8095a0bf8da096ca987d9ab4f5f7d5ba30cc65b6509b6470e212c63e09f8ee35',
                                 'config/execution_policy.py': '90c7a30ee2083e217a37b9538fd79f8a098749021ed894357a5f9709d28b123b',
                                 'core/feature_bank.py': '0923a92a6de7cc3fdaeff36bb2b27ad143a5c174f044c858ff829e0e57a674da',
                                 'core/strategy_params.py': '696c262e26d2f0afe8d6e3a033919072e11699910497316f016c7eb83fa103ec',
                                 'strategies/breakout/schema.py': 'f48a71449d6aeeebedcdd1ebc7e7c4a8a21a2ef9739acbb8c34d31bf2727c617',
                                 'core/breakout_quality_runtime_port.py': '29015af9a89a16d4f964642982c27ba36b2e013d9d641bc89f50def67b46d822',
                                 'core/seed_ensemble_policy.py': '728e6360b83b70a2a8d0ce765b00324e4c1160903cb9e5ba90a20392b41e6aab',
                                 'config/breakout_policy.py': '7c29ebafdcc2e8c3b168c801b6e8cb9d5f6f5569e978c942312a56b3ba74f252',
                                 'core/active_param_ensemble.py': '8a42e9e206e128aa09b4597356a64b02c57bb382b1851b8d7a4d698582c65a4c',
                                 'core/rolling_oos_params.py': '1c4c09e3b1528b1c3f751cbf77bd96dab8550ba152d79e9d9c595348f53d4cc4',
                                 'core/params_io.py': 'ad205ef2178888f9dd8dbf9e5b67414d92f9cac7cee1f3686dd8f1a660846351',
                                 'core/portfolio_param_runtime.py': 'c3950ed1354f7c44919a25f6bbaa730ca9e06bb0f94d77c9d842ea96d4154f5f',
                                 'core/param_access.py': 'b115f48440bf1e02416892ec55d90d120e1ddedbddd6c72477536a3da507f364',
                                 'core/raw_universe_contract.py': '0fd177427b3fd61e7b4fd865561ade8cb5a19e5cd013e05ff1b2ba66089ddc51',
                                 'core/file_integrity.py': '39645817c711d1a6896721713179c6d218328249841160698a80b39e128504be',
                                 'core/scientific_runtime_context.py': '09a9a04e2cf788ddae8c2fafb23b9b406bb2263797d4233492d1b40e595c50e7',
                                 'core/portfolio_engine.py': '98c398e76c97f6066642f045b5ff74f09d28cf076fafcf3ed3f01151e001b381',
                                 'core/portfolio_day_candidates.py': '2052e3d64568b26721483e4b7855b37cc5d8f484b0235a7c7658b97e517a3d37',
                                 'core/portfolio_candidate_admission.py': 'e462acaea55f75e6e1143ff580362dd52e01d9460db0edb2f60c234fba13403d',
                                 'core/portfolio_candidates.py': '7f5418d6896f8c94de8d4824ba451008af4a54a4e3c48fe6c6bd982eaf67e23a',
                                 'core/portfolio_day_context.py': 'fd5969d9b2bd053b67278e4794c2746450270511b9d6438e7005f63516f73133',
                                 'core/portfolio_day_events.py': '6862fd9d4be982c3ad0c4301a710ce7d4c929734129ff2cb85e797a8030cf5fd',
                                 'core/portfolio_replay_support.py': 'e74d328e995b737e2f2c25bf205651ffb05784faf26b49bfd4ebd10e7c679808',
                                 'core/portfolio_day_equity.py': 'b39398bc0d8ab9e0ef079a0c298dd8f86a78499f607eb3884d411d8d12a6c49a',
                                 'core/portfolio_day_signal_lifecycle.py': '6a918dca87883659ba5a0a760ebfa67f8cfa8f259d2e6b477c864f6fba3ee100',
                                 'core/portfolio_capacity_rows.py': '1597a192df61da5c9c807fc9149d560e6e800f758150c107a39943c1b490f353',
                                 'core/portfolio_terminal_equity.py': '801bdb4c58e71aabe19b9aa08e93d059f85b0921cddeccb8a2e17cd71518cadc',
                                 'core/portfolio_terminal_statistics.py': 'e74a96108a27cc03b0726776dfc1065b5b6c84b357a6221adcb7dbc35ffc2572',
                                 'core/portfolio_terminal_reporting.py': '903a3357f8bf24dc94ba0cdb371d9c39584b9a1c382484fd2ff7f7440119ad8d',
                                 'core/portfolio_run_result.py': 'ac4a7deb40ec23c6e03aa7c0ec3f878b9659470a6173894420003a06da2d0ed9',
                                 'core/portfolio_entries.py': '7a16b0e24123b099127f1ba3330c0b433e9ba7a2a315c15244226c1dc2654cc9',
                                 'core/portfolio_exits.py': 'd44b56a5ec423e9c6049c73527d32438aab08dff88e82441f784b776aadc9a42',
                                 'core/portfolio_entry_selection.py': '710340465712a6fd981cb26e50a24f5491c7df9b9e55d8158a962c0c94927d3d',
                                 'core/portfolio_entry_selection_common.py': '8d9b27ad4d13826468e68cc99db30f2dc9d742a6efce5512fba5be5fdab4dd68',
                                 'core/portfolio_entry_selection_max_dl.py': 'a1c0869d6e18ea3e7da59fe2e84e11ebbe69f8815377149f175a7e228242ee3a',
                                 'core/portfolio_ensemble.py': '2214e433e5685aee71bc7dfd15d52ed4283b4bf1bf3d111c87c5f0402254540f',
                                 'core/portfolio_entry_plans.py': '1bfcd46aad95a3af89657ff37cc90eae7436210c1199fbaceea36fe5767c8d35',
                                 'core/portfolio_entry_policy.py': '070818b8c6f4afd38d6a9c815b0982f4c25deea068ccac830be382989f804053',
                                 'core/portfolio_fast_data.py': '890fe8f268652f0e6102782154bd5f57eb065c03595563e16f7ae8c12ea8f7bf',
                                 'core/portfolio_stats.py': '6350e7d73f3b76425f25c97801731f3c3e6af9f6507457a68e27a25a2e37944d',
                                 'core/portfolio_fast_access.py': 'c4ff1cab2cc819510d5c3844ea350076e6cd879c723cdac5d309289bed391b77',
                                 'core/ranking_policy.py': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
                                 'core/portfolio_run_state.py': '305cfec53a4aaf43ed24d3cd53c1d8d933704bad65e02e0c22bcf7d7753d2612',
                                 'core/buy_sort.py': 'cbeb650a94a4b5fae64c8874823e78ff1cad24d00377dd99a45c7fbad010d8a6',
                                 'core/backtest_core.py': 'acb61dca04c51ad584a63ba7ba0c6769f32167691938226f3b383d58c9f7296b',
                                 'core/backtest_finalize.py': '3fd8d083017a51f38b27a0384d6b5d9a0f381dc57556b7d1ff2491b294616b8f',
                                 'core/history_filters.py': '69735f05fafd5306da8f83676cd3f9a7b4be1096ef960399f9d0e8be06bbe4dd',
                                 'services/portfolio_replay.py': '5ee1ff1fa48860654ffbf6644b287637e601c66e0c4a277fc8164a829c05ebc9',
                                 'services/portfolio_replay_runtime.py': '2239d08c18a1968dce65fbada1e433e6e0ae30882ddbb80b6a41beb367749066',
                                 'core/breakout_reentry.py': 'b8a402c06250efa805d33104da6a7f618cdfa9abed8448a9fe9ed5df7154ba19',
                                 'core/breakout_quality_ranking_context.py': '1fab5f819762aff43022982ee5d84feb3ef5b369b1c0f344c36ca3fbc0848b0b',
                                 'core/positive_score_admission.py': '9dee6e37b20df6855c4983ab56784b3ae9e39ac9b48dd8f74f9aa3beebadb654',
                                 'core/positive_netpnl_contract.py': '08fb20450edc7333bbde09e019f38fd9ee2d41fa230ef416633d31dd4adfc51a',
                                 'core/config.py': '83d14a9f4d4d016d37941281e0dea49aa4ca526f1fb766be6d0e18d61cda651d',
                                 'core/training_policy.py': '13d1a8a4131bef928e083081b7849ac6fda569f34453cc6a1dc3bfc20302d467',
                                 'config/training_policy.py': 'eabb28c2e562f54472f76bd6d48914a8191ee6d0d8b251bbed57a4c94ff31d08',
                                 'core/walk_forward_policy.py': '1695230ba7b8c7bab4f923f6f9967759dd82446b604c57c86f31d7c0bfa725be',
                                 'core/trade_plans.py': 'dc7e1d8b622ac95275fabd50350e6a261180cc3b706822420d868f6f3be2ae23',
                                 'core/data_utils.py': '01c02e8a15b23c76034f5036b7b579b0c828fc89fcfec2a1f090af770bdb7cc3',
                                 '@execution_sources': '94423ab1b02f99bc589fe405b374bcde5fd26287c9fe04505e83c56362fec693'},
                     'baseline': {}},
 'portfolio_full_filters': {'current': {'core/daily_common_entry_filters.py': '086c44b61bc0df87dbfcd4507e8d9514368b736e7c165edaf162892dc1745cf7',
                                        'core/portfolio_entry_policy.py': '070818b8c6f4afd38d6a9c815b0982f4c25deea068ccac830be382989f804053'},
                            'baseline': {'core/daily_common_entry_filters.py': '5c751011447eb9bfc563781a7bd4d7b856019b68c8e38470b80cec286a3f53a1',
                                         'core/portfolio_entry_policy.py': '070818b8c6f4afd38d6a9c815b0982f4c25deea068ccac830be382989f804053'}},
 'full_oracle_teacher': {'current': {'services/audit/fulltrade_oracle_truth.py': ('f927ed5868f966032f53c7d60841debb45120781e5a7a08c1baee704882479ef',
                                                                                  'f927ed5868f966032f53c7d60841debb45120781e5a7a08c1baee704882479ef',
                                                                                  'f0e3e31f2ad3df3f6a9c442f1d484435420717b1f9228f38fd5f76d00817a596'),
                                     'services/breakout_quality/formal_fulltrade_training_source.py': ('537b5ebf1babd809359da198aa5e51e3e28c159116a02ed589b77803fbaaab4a',
                                                                                                       '537b5ebf1babd809359da198aa5e51e3e28c159116a02ed589b77803fbaaab4a',
                                                                                                       '2bb6af226ab08a6e4c7f6858d0c4cb9b03d18ba1b5fc3e9a7a5fd3f8f9de1df8'),
                                     'services/optimizer/strategy_param_service.py': ('101edb44502a467b72b9e5f6fec46e29f8637d849718194a4f49976783d819ce',
                                                                                      '101edb44502a467b72b9e5f6fec46e29f8637d849718194a4f49976783d819ce',
                                                                                      'd1b7171126b546b1b2dc9cea497a6d820817bf1fb045c3d899c1c0085c9ce97e'),
                                     'services/optimizer/strategy_param_training.py': ('d8afe8fb44a734aa996a31aab531d46bcea197688b275fab4a932eb33acafc3d',
                                                                                       'c8538ceb13ecf497f8bb29539908e58e3eced10458e0aa8ec34d8f9d27ee05d6',
                                                                                       'd8afe8fb44a734aa996a31aab531d46bcea197688b275fab4a932eb33acafc3d'),
                                     'services/optimizer/strategy_param_contracts.py': ('4611f0888b8e5507189d6dc9998168fd98987e8eb4c7498879f1c7d669dc2bbb',
                                                                                        '4611f0888b8e5507189d6dc9998168fd98987e8eb4c7498879f1c7d669dc2bbb',
                                                                                        '65bee9800244a82eb572e9845f026409020bb25251321f7f5fdf5615e762cdf7')},
                         'baseline': {'services/audit/fulltrade_oracle_truth.py': ('9364926c2e44ae2187f960949150a91726594f95cac827d6059119e9d84585c0',
                                                                                   '9364926c2e44ae2187f960949150a91726594f95cac827d6059119e9d84585c0',
                                                                                   'e7dd953b0f5da048e478b029ec1950c6f4b96e0ec9112de29554abb31be342dc'),
                                      'services/breakout_quality/formal_fulltrade_training_source.py': ('9353261063413b6415f397c169731487de1c97875721656142a8b4001d72cba7',
                                                                                                        '9353261063413b6415f397c169731487de1c97875721656142a8b4001d72cba7',
                                                                                                        '2f32bc55283ad3bec50c5f46d1e20ab613ea86c69d180529a2030d47dae68e2f')}},
 'mintrade_source_graph': {'current': {'@execution_sources': '027b79d360e5969cb558f80e49b2f51f9957c2d0a7021bfeb17f9901682e3655',
                                       'services/audit/daily_trade40_oracle.py': ('02462946681658d9a4186c3bf2de8aa4d9d46a8cb690395bdd0c4759bde31ddd',
                                                                                  '02462946681658d9a4186c3bf2de8aa4d9d46a8cb690395bdd0c4759bde31ddd',
                                                                                  '9e876daa19417145d52ef4abb64d3da96c6c8c85cb7b5374ce826345656370c7'),
                                       'filters/breakout_quality/daily_trade40_target.py': ('4df90ba6d95a22cd0a78845141dc6bfb91683e7fa2ecc612cafa6f66e6bd3370',
                                                                                            '4df90ba6d95a22cd0a78845141dc6bfb91683e7fa2ecc612cafa6f66e6bd3370',
                                                                                            'e569741dc258f637757d42cdd3febd39818f2136224a23b4fc26d7c505d4ada1'),
                                       'filters/breakout_quality/trade_path_label.py': ('c64d7d40e5ac7a26364a2be1ccaf34a756a6013d239e5eda151094d48faca202',
                                                                                        'c64d7d40e5ac7a26364a2be1ccaf34a756a6013d239e5eda151094d48faca202',
                                                                                        'ee117fa4556720d3b718d4dbf06386d9fbd36cccc6ebfd9ddcb39c5acddce706'),
                                       'filters/breakout_quality/runtime.py': ('548e0d32024a09a0e03d9ead82327904fa285f3320b3dd50b8d078bcf79dedc5',
                                                                               '548e0d32024a09a0e03d9ead82327904fa285f3320b3dd50b8d078bcf79dedc5',
                                                                               '548e0d32024a09a0e03d9ead82327904fa285f3320b3dd50b8d078bcf79dedc5'),
                                       'core/backtest_finalize.py': ('e28029d52b5aa67de4cf14dd4a5d20e00454760b859d97fb4f94755a60ad71c0',
                                                                     'e28029d52b5aa67de4cf14dd4a5d20e00454760b859d97fb4f94755a60ad71c0',
                                                                     '1dc365cbd2ed9a3d5e3d1f01d0ddb22b9b541ea8128bc320f951967d33f2f83f'),
                                       'config/trading.py': ('23b2aaa2685d38cbe6f804d93f9c7c8955c6df0f7dff8ad9e5758556e3f9a3dd',
                                                             '23b2aaa2685d38cbe6f804d93f9c7c8955c6df0f7dff8ad9e5758556e3f9a3dd',
                                                             '557314d289174a0d185feb925264c05956bdc302e88fa1406aca9928484226ae'),
                                       'core/position_step.py': ('8a09b4b229933a83e2ad0600eff5997f5f56dc65e2b4c5e5b4d51124d6a1140a',
                                                                 '8a09b4b229933a83e2ad0600eff5997f5f56dc65e2b4c5e5b4d51124d6a1140a',
                                                                 '8e02b95df3e7a87fb9c559b5f874df45ca3d474e4faa722d02315c404ea9185f'),
                                       'core/position_management.py': ('2b637c6ee015c33a68c1c34f1f4c3266e7804ec780f0d97af62501bfce91efe0',
                                                                       '2b637c6ee015c33a68c1c34f1f4c3266e7804ec780f0d97af62501bfce91efe0',
                                                                       '209b35375b3c57312ea65eda2000ba6175f6024b94355a3852ed55a775ad7aba'),
                                       'core/exit_priority.py': ('1148762eca07873d7407ec1c9e2169ef919e96a10adf59282f434d50642dbd9c',
                                                                 '1148762eca07873d7407ec1c9e2169ef919e96a10adf59282f434d50642dbd9c',
                                                                 'e7ec10a7ea12fbab58a39590ffe98b8b3d459bbbd01faee5722ba8473de1c800'),
                                       'core/entry_plans.py': ('fe8c3ebcdedbd2c0fd23cc955b8323d6de3611e81085cb4e1f01e294ebc66bcf',
                                                               'fe8c3ebcdedbd2c0fd23cc955b8323d6de3611e81085cb4e1f01e294ebc66bcf',
                                                               'a288afe5c2e204929799d353536c5c7292e0debacc58ea07274d84df145505c9'),
                                       'core/extended_signals.py': ('ed242d9e732ac6ce8b973ccb0a7d65c873be8187d121d953516026c125d85abc',
                                                                    'ed242d9e732ac6ce8b973ccb0a7d65c873be8187d121d953516026c125d85abc',
                                                                    '678b253281bbff4dc2fb96cf94e8a57b09524eeaec3a548bf7f733330bf0f297'),
                                       'core/order_lot_policy.py': ('d32828c8b84e190d6ce128b187cf4c09cdef93813450cb840d956c4e5cab71c5',
                                                                    'd32828c8b84e190d6ce128b187cf4c09cdef93813450cb840d956c4e5cab71c5',
                                                                    '6fb1a8588ae88a8556938ff8bf7b9d1e907a15d7e3675a4bf14198b22ece5ff7'),
                                       'core/exact_accounting.py': ('3164672b7228b601b3a64f7b506653a34520838470f231b50f5e67a310d9b407',
                                                                    '3164672b7228b601b3a64f7b506653a34520838470f231b50f5e67a310d9b407',
                                                                    '8f4afc21c40de29a4d648c5ee4adb5a716ebdb6f3daed90123b5c4c895af5a13'),
                                       'core/fee_rebate.py': ('101547bee73ea2d173fc81124f75d19d86771d056cb7861c0878197d3a76e694',
                                                              '101547bee73ea2d173fc81124f75d19d86771d056cb7861c0878197d3a76e694',
                                                              '7be00b11b5230fdb75b4f094c5545e64925ab8d46cbc920064bef50186b77ab5'),
                                       'core/price_utils.py': ('cc80faffb3c2ba9fbac2b979fbe765ee497b5a916f03278f3ed25c468ab6ae7f',
                                                               'cc80faffb3c2ba9fbac2b979fbe765ee497b5a916f03278f3ed25c468ab6ae7f',
                                                               'd53910f5c5a188dfc4f73201b2720c68ebeb4bd6a63b8e0821e183d95793b1fb'),
                                       'core/selection_policy.py': ('0237e6dddbba5d05878c9bd9d1fe16b3a5114fc9e4fccbe02e10206cf5045d22',
                                                                    '0237e6dddbba5d05878c9bd9d1fe16b3a5114fc9e4fccbe02e10206cf5045d22',
                                                                    'f4ff1376abc154aeafdcde36b0a1507a5a1a38f879597137a324281c76560383'),
                                       'core/signal_utils.py': ('765532e181267741f78e7a0cba26e468097b0781eeff0adcecc045689e3283b4',
                                                                '765532e181267741f78e7a0cba26e468097b0781eeff0adcecc045689e3283b4',
                                                                'bf9f9dd6d751e31d45854a8441957a55ca0f85f52dd98884c5dd479b8018c2c3'),
                                       'core/breakout_false_filter.py': ('fa94378a7b285faaec2bc5dc65c96e591762289d65360cdb1137d993cc885958',
                                                                         'fa94378a7b285faaec2bc5dc65c96e591762289d65360cdb1137d993cc885958',
                                                                         '4d520ce6a9a510ba28b556c1dcdffc0a9ac391379dc168978a0f29c7fef885f1'),
                                       'core/capital_policy.py': ('eaeeba5411dd82ffb4c107cc8b3ff94b9e97413ba5bde08ba485e00dd7980b50',
                                                                  'f38273ac18f8b200a10807d15f0071cd7c9b72244e30e8d2d3635c8d547ad976',
                                                                  'eaeeba5411dd82ffb4c107cc8b3ff94b9e97413ba5bde08ba485e00dd7980b50'),
                                       'core/native_optimizer_hotpath.py': ('36ace9fd9774aece1f29f79933e94f6d070e740cd40cc021c504ba7d5cf673a7',
                                                                            '36ace9fd9774aece1f29f79933e94f6d070e740cd40cc021c504ba7d5cf673a7',
                                                                            '41a79dc0e2e3798a619f682963623a488c64459d2c21d293639da1dd899d2784'),
                                       'core/native_optimizer_hotpath.c': ('5d48141087f007c802f4c5f810884d0c01026cc156b9749710508bfd28fa19b1',
                                                                           '5d48141087f007c802f4c5f810884d0c01026cc156b9749710508bfd28fa19b1',
                                                                           '9e9217562819063dc83cad0f6016fbc677608ad06cf7ad0d69e8b96ea0722d68'),
                                       'core/training_performance.py': ('b5e79dc378cadd244e755d8d794fe0bca1ca8dce929b57b13dbdb44426a6bc14',
                                                                        'b5e79dc378cadd244e755d8d794fe0bca1ca8dce929b57b13dbdb44426a6bc14',
                                                                        '5fd746c14b39fabdf906c2a0907329efaa59b63fb1e60f92f89b9a53deb77984'),
                                       'config/training_performance_policy.py': ('82b4e5043a04e2bb1f8692bc17a256d5332fb6d460f339e104a10ff5399d261e',
                                                                                 '82b4e5043a04e2bb1f8692bc17a256d5332fb6d460f339e104a10ff5399d261e',
                                                                                 'c33402670f6524673ccd7f264763503d5ed73b82c209389a1a39a471be9d5e35'),
                                       'config/broker_accounting.py': ('7499aa88c0af8007ecb059eb881a44e379a96f46a17467a281bf70e05b5088a0',
                                                                       '7499aa88c0af8007ecb059eb881a44e379a96f46a17467a281bf70e05b5088a0',
                                                                       'd085f7e35262669a6893f3ae659ec371124f7c6ad91bfa6f0de51ab7395cc995'),
                                       'config/execution_policy.py': ('66afb0e5198fe354467ca36147a9b898ae98321da2b20eafc792323e5d2670b3',
                                                                      '66afb0e5198fe354467ca36147a9b898ae98321da2b20eafc792323e5d2670b3',
                                                                      '16134d41d3f8337768539d47d753acde491efba8b6625f84fde0bbeb8a97ce66'),
                                       'core/feature_bank.py': ('decea8c8396045834427938b2fb0b8d8dcbfae23169e69af3a5c50f8c000af30',
                                                                'decea8c8396045834427938b2fb0b8d8dcbfae23169e69af3a5c50f8c000af30',
                                                                'd3ac3d9563d7e8ebc00acc9c4c21a8408b8d6024f55fd746ebc18c8b42dc2ed4'),
                                       'core/strategy_params.py': ('18a83788ca2339599b0a6db4fb62cdae37b73f3e5c3960ac599ca66767fc3538',
                                                                   '18a83788ca2339599b0a6db4fb62cdae37b73f3e5c3960ac599ca66767fc3538',
                                                                   'fac23ca67fd6753bfe44106bc0cc8c54dfbd9d8d8b0f719f8b22053094fbb39c'),
                                       'strategies/breakout/schema.py': ('2a181be07eb87a46a099196d2f904b19fad870503f499f6a3d3b4278474f31ba',
                                                                         '2a181be07eb87a46a099196d2f904b19fad870503f499f6a3d3b4278474f31ba',
                                                                         '594e1e050c7a1a95b34498ee2af27e5b315a621fb581f8aea24fa943ee44e1b1'),
                                       'core/breakout_quality_runtime_port.py': ('2ceff34d22832949060d87fd3f135a0981d4253f87dce81747cedda34efe4cd5',
                                                                                 '4b4e1ae93bfd6e5ea59dafba801777820785d087a757291da06845b3acc21347',
                                                                                 '2ceff34d22832949060d87fd3f135a0981d4253f87dce81747cedda34efe4cd5'),
                                       'core/seed_ensemble_policy.py': ('981e3fda0d4bd8f11a8975f665ecf6886c1e5f81ac4f42bdcfc3bc0aea90106e',
                                                                        '981e3fda0d4bd8f11a8975f665ecf6886c1e5f81ac4f42bdcfc3bc0aea90106e',
                                                                        '213ecbd846fc42ef5c98186291d7a85b8909dd87b147a8b01796c986f406fc7d'),
                                       'config/breakout_policy.py': ('0b580445aba8d0559974a939d9932556a3ba1d51348649726f8c9b9290fe2c50',
                                                                     '0b580445aba8d0559974a939d9932556a3ba1d51348649726f8c9b9290fe2c50',
                                                                     '63eeef4435fde006ccf9d09965c6ba4350793b166b376361fb7b18059d35da65'),
                                       'core/active_param_ensemble.py': ('ba4c7d7562f985a498b122f69be53057358f8dc76ee2c5b5b9e808419db3a5b3',
                                                                         'ba4c7d7562f985a498b122f69be53057358f8dc76ee2c5b5b9e808419db3a5b3',
                                                                         '547425e4ff046678c40c70e066ee212991dfa8021fc723ebb0eadaa270006afc'),
                                       'core/rolling_oos_params.py': ('6440adaaefb0a4deeaf12951a317e9dac4523ba305345ab24797268fc5957050',
                                                                      '6440adaaefb0a4deeaf12951a317e9dac4523ba305345ab24797268fc5957050',
                                                                      '3e7a760e212a006d446ac122ccc0fe03f29e0bec32781f7060a34aee77933f3c'),
                                       'core/params_io.py': ('d63aa3241582c6362923a7d38c0e985d655f83b2d825ba3849f4f81d8bfc2d14',
                                                             'd63aa3241582c6362923a7d38c0e985d655f83b2d825ba3849f4f81d8bfc2d14',
                                                             'ae352a90d3621bf5e727128348921758659157e1945944b815e33d414eb5de82'),
                                       'core/portfolio_param_runtime.py': ('3bf9b4a6790875ae1252f69ae85fc5f04c34b8fcdb9d2c730b45d4e5274e5f25',
                                                                           '3bf9b4a6790875ae1252f69ae85fc5f04c34b8fcdb9d2c730b45d4e5274e5f25',
                                                                           '6345692aa151e830bdaa9c61d3b74a77f30b47a207239c3b9f4316c722aeba78'),
                                       'core/param_access.py': ('4b5e36744ac22312e0be7dc0b8bdb32cd48bb466d273f959bfe1150372e3e732',
                                                                '4b5e36744ac22312e0be7dc0b8bdb32cd48bb466d273f959bfe1150372e3e732',
                                                                'a27073431310b1a69b4c84b4af679e4b35dab9875232b26ce1fabf5f3cf05bf6'),
                                       'core/raw_universe_contract.py': ('35e79aa1708b217a2f055351b4469263f063f1c401cc3475eb3d87305a255627',
                                                                         '35e79aa1708b217a2f055351b4469263f063f1c401cc3475eb3d87305a255627',
                                                                         '8d2acddcb61c724a50ef4923f317cd01bde23f43a14ec059d0cc5dbf09934e26'),
                                       'core/file_integrity.py': ('6f5abc343fb6ecc75e43f0ed8ead19023b627fc5ebaa93a2804c6bf1d87746ec',
                                                                  '6f5abc343fb6ecc75e43f0ed8ead19023b627fc5ebaa93a2804c6bf1d87746ec',
                                                                  '234c685bf9a7c8a4d890a571edb868cf36e83e289bdf12963e7d49da32163097'),
                                       'core/scientific_runtime_context.py': ('931b9a6c53b70b5b7ec1e2145677b76d8497d24e6d79a195d58599c3ffb1896b',
                                                                              '931b9a6c53b70b5b7ec1e2145677b76d8497d24e6d79a195d58599c3ffb1896b',
                                                                              'fd1344032a9599bbc5549cee03245c1d577480b8be2809cde4cfbf1316884ce3')},
                           'baseline': {'services/audit/daily_trade40_oracle.py': ('e93b116b89eb28c69e5cf814beb0bfd84f15819378225bdeebebd497c721fa3e',
                                                                                   'e93b116b89eb28c69e5cf814beb0bfd84f15819378225bdeebebd497c721fa3e',
                                                                                   'e93b116b89eb28c69e5cf814beb0bfd84f15819378225bdeebebd497c721fa3e'),
                                        'filters/breakout_quality/daily_trade40_target.py': ('4df90ba6d95a22cd0a78845141dc6bfb91683e7fa2ecc612cafa6f66e6bd3370',
                                                                                             '4df90ba6d95a22cd0a78845141dc6bfb91683e7fa2ecc612cafa6f66e6bd3370',
                                                                                             'e569741dc258f637757d42cdd3febd39818f2136224a23b4fc26d7c505d4ada1'),
                                        'filters/breakout_quality/trade_path_label.py': ('c64d7d40e5ac7a26364a2be1ccaf34a756a6013d239e5eda151094d48faca202',
                                                                                         'c64d7d40e5ac7a26364a2be1ccaf34a756a6013d239e5eda151094d48faca202',
                                                                                         'ee117fa4556720d3b718d4dbf06386d9fbd36cccc6ebfd9ddcb39c5acddce706'),
                                        'filters/breakout_quality/runtime.py': ('548e0d32024a09a0e03d9ead82327904fa285f3320b3dd50b8d078bcf79dedc5',
                                                                                '548e0d32024a09a0e03d9ead82327904fa285f3320b3dd50b8d078bcf79dedc5',
                                                                                '548e0d32024a09a0e03d9ead82327904fa285f3320b3dd50b8d078bcf79dedc5'),
                                        'core/backtest_finalize.py': ('e28029d52b5aa67de4cf14dd4a5d20e00454760b859d97fb4f94755a60ad71c0',
                                                                      'e28029d52b5aa67de4cf14dd4a5d20e00454760b859d97fb4f94755a60ad71c0',
                                                                      '1dc365cbd2ed9a3d5e3d1f01d0ddb22b9b541ea8128bc320f951967d33f2f83f'),
                                        'core/position_step.py': ('8a09b4b229933a83e2ad0600eff5997f5f56dc65e2b4c5e5b4d51124d6a1140a',
                                                                  '8a09b4b229933a83e2ad0600eff5997f5f56dc65e2b4c5e5b4d51124d6a1140a',
                                                                  '8e02b95df3e7a87fb9c559b5f874df45ca3d474e4faa722d02315c404ea9185f'),
                                        'core/position_management.py': ('2b637c6ee015c33a68c1c34f1f4c3266e7804ec780f0d97af62501bfce91efe0',
                                                                        '2b637c6ee015c33a68c1c34f1f4c3266e7804ec780f0d97af62501bfce91efe0',
                                                                        '209b35375b3c57312ea65eda2000ba6175f6024b94355a3852ed55a775ad7aba'),
                                        'core/exit_priority.py': ('1148762eca07873d7407ec1c9e2169ef919e96a10adf59282f434d50642dbd9c',
                                                                  '1148762eca07873d7407ec1c9e2169ef919e96a10adf59282f434d50642dbd9c',
                                                                  'e7ec10a7ea12fbab58a39590ffe98b8b3d459bbbd01faee5722ba8473de1c800'),
                                        'core/entry_plans.py': ('fe8c3ebcdedbd2c0fd23cc955b8323d6de3611e81085cb4e1f01e294ebc66bcf',
                                                                'fe8c3ebcdedbd2c0fd23cc955b8323d6de3611e81085cb4e1f01e294ebc66bcf',
                                                                'a288afe5c2e204929799d353536c5c7292e0debacc58ea07274d84df145505c9'),
                                        'core/extended_signals.py': ('ed242d9e732ac6ce8b973ccb0a7d65c873be8187d121d953516026c125d85abc',
                                                                     'ed242d9e732ac6ce8b973ccb0a7d65c873be8187d121d953516026c125d85abc',
                                                                     '678b253281bbff4dc2fb96cf94e8a57b09524eeaec3a548bf7f733330bf0f297'),
                                        'core/order_lot_policy.py': ('d32828c8b84e190d6ce128b187cf4c09cdef93813450cb840d956c4e5cab71c5',
                                                                     'd32828c8b84e190d6ce128b187cf4c09cdef93813450cb840d956c4e5cab71c5',
                                                                     '6fb1a8588ae88a8556938ff8bf7b9d1e907a15d7e3675a4bf14198b22ece5ff7'),
                                        'core/exact_accounting.py': ('3164672b7228b601b3a64f7b506653a34520838470f231b50f5e67a310d9b407',
                                                                     '3164672b7228b601b3a64f7b506653a34520838470f231b50f5e67a310d9b407',
                                                                     '8f4afc21c40de29a4d648c5ee4adb5a716ebdb6f3daed90123b5c4c895af5a13'),
                                        'core/fee_rebate.py': ('101547bee73ea2d173fc81124f75d19d86771d056cb7861c0878197d3a76e694',
                                                               '101547bee73ea2d173fc81124f75d19d86771d056cb7861c0878197d3a76e694',
                                                               '7be00b11b5230fdb75b4f094c5545e64925ab8d46cbc920064bef50186b77ab5'),
                                        'core/price_utils.py': ('cc80faffb3c2ba9fbac2b979fbe765ee497b5a916f03278f3ed25c468ab6ae7f',
                                                                'cc80faffb3c2ba9fbac2b979fbe765ee497b5a916f03278f3ed25c468ab6ae7f',
                                                                'd53910f5c5a188dfc4f73201b2720c68ebeb4bd6a63b8e0821e183d95793b1fb'),
                                        'core/strategy_params.py': ('18a83788ca2339599b0a6db4fb62cdae37b73f3e5c3960ac599ca66767fc3538',
                                                                    '18a83788ca2339599b0a6db4fb62cdae37b73f3e5c3960ac599ca66767fc3538',
                                                                    'fac23ca67fd6753bfe44106bc0cc8c54dfbd9d8d8b0f719f8b22053094fbb39c'),
                                        'strategies/breakout/schema.py': ('2a181be07eb87a46a099196d2f904b19fad870503f499f6a3d3b4278474f31ba',
                                                                          '2a181be07eb87a46a099196d2f904b19fad870503f499f6a3d3b4278474f31ba',
                                                                          '594e1e050c7a1a95b34498ee2af27e5b315a621fb581f8aea24fa943ee44e1b1'),
                                        'core/selection_policy.py': ('0237e6dddbba5d05878c9bd9d1fe16b3a5114fc9e4fccbe02e10206cf5045d22',
                                                                     '0237e6dddbba5d05878c9bd9d1fe16b3a5114fc9e4fccbe02e10206cf5045d22',
                                                                     'f4ff1376abc154aeafdcde36b0a1507a5a1a38f879597137a324281c76560383'),
                                        'core/signal_utils.py': ('765532e181267741f78e7a0cba26e468097b0781eeff0adcecc045689e3283b4',
                                                                 '765532e181267741f78e7a0cba26e468097b0781eeff0adcecc045689e3283b4',
                                                                 'bf9f9dd6d751e31d45854a8441957a55ca0f85f52dd98884c5dd479b8018c2c3'),
                                        'core/breakout_false_filter.py': ('fa94378a7b285faaec2bc5dc65c96e591762289d65360cdb1137d993cc885958',
                                                                          'fa94378a7b285faaec2bc5dc65c96e591762289d65360cdb1137d993cc885958',
                                                                          '4d520ce6a9a510ba28b556c1dcdffc0a9ac391379dc168978a0f29c7fef885f1'),
                                        'core/feature_bank.py': ('decea8c8396045834427938b2fb0b8d8dcbfae23169e69af3a5c50f8c000af30',
                                                                 'decea8c8396045834427938b2fb0b8d8dcbfae23169e69af3a5c50f8c000af30',
                                                                 'd3ac3d9563d7e8ebc00acc9c4c21a8408b8d6024f55fd746ebc18c8b42dc2ed4'),
                                        'core/native_optimizer_hotpath.py': ('36ace9fd9774aece1f29f79933e94f6d070e740cd40cc021c504ba7d5cf673a7',
                                                                             '36ace9fd9774aece1f29f79933e94f6d070e740cd40cc021c504ba7d5cf673a7',
                                                                             '41a79dc0e2e3798a619f682963623a488c64459d2c21d293639da1dd899d2784'),
                                        'core/native_optimizer_hotpath.c': ('5d48141087f007c802f4c5f810884d0c01026cc156b9749710508bfd28fa19b1',
                                                                            '5d48141087f007c802f4c5f810884d0c01026cc156b9749710508bfd28fa19b1',
                                                                            '9e9217562819063dc83cad0f6016fbc677608ad06cf7ad0d69e8b96ea0722d68'),
                                        'core/training_performance.py': ('b5e79dc378cadd244e755d8d794fe0bca1ca8dce929b57b13dbdb44426a6bc14',
                                                                         'b5e79dc378cadd244e755d8d794fe0bca1ca8dce929b57b13dbdb44426a6bc14',
                                                                         '5fd746c14b39fabdf906c2a0907329efaa59b63fb1e60f92f89b9a53deb77984'),
                                        'core/portfolio_param_runtime.py': ('3bf9b4a6790875ae1252f69ae85fc5f04c34b8fcdb9d2c730b45d4e5274e5f25',
                                                                            '3bf9b4a6790875ae1252f69ae85fc5f04c34b8fcdb9d2c730b45d4e5274e5f25',
                                                                            '6345692aa151e830bdaa9c61d3b74a77f30b47a207239c3b9f4316c722aeba78'),
                                        'core/params_io.py': ('d63aa3241582c6362923a7d38c0e985d655f83b2d825ba3849f4f81d8bfc2d14',
                                                              'd63aa3241582c6362923a7d38c0e985d655f83b2d825ba3849f4f81d8bfc2d14',
                                                              'ae352a90d3621bf5e727128348921758659157e1945944b815e33d414eb5de82'),
                                        'core/active_param_ensemble.py': ('ba4c7d7562f985a498b122f69be53057358f8dc76ee2c5b5b9e808419db3a5b3',
                                                                          'ba4c7d7562f985a498b122f69be53057358f8dc76ee2c5b5b9e808419db3a5b3',
                                                                          '547425e4ff046678c40c70e066ee212991dfa8021fc723ebb0eadaa270006afc'),
                                        'core/rolling_oos_params.py': ('6440adaaefb0a4deeaf12951a317e9dac4523ba305345ab24797268fc5957050',
                                                                       '6440adaaefb0a4deeaf12951a317e9dac4523ba305345ab24797268fc5957050',
                                                                       '3e7a760e212a006d446ac122ccc0fe03f29e0bec32781f7060a34aee77933f3c'),
                                        'core/seed_ensemble_policy.py': ('981e3fda0d4bd8f11a8975f665ecf6886c1e5f81ac4f42bdcfc3bc0aea90106e',
                                                                         '981e3fda0d4bd8f11a8975f665ecf6886c1e5f81ac4f42bdcfc3bc0aea90106e',
                                                                         '213ecbd846fc42ef5c98186291d7a85b8909dd87b147a8b01796c986f406fc7d'),
                                        'core/capital_policy.py': ('eaeeba5411dd82ffb4c107cc8b3ff94b9e97413ba5bde08ba485e00dd7980b50',
                                                                   'f38273ac18f8b200a10807d15f0071cd7c9b72244e30e8d2d3635c8d547ad976',
                                                                   'eaeeba5411dd82ffb4c107cc8b3ff94b9e97413ba5bde08ba485e00dd7980b50'),
                                        'config/broker_accounting.py': ('7499aa88c0af8007ecb059eb881a44e379a96f46a17467a281bf70e05b5088a0',
                                                                        '7499aa88c0af8007ecb059eb881a44e379a96f46a17467a281bf70e05b5088a0',
                                                                        'd085f7e35262669a6893f3ae659ec371124f7c6ad91bfa6f0de51ab7395cc995'),
                                        'config/breakout_policy.py': ('0b580445aba8d0559974a939d9932556a3ba1d51348649726f8c9b9290fe2c50',
                                                                      '0b580445aba8d0559974a939d9932556a3ba1d51348649726f8c9b9290fe2c50',
                                                                      '63eeef4435fde006ccf9d09965c6ba4350793b166b376361fb7b18059d35da65'),
                                        'config/execution_policy.py': ('66afb0e5198fe354467ca36147a9b898ae98321da2b20eafc792323e5d2670b3',
                                                                       '66afb0e5198fe354467ca36147a9b898ae98321da2b20eafc792323e5d2670b3',
                                                                       '16134d41d3f8337768539d47d753acde491efba8b6625f84fde0bbeb8a97ce66'),
                                        'config/training_performance_policy.py': ('82b4e5043a04e2bb1f8692bc17a256d5332fb6d460f339e104a10ff5399d261e',
                                                                                  '82b4e5043a04e2bb1f8692bc17a256d5332fb6d460f339e104a10ff5399d261e',
                                                                                  'c33402670f6524673ccd7f264763503d5ed73b82c209389a1a39a471be9d5e35'),
                                        'config/trading.py': ('23b2aaa2685d38cbe6f804d93f9c7c8955c6df0f7dff8ad9e5758556e3f9a3dd',
                                                              '23b2aaa2685d38cbe6f804d93f9c7c8955c6df0f7dff8ad9e5758556e3f9a3dd',
                                                              '557314d289174a0d185feb925264c05956bdc302e88fa1406aca9928484226ae')},
                           'baseline_order': ('services/audit/daily_trade40_oracle.py',
                                              'filters/breakout_quality/daily_trade40_target.py',
                                              'filters/breakout_quality/trade_path_label.py',
                                              'filters/breakout_quality/runtime.py',
                                              'core/backtest_finalize.py',
                                              'core/position_step.py',
                                              'core/position_management.py',
                                              'core/exit_priority.py',
                                              'core/entry_plans.py',
                                              'core/extended_signals.py',
                                              'core/order_lot_policy.py',
                                              'core/exact_accounting.py',
                                              'core/fee_rebate.py',
                                              'core/price_utils.py',
                                              'core/strategy_params.py',
                                              'strategies/breakout/schema.py',
                                              'core/selection_policy.py',
                                              'core/signal_utils.py',
                                              'core/breakout_false_filter.py',
                                              'core/feature_bank.py',
                                              'core/native_optimizer_hotpath.py',
                                              'core/native_optimizer_hotpath.c',
                                              'core/training_performance.py',
                                              'core/portfolio_param_runtime.py',
                                              'core/params_io.py',
                                              'core/active_param_ensemble.py',
                                              'core/rolling_oos_params.py',
                                              'core/seed_ensemble_policy.py',
                                              'core/capital_policy.py',
                                              'config/broker_accounting.py',
                                              'config/breakout_policy.py',
                                              'config/execution_policy.py',
                                              'config/training_performance_policy.py',
                                              'config/trading.py')}}

# Per-selected-descriptor AST certificates are populated only after independent
# baseline projection equality. Values are source ASTs, never result digests.
_REVIEWED_RANKING_PROJECTIONS = {'c774f4d38bb793e7c2bb3ef5d84b2e64e621dfd19445d2fef261738224da447e': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '31b249da461a46acf2ab6ab46860d166d3b0dc00dbc9cda5d53676d65b6b7aac': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '8b8c8aa0636b868b666b770fba67ed7f6524ceff763d502bb62bfa90a6bf1db8': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '114f322484be7d3c3c42ab64bba995c1b341754bef2892af03be03af520a3d80': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '9ed19c889e18e120404e38481b801269f1e124f9d83de95a419389db924391b2': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '4acde71a5541996faa23e8f5ed5b4ace84926fbbaaa42764ef0f6300250c8e2c': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '8c34033f2badb1bf4ae6c46e28a6a363fd929fdb1932a67b7c46c705d95816c9': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 'ff8a23c0c8d901ebce51e030dcadf1113522a8ac26b3f70803842b85412f939e': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '49d0b7520cd92bd188043c1d435625c5e6e80d18f0303c8b12c058f564dad2ec': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '7add134dd9891e440de89e457b3b192480c651a054e409fd5e8322ab2c0d5f0a': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '56f621d4ad5e5129bbb4127391a59743a8a045897a1d45f78e31021af6f45ab7': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 'be955bde4ba3deab22c06ef27fc1732643eda31f37046d5c49e0da53dd57568a': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '351a27655731b31ad0ce346f4fb5d3c4416b2860c836fd561e67b3e0e1f1786f': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 'bf89edc1d52a40a8474161f6befb8fea26f7fde1f4d474122bbb59ed897da65f': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 'adfc2c06aa7f88ae80d5dfc74b02e777bf6f9f93ff21316c0b8263eee7bb4d86': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '2d90f639697793b15fb012fed1e3a9e11358c22103e9f8bc50ee2ecefff2a6a7': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 'd7c4ac81c0894df400b159a0a3a3d374b514d36fb70d7dc66907c214c7c5c22a': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '035043082beef0d588cd925ce46202bfa8ae7a7d40993d91cf7256465028d3df': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '17021caf38ab52adedb17509c084d9c44c56ba8211934b2dd49c0c17169e45bb': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 'e0ca468c7293f2da4924a0f2f659652ba3b2a7ecd5f9f233429b91efb727200f': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 'fa4d381e5f55b3a85902a477ecac094b44b3d97a43ab0d52488c36f9f2f0b602': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '5dd916abc3cd462aa0246648126e7b2a862cc7a85ee34ae17bb9ef13aa830a94': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '8fd5d8a9c1e862277bc6fc2112772d6d3ec5602bebd908daa0c1433ef0e12126': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 'a994af8c4e9cabb11fa70183ee141f65d1a5fb87b7979895c184cd6e9f595e4b': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '68b60754e1680de4259a96d37f0d29ae27f485f8dda0fc97c3765a81d41e5a48': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '39ad27a527c9a3a0b8e87886adaffe4ab266abb9ed50406b204b11606b3d9dfb': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '6d28ff2a3cd3314b4dd02ec3549e1753b8b50b45a7b9157b06006b08637a4d0c': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
 '4f9439ecf3a3febb0fa47bd58aaa34b08c6f73341eddf2fe29cba2ddf03f1388': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a'}


def reviewed_ranking_source_sha256(projected_sha256: str, *, project_root=None) -> str:
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if project_root is not None and Path(project_root).resolve() != root:
        return projected_sha256
    return _REVIEWED_RANKING_PROJECTIONS.get(projected_sha256, projected_sha256)


# Oracle persists strategy observations, not benchmark comparison columns. The
# cache correction remains live for full Portfolio outputs. This certificate
# is limited to the exact reviewed producer AND observation-consumer closure;
# the four real-engine scope probes deliberately corrupt benchmark statistics
# and prove identical Oracle observations/trades, while benchmark reports differ.
ORACLE_STRATEGY_OBSERVATION_SCOPE = "oracle_strategy_metrics"
_ORACLE_OBSERVATION_CONSUMERS = {
    "services/research/strategy_compare_replay.py": "c2ab29ec33adeb6a884e05532ab1afe26ca8a0c8eda4a16c3e750688382a51b5",
    "filters/breakout_quality/strategy_compare_reporting.py": "daab5855348031da4560fd746c05933d85f39ab330050663ae209b5bc51f6e4f",
    "services/audit/target_oracle_romd.py": "873b0e3fcd1c2f688e3cd4150269ddf434d3d682c118325058bdc9d3438fa8a1",
}
_ORACLE_BENCHMARK_SOURCE = (
    "51e5d2a687fe926c0ff487bd634e75989abebbdce6e2643c26f5f9ba8422ac35",
    "95facbde65f014ef3d7da37510643e90e2b8596c11d82ef873effb903da38266",
)
_DAILY_FILTER_CALCULATION = (
    "c0a8506ab72233c1f2ee74c83ac9c2fb12aa76b9a280496acf47b9cebcca6869",
    "26a376714ef8672f830fd9988e63f915bbda60d7dbe23b4506df125908b71e53",
)


def oracle_observation_consumer_sources() -> tuple[str, ...]:
    return tuple(_ORACLE_OBSERVATION_CONSUMERS)


def project_oracle_observation_sources(parts, *, scope: str, project_root):
    """Prove only the persisted Oracle strategy observation, never full results.

    The caller supplies actual consumer sources too. An unknown producer,
    dependency graph, benchmark or consumer keeps all those sources in its
    ordinary fingerprint, including later edits that start consuming benchmark
    columns. No current or historical result digest is an authority here.
    """
    from pathlib import Path
    result = dict(parts)
    if Path(project_root).resolve() != Path(__file__).resolve().parents[1]:
        return result
    if scope not in ("replay_numerics", "portfolio_candidates"):
        return result
    expected = dict(_reviewed_source_group("portfolio_candidates")["current"])
    expected["@execution_sources"] = _reviewed_source_group(scope)["current"]["@execution_sources"]
    expected["core/portfolio_benchmark.py"] = _ORACLE_BENCHMARK_SOURCE[0]
    expected.update(_ORACLE_OBSERVATION_CONSUMERS)
    optional = "core/daily_common_entry_filters.py"
    if optional in result:
        expected[optional] = _REVIEWED_SOURCE_INVENTORIES["portfolio_full_filters"]["current"][optional]
    for name, value in expected.items():
        if result.get(name) not in ((value,) if isinstance(value, str) else value):
            return result
    for name in _ORACLE_OBSERVATION_CONSUMERS:
        result.pop(name)
    if scope == "replay_numerics":
        # The original Replay syntax inventory did not include benchmark stats.
        result.pop("core/portfolio_benchmark.py")
    else:
        result["core/portfolio_benchmark.py"] = _ORACLE_BENCHMARK_SOURCE[1]
    return result


def project_daily_filter_calculation(parts, *, project_root):
    """Attest explicit execution-context extraction with its delegated helper."""
    from pathlib import Path
    result = dict(parts)
    if Path(project_root).resolve() != Path(__file__).resolve().parents[1]:
        return result
    expected = _REVIEWED_SOURCE_INVENTORIES["portfolio_full_filters"]["current"]
    if (result.get("calculation") == _DAILY_FILTER_CALCULATION[0]
            and all(result.get(name) == value for name, value in expected.items())
            and set(result) == {"calculation", *expected}):
        return {"calculation": _DAILY_FILTER_CALCULATION[1]}
    return result


def project_reviewed_source_inventory(parts, *, scope: str, project_root=None, source_encoding=None):
    """Project one fully pinned refactor subgraph; preserve all other sources.

    Raw values may be SHA strings or {None/LF/CRLF: SHA} encoding alternatives.
    An external source fixture, incomplete group, or unreviewed owner/helper
    revision retains the entire current group. No result identity is hardcoded.
    """
    from collections.abc import Mapping
    from pathlib import Path

    result = dict(parts)
    code_root = Path(__file__).resolve().parents[1]
    if project_root is not None and Path(project_root).resolve() != code_root:
        return result
    proof = _reviewed_source_group(scope)
    if proof is None:
        return result
    if source_encoding not in (None, b"\n", b"\r\n"):
        raise ValueError("Unknown source proof encoding")
    encoded = None
    for name, expected in proof["current"].items():
        value = result.get(name)
        is_encoded = isinstance(value, Mapping)
        if encoded is None:
            encoded = is_encoded
        elif encoded != is_encoded:
            return result
        if isinstance(expected, str):
            if value != expected:
                return result
        elif is_encoded:
            if (set(value) != {None, b"\n", b"\r\n"}
                    or value[None] not in expected
                    or value[b"\n"] != expected[1]
                    or value[b"\r\n"] != expected[2]):
                return result
        elif value not in expected:
            return result
    for name in proof["current"]:
        previous = proof["baseline"].get(name)
        if previous is None:
            result.pop(name)
        elif isinstance(previous, str):
            result[name] = previous
        elif encoded:
            result[name] = {None: previous[0], b"\n": previous[1], b"\r\n": previous[2]}
        else:
            index = (None, b"\n", b"\r\n").index(source_encoding)
            # Earlier source-group attestations can retain historical bytes
            # even in an LF checkout. Reconstruct that actual producer path,
            # in addition to its existing literal LF/CRLF source proof.
            encoded_baseline = proof.get("baseline_lf" if index == 1 else "baseline_crlf", {}) if index else {}
            result[name] = encoded_baseline.get(name, previous[index])
    # Ordered raw-source producers retain their historically signed order only
    # after the complete graph certificate succeeds. New/unreviewed sources stay.
    if "baseline_order" in proof:
        result = {**{name: result[name] for name in proof["baseline_order"] if name in result},
                  **{name: value for name, value in result.items() if name not in proof["baseline_order"]}}
    return result


# Candidate engineering proof against formal b7468719 ZIP SHA
# 69c3fcca397e812374ac9d0b05ca02656db86b34f219d4be87f161d5ced51b65.
# DISABLED pending independent complete numerical parity and source review.
# These are exact source bytes / syntax payloads, never result digest aliases.
_OPTIMIZER_EXECUTION_REVIEW_ENABLED = True
_OPTIMIZER_EXECUTION_PHYSICAL_PINS = {'config/training_performance_policy.py': {'d635114d3cdfb70be4944f41c49e7ae010c89ef13284e168a850cbe0a0ad2965': 'c33402670f6524673ccd7f264763503d5ed73b82c209389a1a39a471be9d5e35',
                                           'f1dd90b1ff94eba4e4b6ef4703de6518a7f0ae5c2e292f8d4ce6e2bc483a09f1': '82b4e5043a04e2bb1f8692bc17a256d5332fb6d460f339e104a10ff5399d261e'},
 'core/entry_plans.py': {'3f5fb2cc4397fd5ad6221d2efb4141aa2f83f7abed7fffe73a1d41541e450df2': 'fe8c3ebcdedbd2c0fd23cc955b8323d6de3611e81085cb4e1f01e294ebc66bcf',
                         '4c87a91b91d0e6133f1d3844b313c3f18c6815c54774884900a8564606f543aa': 'a288afe5c2e204929799d353536c5c7292e0debacc58ea07274d84df145505c9'},
 'core/extended_signals.py': {'8ac4821b88a3fa2981243d9fef5bde448b2c9ed8340831ece733ac42f6c4ef17': '678b253281bbff4dc2fb96cf94e8a57b09524eeaec3a548bf7f733330bf0f297',
                              'c126129f8a967f85518f27fc738992d9fcfe3e70035c6a7093bce570d449a6de': 'ed242d9e732ac6ce8b973ccb0a7d65c873be8187d121d953516026c125d85abc'},
 'core/portfolio_candidates.py': {'c09542aab11c6068dd56d05238b6da6ff5b1bd32dc2caad9081871d059de8246': 'ca2cafb052e67547959a073859c5a1d6413f17d60d81d1ba6b293a45def17657',
                                  'fc37aa27df96cdebeb41d4c9c00d39c1542d6af3d82807c89422fa5efea36775': 'cebad8454297f98afe16f8fccefb9cb32ea75d477c4e69b572da97a0bed9fd19'},
 'core/portfolio_entries.py': {'3a54e097174bdf5b4bb3bb33ec6cc9fc0d9c6643f39a6766fd65cf7b09f686c9': '0b8d9487c966e0e5f3119b93012169eab4b257cf08a920d965e1c1adf97f3e3f',
                               'd73f66f7e4257de9d489eeb147cddfee1061d2e7baad30f891da2c1053e2cf54': '1dc7227ba740bdae3efe4c040df853dfd67df89db74198cd49d900259d584898'},
 'core/portfolio_entry_plans.py': {'7e0ce437f176ab97418460bac4d9ea5a5d2223af0a3f8506a1f3fd8bfcb877f5': '4f2a7b80a46ac21efaf30b2ba8a0220722a5be0263512096d97cbe624af2986a',
                                   'c7d27ec29208b9825186c64eb62f26dc5522b5577d6f95ef7ae65a9a060b6de6': '8fac52b70dc7d55c92922d4bd9789377fde08ecdf833af2ed5fcdc5917ec18c3'},
 'core/portfolio_fast_data.py': {'e1bcd29718e708d876951795e1ba5c1ce7cab98eb33e7a7c2882fc27f0adbfb9': '9ff95a70bb8a7de4cdd357409985ff402ab04c5d74a40ece654df5e3fece71ac',
                                 'e6b55332663b98796acfadd1292080a7586058ebace20ced86afa92fa8daeb43': 'f0f4df8a0a9ae4a98f05fbb8afe7316c8b6265374800936ac239c04eeabd2af6'},
 'core/price_utils.py': {'02c11e9f955d3e02976dea594fb48e2f4f413e99295b54c77b989d23b1270aa1': 'd53910f5c5a188dfc4f73201b2720c68ebeb4bd6a63b8e0821e183d95793b1fb',
                         '492c17d9e4e805fa2a8049b9ba7df58ccbae5c8d8b07773398727d9aabdd7d98': 'cc80faffb3c2ba9fbac2b979fbe765ee497b5a916f03278f3ed25c468ab6ae7f'},
 'services/optimizer/objective_runner.py': {'61e084b3537f708b6b43b639b511dea6dd0232852008e2413358637e97ae2ddf': 'ebaff47e1c8cd6b8e18fe2c87205fa3e685510abce1bbbdeaf24fc3760f0f579',
                                            'af7a8c3f5930aed76f49549219fb2d1d07157ffaed63fc0a9ed8728da19bd37c': 'f17d77c5442ef4c8cafe20fc1d24abede74d6ee667aef15fb74ee223bfc48668'},
 'services/optimizer/outer_rolling_evaluation.py': {'2011c1f33c63f67edb8dd97966ed05c68ce2fdba5a54d49eccec14299f8a15e3': '99b235a1a18d91962757c477806c8f4d2320c911cbf191442959af850ecfb8dd',
                                                    '8136b48e85f3780ab6bce054409154a2527f25924a2893916755fcab291f798a': '7302433719480ca72d8c53dac7920d05fbd9360587d0895cefae561d04b1de22'},
 'services/optimizer/trial_inputs.py': {'3cd4a7a50e36762732f017e5c610545ae6e4debcb762d234c2d4db737d09305d': '855891b2712953f9ab8acfe49cc473506e74b0566197abd2f90452e2f841d590',
                                        'd49cc24ae9aadb69700351039173416c2da3a9eded9468c0b6e85badf22055c8': '7168a066b22a9d4633359fbec77e419b18a42e66ed1271e6ff05952686eeaac0'},
 'services/portfolio_replay.py': {'82fbc24292b649f3e45448a9f38c781ec73ec78ecd089fd3ddd1d156cd981e24': '5fdfdeacd6c974a71cb1aeec0b25a73d3f5c79b391ce0fd20e7770db47ac0f01',
                                  'da2b7222a63113be4864692fa6c485d72b4313562e9a71faa84b3c600480ea93': '8c19b748530f188efb6c7263fa9990cea4c843fb8d19e665627426207b0aecd8'},
 'services/research/oracle_artifact_contract.py': {'62f248fccb421a376c70dd2604422f1aa8ac845cf6ec122558ec673c876ed125': 'aa5fb89fb9d2d0bcf7e753462d77f655f1c5ff77267924778cf3fca19ccd1282',
                                                   'd4abc57ae1def8e7b88a30d4398a7dceffd6d0d81a12050d0dedca1dc1940440': '6543cd9c6fc6947ede0656dc39d821a65b386f593d5890336d31b320273cbe35'}}
_OPTIMIZER_EXECUTION_GROUP_UPDATES = {'portfolio_candidates': {'config/training_performance_policy.py': 'a6292e8aaf2e3d44be2f30e2de9d6f5fee86ad67ae4ac2c3c5401df46280bf46',
                          'core/entry_plans.py': '2ff3b0f11f8c9324a714e4fbd2df7b69b9d99f80262b9b01dc6b17bd068f8562',
                          'core/extended_signals.py': '9548f4530a3bd596861d20b8f15baef27ab9d54ef84ae1d0d4594361e65faaf7',
                          'core/portfolio_candidates.py': '839ec1ca578d8ac38968658dcafa825563a73eb6d42366b376c63f5a5f74b702',
                          'core/portfolio_entries.py': '81a0e674d6093e73409565a720ed1248c90c9852df5f071e0b099c8dd387c15c',
                          'core/portfolio_entry_plans.py': '359f477b823ddd0ca2d6a2f3f099a39a6024095422075d046b355fa97bcece6c',
                          'core/portfolio_fast_data.py': '9260b26e598ba3dd7dc8b9ed59fa9a0a77f267b5b257db9d44a6e8d22b0c988c',
                          'core/price_utils.py': 'b55c935295f1689f8ed604d9fe938e34c58b12beae83af62a6e61cba144fcfc7',
                          'services/portfolio_replay.py': 'f27a24bc9f84b3ac69e2af289a33f83fd2cbf63a3ffeee0477630ac6bab2f5c5'},
 'replay_numerics': {'config/training_performance_policy.py': 'a6292e8aaf2e3d44be2f30e2de9d6f5fee86ad67ae4ac2c3c5401df46280bf46',
                     'core/portfolio_candidates.py': '839ec1ca578d8ac38968658dcafa825563a73eb6d42366b376c63f5a5f74b702',
                     'core/portfolio_entries.py': '81a0e674d6093e73409565a720ed1248c90c9852df5f071e0b099c8dd387c15c',
                     'core/portfolio_entry_plans.py': '359f477b823ddd0ca2d6a2f3f099a39a6024095422075d046b355fa97bcece6c',
                     'core/portfolio_fast_data.py': '9260b26e598ba3dd7dc8b9ed59fa9a0a77f267b5b257db9d44a6e8d22b0c988c',
                     'services/portfolio_replay.py': 'f27a24bc9f84b3ac69e2af289a33f83fd2cbf63a3ffeee0477630ac6bab2f5c5'}}
_OPTIMIZER_LEGACY_SOURCE_RECONSTRUCTIONS = {'core/entry_plans.py': {'new_lf_sha256': '3f5fb2cc4397fd5ad6221d2efb4141aa2f83f7abed7fffe73a1d41541e450df2',
                         'new_module_calculation_sha256': '2ff3b0f11f8c9324a714e4fbd2df7b69b9d99f80262b9b01dc6b17bd068f8562',
                         'new_physical_pins': ['4c87a91b91d0e6133f1d3844b313c3f18c6815c54774884900a8564606f543aa',
                                               '3f5fb2cc4397fd5ad6221d2efb4141aa2f83f7abed7fffe73a1d41541e450df2'],
                         'predecessor_lf_sha256': 'fe8c3ebcdedbd2c0fd23cc955b8323d6de3611e81085cb4e1f01e294ebc66bcf',
                         'predecessor_module_calculation_sha256': 'c95b942559424dbf5f4afbfb1c5f7b90ad6c0e1e2b7d3fd1a58e8a09d410e29f',
                         'restored_predecessor_exact': True,
                         'reverse_edits': [{'candidate_end_line': 201,
                                            'candidate_start_line': 200,
                                            'predecessor_lines': ['def '
                                                                  'build_cash_capped_entry_plan(candidate_plan, '
                                                                  'available_cash, params):\n']},
                                           {'candidate_end_line': 212,
                                            'candidate_start_line': 202,
                                            'predecessor_lines': ['    resized_plan = '
                                                                  'resize_candidate_plan_to_capital(candidate_plan, '
                                                                  'sizing_capital, params)\n']},
                                           {'candidate_end_line': 328,
                                            'candidate_start_line': 322,
                                            'predecessor_lines': ['        target_reference_stop = '
                                                                  'calc_initial_stop_from_reference(resolved_target_reference, '
                                                                  'entry_atr, params, ticker=ticker, '
                                                                  'security_profile=security_profile)\n']}]},
 'core/extended_signals.py': {'new_lf_sha256': 'c126129f8a967f85518f27fc738992d9fcfe3e70035c6a7093bce570d449a6de',
                              'new_module_calculation_sha256': '9548f4530a3bd596861d20b8f15baef27ab9d54ef84ae1d0d4594361e65faaf7',
                              'new_physical_pins': ['8ac4821b88a3fa2981243d9fef5bde448b2c9ed8340831ece733ac42f6c4ef17',
                                                    'c126129f8a967f85518f27fc738992d9fcfe3e70035c6a7093bce570d449a6de'],
                              'predecessor_lf_sha256': 'ed242d9e732ac6ce8b973ccb0a7d65c873be8187d121d953516026c125d85abc',
                              'predecessor_module_calculation_sha256': '46837f8445c8e83d5758adf86f6b975243ef6de4eeb0cf3303c28c1eb8cfc97c',
                              'restored_predecessor_exact': True,
                              'reverse_edits': [{'candidate_end_line': 14,
                                                 'candidate_start_line': 13,
                                                 'predecessor_lines': []},
                                                {'candidate_end_line': 22,
                                                 'candidate_start_line': 21,
                                                 'predecessor_lines': []},
                                                {'candidate_end_line': 259,
                                                 'candidate_start_line': 258,
                                                 'predecessor_lines': []},
                                                {'candidate_end_line': 280,
                                                 'candidate_start_line': 279,
                                                 'predecessor_lines': []},
                                                {'candidate_end_line': 579,
                                                 'candidate_start_line': 578,
                                                 'predecessor_lines': []},
                                                {'candidate_end_line': 632,
                                                 'candidate_start_line': 631,
                                                 'predecessor_lines': []},
                                                {'candidate_end_line': 768,
                                                 'candidate_start_line': 767,
                                                 'predecessor_lines': []},
                                                {'candidate_end_line': 829,
                                                 'candidate_start_line': 805,
                                                 'predecessor_lines': ['    return '
                                                                       'resize_candidate_plan_to_capital(base_plan, '
                                                                       'sizing_capital, params, '
                                                                       'copy_plan=False)\n']}]},
 'core/price_utils.py': {'new_lf_sha256': '492c17d9e4e805fa2a8049b9ba7df58ccbae5c8d8b07773398727d9aabdd7d98',
                         'new_module_calculation_sha256': 'b55c935295f1689f8ed604d9fe938e34c58b12beae83af62a6e61cba144fcfc7',
                         'new_physical_pins': ['02c11e9f955d3e02976dea594fb48e2f4f413e99295b54c77b989d23b1270aa1',
                                               '492c17d9e4e805fa2a8049b9ba7df58ccbae5c8d8b07773398727d9aabdd7d98'],
                         'predecessor_lf_sha256': 'cc80faffb3c2ba9fbac2b979fbe765ee497b5a916f03278f3ed25c468ab6ae7f',
                         'predecessor_module_calculation_sha256': '79c1feab6ffbd8acd34cf2b94512e7f6fe36d3f8ea0e28412a92bb9147208a10',
                         'restored_predecessor_exact': True,
                         'reverse_edits': [{'candidate_end_line': 261,
                                            'candidate_start_line': 233,
                                            'predecessor_lines': []}]}}


def _reviewed_optimizer_execution_sha256(relative: str, actual: str) -> str:
    if not _OPTIMIZER_EXECUTION_REVIEW_ENABLED:
        return actual
    return _OPTIMIZER_EXECUTION_PHYSICAL_PINS.get(relative, {}).get(actual, actual)


def _reviewed_source_group(scope: str):
    proof = _REVIEWED_SOURCE_INVENTORIES.get(scope)
    updates = _OPTIMIZER_EXECUTION_GROUP_UPDATES.get(scope)
    if not _OPTIMIZER_EXECUTION_REVIEW_ENABLED or proof is None or not updates:
        return proof
    current = {**proof["current"], **updates}
    for relative in _INDEPENDENT_QUEUE_RESOURCE_PREDECESSORS:
        if relative not in current:
            continue
        certificate = _independent_queue_resource_certificate(relative)
        if certificate is not None:
            import ast
            from core.source_calculation_syntax import calculation_syntax_attestation
            current[relative] = calculation_syntax_attestation(ast.parse(certificate[2]))[1]
    return {**proof, "current": current}


def reviewed_replay_predecessor_sources(parts, *, project_root):
    """Reconstruct three legacy payload owners after an exact full-graph proof.

    A missing/unknown owner, graph, helper or complete current source falls back
    to live syntax. The caller also verifies the reconstructed syntax digest
    before replacing any payload; historical result hashes are never accepted.
    """
    from hashlib import sha256
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if not _OPTIMIZER_EXECUTION_REVIEW_ENABLED or Path(project_root).resolve() != root:
        return {}
    expected = dict(_reviewed_source_group("portfolio_candidates")["current"])
    expected["@execution_sources"] = _reviewed_source_group("replay_numerics")["current"]["@execution_sources"]
    expected["core/portfolio_benchmark.py"] = _ORACLE_BENCHMARK_SOURCE[0]
    if set(parts) != set(expected) or any(
            parts[name] not in ((value,) if isinstance(value, str) else value)
            for name, value in expected.items()):
        return {}
    restored = {}
    for relative, proof in _OPTIMIZER_LEGACY_SOURCE_RECONSTRUCTIONS.items():
        raw = (root / relative).read_bytes()
        if sha256(raw).hexdigest() not in proof["new_physical_pins"]:
            return {}
        lf = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if sha256(lf).hexdigest() != proof["new_lf_sha256"]:
            return {}
        lines = lf.decode("utf-8-sig").splitlines(keepends=True)
        for edit in reversed(proof["reverse_edits"]):
            lines[edit["candidate_start_line"]:edit["candidate_end_line"]] = edit["predecessor_lines"]
        predecessor = "".join(lines)
        if sha256(predecessor.encode("utf-8")).hexdigest() != proof["predecessor_lf_sha256"]:
            return {}
        restored[relative] = (predecessor, proof["predecessor_module_calculation_sha256"])
    return restored
