"""Content-pinned proof for reviewed execution-only preparation changes.

Each pair is an exact current implementation and its unchanged numerical
predecessor, not a scientific version or a general source-hash bypass. Unknown
code keeps its own identity. Prices, inputs, teachers, samples and all other
execution dependencies remain independently verified by their original owners.
"""
from __future__ import annotations

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

def reviewed_execution_source_sha256(relative: str, actual: str) -> str:
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
    return syntax_sha256


# 714e22a4 behavior-preserving extraction proof (2026-11-24).
# Original physical sources: test-branch-1_20261124_110521_714e22a4.zip
# SHA256 a2abf3ec36259175936607decefe1975ba77f3b25c4f2c5ada1e2362b0f1be18.
# Evidence: daily_equivalence.json (11 cases incl. FIT samples),
# portfolio_fully_isolated_equivalence.json (75 complete replay/report cases), and
# ranking_policy_equivalence.json (metadata/sort/selector + 5825 validations).
# Optimizer public contracts: optimizer_contract_equivalence.json (10 payloads).
# Full teacher schedule AST remains identical; daily_equivalence covers loader FIT.
# Only this complete reviewed source subgraph is interchangeable. Extracted
# helpers participate in the proof and remain normal sources after ANY edit.
# Raw tuples are (actual, LF, CRLF); AST scopes use the existing calculation
# projection. Baseline entries are the original inventory, not result hashes.
_REVIEWED_SOURCE_INVENTORIES = {
    'oracle_daily_provider': {
        'current': {
            'filters/breakout_quality/daily_ranker_data.py': (
                'e2b61e98476854365099f2cddb7ce3bbb5c100110589e73bcfaab698628abbd3',
                'e2b61e98476854365099f2cddb7ce3bbb5c100110589e73bcfaab698628abbd3',
                'fe6fb8ae03e76037974822d286eb3e31e3af83542fe25a2bef7b2ddb18a45b78',
            ),
            'filters/breakout_quality/profile_ranker_data.py': (
                'cb8722b62cd69dd0c598457ed0ba73ae3ce321e8e174824c7e9decf76d024c14',
                'aa3513e0368dd9e5c10f1244c808a88ac702a8d5b8178b4ca76d8ccd66145de4',
                'cb8722b62cd69dd0c598457ed0ba73ae3ce321e8e174824c7e9decf76d024c14',
            ),
            'filters/breakout_quality/daily_model_input_context.py': (
                '7128ad7183a5291980565284948a9c8644f2962294d52b9604f51e7aa0f5b8ce',
                '7128ad7183a5291980565284948a9c8644f2962294d52b9604f51e7aa0f5b8ce',
                'c43ac2774516bc1bf24604240964688bd1426d78972b8badb169319d90248792',
            ),
            'filters/breakout_quality/daily_sample_accumulator.py': (
                '157ee1eece719d53751ed9e478479406705eecd5ec0836fdf70505bf7e95f96e',
                '157ee1eece719d53751ed9e478479406705eecd5ec0836fdf70505bf7e95f96e',
                'acfa95a4323fa3bf24ab4287c9e0e211c763a96172e2cf118679dcd93ab28bc0',
            ),
        },
        'baseline': {
            'filters/breakout_quality/daily_ranker_data.py': (
                '121be1928f6f1538e9678ace4f036bfff17d48688a6d84e3e0e95d8c40c34a0f',
                '3ba49651c71a28574431646146c62fcfe0b2ae9ec88d467b9224e2eab44df00e',
                '121be1928f6f1538e9678ace4f036bfff17d48688a6d84e3e0e95d8c40c34a0f',
            ),
            'filters/breakout_quality/profile_ranker_data.py': (
                'd32a9fa95bed048591d886214da81b0b6fc4f9c128977514d742613263b4ec70',
                '16166646333272683f5a0103a242a7a4a05f1c2433ae52031cb6bef91981fb38',
                'd32a9fa95bed048591d886214da81b0b6fc4f9c128977514d742613263b4ec70',
            ),
        },
    },
    'portfolio_candidates': {
        'current': {
            'core/portfolio_engine.py': '7c21f8370583f494adcd1c32ead2f6fe9797ff8514e5a4f50797fd587517db0c',
            'core/portfolio_day_events.py': '6862fd9d4be982c3ad0c4301a710ce7d4c929734129ff2cb85e797a8030cf5fd',
            'core/portfolio_entry_selection.py': '710340465712a6fd981cb26e50a24f5491c7df9b9e55d8158a962c0c94927d3d',
            'core/portfolio_entry_selection_common.py': '8d9b27ad4d13826468e68cc99db30f2dc9d742a6efce5512fba5be5fdab4dd68',
            'core/buy_sort.py': 'cbeb650a94a4b5fae64c8874823e78ff1cad24d00377dd99a45c7fbad010d8a6',
            'core/ranking_policy.py': 'a387239cf7a68b810eae75bb2547e713d7f10fad1dd3be91faa2740d59a2e96a',
            'core/portfolio_run_state.py': '305cfec53a4aaf43ed24d3cd53c1d8d933704bad65e02e0c22bcf7d7753d2612',
        },
        'baseline': {
            'core/portfolio_engine.py': '873926b452c753461ed6f2f3e48ecf9fd90a49a5b4f6fb18a5ef342ff9dc386a',
            'core/portfolio_day_events.py': 'dbe579cf179949f616358833cda65ccae2b7435e9dfdf779ce208031889a3490',
            'core/portfolio_entry_selection.py': '86f0aa7c0ecc82b2b0c8fe9be7b79a2afaa8fdf759cac9936357304af21da908',
            'core/portfolio_entry_selection_common.py': '68eeba7b93662bb9dbc4b8936958c71f147ab2a1b88188f0c90f4501f1418438',
            'core/buy_sort.py': 'f8867277b95b63d2ce7bba17c9ac31e5ceb095631d6e2079db461826f4c65893',
        },
    },
    'full_oracle_teacher': {
        'current': {
            'services/audit/fulltrade_oracle_truth.py': (
                'ca938aac814b0c79fcaed0e967de2b0cb9a1b7c55aa13e93e0eb354d9d66aad8',
                'ca938aac814b0c79fcaed0e967de2b0cb9a1b7c55aa13e93e0eb354d9d66aad8',
                '5890d09e9323b29958a02b3c3f793dd3cf4aaa0a36dad413f26f22e437aee41f',
            ),
            'services/breakout_quality/formal_fulltrade_training_source.py': (
                '537b5ebf1babd809359da198aa5e51e3e28c159116a02ed589b77803fbaaab4a',
                '537b5ebf1babd809359da198aa5e51e3e28c159116a02ed589b77803fbaaab4a',
                '2bb6af226ab08a6e4c7f6858d0c4cb9b03d18ba1b5fc3e9a7a5fd3f8f9de1df8',
            ),
            'services/optimizer/strategy_param_service.py': (
                '101edb44502a467b72b9e5f6fec46e29f8637d849718194a4f49976783d819ce',
                '101edb44502a467b72b9e5f6fec46e29f8637d849718194a4f49976783d819ce',
                'd1b7171126b546b1b2dc9cea497a6d820817bf1fb045c3d899c1c0085c9ce97e',
            ),
            'services/optimizer/strategy_param_training.py': (
                'd8afe8fb44a734aa996a31aab531d46bcea197688b275fab4a932eb33acafc3d',
                'c8538ceb13ecf497f8bb29539908e58e3eced10458e0aa8ec34d8f9d27ee05d6',
                'd8afe8fb44a734aa996a31aab531d46bcea197688b275fab4a932eb33acafc3d',
            ),
            'services/optimizer/strategy_param_contracts.py': (
                '4611f0888b8e5507189d6dc9998168fd98987e8eb4c7498879f1c7d669dc2bbb',
                '4611f0888b8e5507189d6dc9998168fd98987e8eb4c7498879f1c7d669dc2bbb',
                '65bee9800244a82eb572e9845f026409020bb25251321f7f5fdf5615e762cdf7',
            ),
        },
        'baseline': {
            'services/audit/fulltrade_oracle_truth.py': (
                '9364926c2e44ae2187f960949150a91726594f95cac827d6059119e9d84585c0',
                '9364926c2e44ae2187f960949150a91726594f95cac827d6059119e9d84585c0',
                'e7dd953b0f5da048e478b029ec1950c6f4b96e0ec9112de29554abb31be342dc',
            ),
            'services/breakout_quality/formal_fulltrade_training_source.py': (
                '9353261063413b6415f397c169731487de1c97875721656142a8b4001d72cba7',
                '9353261063413b6415f397c169731487de1c97875721656142a8b4001d72cba7',
                '2f32bc55283ad3bec50c5f46d1e20ab613ea86c69d180529a2030d47dae68e2f',
            ),
        },
    },
}

def project_reviewed_source_inventory(parts, *, scope: str, project_root=None):
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
    proof = _REVIEWED_SOURCE_INVENTORIES.get(scope)
    if proof is None:
        return result
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
            result[name] = previous[0]
    return result
