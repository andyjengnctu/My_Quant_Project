"""Source-only dependency projection for the Oracle numerical build gate.

Reachability is derived from executable names/imports, not a historical SHA
allowlist. Renderers and unreferenced helper definitions are not numerical
inputs. The same reader API works on a frozen ZIP without importing its code.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Callable


_ROOTS = {
    "core/portfolio_engine.py": ("run_portfolio_timeline",),
    "services/portfolio_replay.py": (
        "run_portfolio_simulation_prepared", "run_portfolio_simulation_with_param_schedule",
        "run_portfolio_simulation_with_param_ensemble",
    ),
    "services/audit/daily_trade40_oracle.py": ("build_daily_universal_trade40_oracle_scores",),
    "services/audit/fulltrade_oracle_truth.py": ("build_fulltrade_oracle_scores",),
    "services/audit/daily_full_p20_oracle_truth.py": ("build_daily_full_p20_oracle_scores",),
    "services/audit/target_oracle_romd.py": (
        "_build_oracle_score_panels", "_build_price_positive_scores", "_resolve_oracle_target_score_contracts",
        "_metrics", "_execute_single_stage_missing", "_run_scenario", "_aggregate",
    ),
    "filters/breakout_quality/daily_ranker_data.py": ("load_daily_universal_ranker_data",),
    "core/execution_source_dependencies.py": ("execution_source_inventory",),
    "core/daily_trade_path_capabilities.py": ("DAILY_TRUTH_CAPABILITIES",),
}
_LOCAL_ROOTS = {"core", "filters", "services", "strategies", "config"}
_ADMISSION_OWNERS = {
    "services/research/oracle_build_attestation.py",
    "services/research/oracle_artifact_contract.py",
    "services/research/oracle_result_contract.py",
    "services/research/oracle_legacy_admission.py",
    "services/audit/oracle_truth_references.py",
    "services/audit/mintrade_truth_source_attestation.py",
    "core/research_execution_equivalence.py",
}
_PRESENTATION_ENTRY_POINTS = {
    "filters/breakout_quality/strategy_compare_reporting.py": frozenset({
        "render_strategy_pair_simple_report", "render_strategy_pair_markdown",
        "materialize_strategy_pair_readable_report",
    }),
}


@lru_cache(maxsize=1024)
def _parse_source(relative: str, raw: bytes):
    return ast.parse(raw.decode("utf-8-sig"), filename=relative)


class _Syntax(ast.NodeTransformer):
    def _owner(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        return node

    visit_Module = _owner
    visit_FunctionDef = _owner
    visit_AsyncFunctionDef = _owner
    visit_ClassDef = _owner


def _assigned(node):
    targets = (node.targets if isinstance(node, ast.Assign) else
               [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
    return {item.id for target in targets for item in ast.walk(target)
            if isinstance(item, ast.Name)}


@lru_cache(maxsize=512)
def _analyze_module(relative: str, raw: bytes):
    """Private read-only AST analysis keyed by the complete source content."""
    tree = _parse_source(relative, raw)
    definitions, imports, effects = {}, {}, []
    package = relative[:-3].split("/")[:-1]
    deferred_annotations = any(isinstance(item, ast.ImportFrom) and item.module == "__future__"
                               and any(alias.name == "annotations" for alias in item.names)
                               for item in tree.body)

    def eager_header(function):
        expressions = [*function.args.defaults,
                       *(item for item in function.args.kw_defaults if item is not None)]
        if not deferred_annotations:
            arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs,
                         function.args.vararg, function.args.kwarg]
            expressions.extend(item.annotation for item in arguments
                               if item is not None and item.annotation is not None)
            if function.returns is not None:
                expressions.append(function.returns)
        return [expression for expression in expressions
                if any(isinstance(item, (ast.Call, ast.Attribute, ast.Subscript, ast.BinOp,
                                         ast.UnaryOp, ast.Compare, ast.NamedExpr))
                       for item in ast.walk(expression))]

    for position, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = (position, node)
            if isinstance(node, ast.ClassDef):
                # Bases/metaclasses may register or inspect every method.
                if (node.bases or node.keywords or node.decorator_list
                        or any(isinstance(child, (ast.Call, ast.Global, ast.Nonlocal))
                               or (isinstance(child, (ast.Attribute, ast.Subscript))
                                   and isinstance(child.ctx, ast.Store))
                               for item in node.body
                               if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                               for child in ast.walk(item))):
                    effects.append((position, node))
                else:
                    headers = [expression for item in node.body
                               if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                               for expression in (*item.decorator_list, *eager_header(item))]
                    if headers:
                        effects.append((position, node))
            elif node.decorator_list:
                # A decorator receives the function object and can inspect
                # its body or publish it through a dynamic registry.
                effects.append((position, node))
            else:
                eager = eager_header(node)
                if eager:
                    effects.append((position, ast.Expr(value=ast.Tuple(elts=eager, ctx=ast.Load()))))
        else:
            for name in _assigned(node):
                definitions[name] = (position, node)
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0], None, node)
        elif isinstance(node, ast.ImportFrom):
            parent = package[:len(package) - node.level + 1] if node.level else []
            module = ".".join(parent + ([node.module] if node.module else []))
            for alias in node.names:
                imports[alias.asname or alias.name] = (module, alias.name, node)
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Nondeclarative module effects execute even when no name is used.
            if (not isinstance(node, (ast.Assign, ast.AnnAssign))
                    or any(isinstance(child, ast.Call) for child in ast.walk(node))):
                if not (position == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)):
                    effects.append((position, node))
    node_refs = {id(node): (position, False) for position, node in enumerate(tree.body)}
    for position, node in effects:
        node_refs.setdefault(id(node), (position, True))
    return (tree, MappingProxyType(definitions), MappingProxyType(imports), tuple(effects),
            MappingProxyType(node_refs))


@lru_cache(maxsize=8192)
def _node_facts(relative: str, raw: bytes, position: int, effect: bool):
    """Cache lexical facts, never cross-module lookup or reader availability."""
    tree, _, _, effects, _ = _analyze_module(relative, raw)
    node = dict(effects)[position] if effect else tree.body[position]
    imports, dynamic_modules, delegated = {}, [], []
    used, attributes = set(), defaultdict(set)
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            used.add(child.id)
        elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            attributes[child.value.id].add(child.attr)
        if isinstance(child, ast.Import):
            for alias in child.names:
                imports[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0], None)
        elif isinstance(child, ast.ImportFrom) and child.module and not child.level:
            for alias in child.names:
                imports[alias.asname or alias.name] = (child.module, alias.name)
        elif (isinstance(child, ast.Call) and child.args
              and isinstance(child.args[0], ast.Constant) and isinstance(child.args[0].value, str)
              and ((isinstance(child.func, ast.Name) and child.func.id in {"import_module", "__import__"})
                   or (isinstance(child.func, ast.Attribute) and child.func.attr == "import_module"))):
            dynamic_modules.append(child.args[0].value)
        elif isinstance(child, (ast.Tuple, ast.List)):
            for owner, function in zip(child.elts, child.elts[1:]):
                if (isinstance(owner, ast.Constant) and isinstance(owner.value, str)
                        and isinstance(function, ast.Constant) and isinstance(function.value, str)
                        and function.value.isidentifier()):
                    delegated.append((owner.value, function.value))
    return (tuple((name, module, symbol) for name, (module, symbol) in imports.items()),
            tuple(dynamic_modules), tuple(delegated), frozenset(used),
            tuple((name, frozenset(values)) for name, values in attributes.items()))


@lru_cache(maxsize=1024)
def _project_module(relative: str, raw: bytes, selected_names: tuple[str, ...]) -> str:
    """Immutable projection keyed by source bytes and the entire selection."""
    tree, definitions, imports, effects, _ = _analyze_module(relative, raw)
    nodes = {position: node for position, node in effects}
    positions = {id(node): position for position, node in enumerate(tree.body)}
    for name in selected_names:
        if name in definitions:
            position, node = definitions[name]
            nodes[position] = node
        elif name in imports:
            node = imports[name][2]
            nodes[positions[id(node)]] = node
    projected = ast.Module(body=[node for _, node in sorted(nodes.items())], type_ignores=[])
    # The cached parse/analysis remains untouched. Only a first-time selected
    # projection needs a private mutable copy for docstring normalization.
    return ast.dump(_Syntax().visit(deepcopy(projected)), include_attributes=False)


@lru_cache(maxsize=16)
def _whole_source_projection(relative: str, raw: bytes) -> str:
    return ast.dump(_Syntax().visit(deepcopy(_parse_source(relative, raw))), include_attributes=False)


def oracle_build_source_parts_from_reader(read_source: Callable[[str], bytes]) -> dict[str, str]:
    """Return the selected executable AST closure, including native source.

    Unknown local imports fail closed. Function/class bodies and referenced
    globals are followed transitively; runtime module attribute access follows
    its imported owner. An opaque module use conservatively includes its public
    definitions. Top-level executable effects remain in the projection.
    """
    modules = {}
    selected = defaultdict(set)
    pending = []
    contents = {}

    def read_optional(relative):
        if relative in contents:
            return contents[relative]
        try:
            value = read_source(relative)
        except (FileNotFoundError, KeyError):
            value = None
        contents[relative] = value
        return value

    def local_module(module):
        if not module or module.split(".")[0] not in _LOCAL_ROOTS:
            return None
        relative = module.replace(".", "/") + ".py"
        if read_optional(relative) is not None:
            return relative
        relative = module.replace(".", "/") + "/__init__.py"
        if read_optional(relative) is not None:
            return relative
        return None

    def load(relative):
        if relative in modules:
            return modules[relative]
        raw = read_optional(relative)
        if raw is None:
            raise ValueError(f"Oracle numerical owner is missing: {relative}")
        analyzed = _analyze_module(relative, raw)
        modules[relative] = analyzed
        for _, effect in analyzed[3]:
            pending.append((relative, effect))
        return analyzed

    def select(relative, name):
        # These owners prove/adopt results; their call sites stay visible, but
        # their implementation is checked by formal contract tests rather
        # than recursively becoming its own numerical build prerequisite.
        if relative in _ADMISSION_OWNERS:
            return
        _, definitions, imports, _, _ = load(relative)
        if name == "*":
            for declared in definitions:
                select(relative, declared)
            return
        if name in selected[relative]:
            return
        selected[relative].add(name)
        if name in definitions:
            pending.append((relative, definitions[name][1]))
        elif name in imports:
            module, symbol, node = imports[name]
            follow_import(relative, module, symbol, None)
        elif "*" in imports:
            module, _, _ = imports["*"]
            selected[relative].add("*")
            follow_import(relative, module, name, None)
        else:
            raise ValueError(f"Oracle numerical dependency is missing: {relative}:{name}")

    def follow_import(relative, module, symbol, attributes):
        if module.split(".")[0] not in _LOCAL_ROOTS:
            return
        if symbol and symbol != "*":
            submodule = local_module(module + "." + symbol)
            if submodule is not None:
                destination = submodule
                symbol = None
            else:
                destination = local_module(module)
        else:
            destination = local_module(module)
        if destination is None:
            raise ValueError(f"Oracle numerical local import is missing: {module}.{symbol or ''}")
        if symbol in _PRESENTATION_ENTRY_POINTS.get(destination, ()):
            # These explicit presentation sinks publish human-readable
            # output. Every other referenced helper, including newly added
            # numerical helpers in the reporting module, is followed.
            return
        if symbol:
            select(destination, symbol)
        elif attributes:
            for attribute in attributes:
                select(destination, attribute)
        else:
            select(destination, "*")

    for relative, names in _ROOTS.items():
        for name in names:
            select(relative, name)
    observation = "services/audit/oracle_observation.py"
    if read_optional(observation) is not None:
        select(observation, "oracle_observation")
    else:
        select("services/audit/target_oracle_romd.py", "_metrics")

    visited_nodes = set()
    while pending:
        relative, node = pending.pop()
        marker = (relative, id(node))
        if marker in visited_nodes:
            continue
        visited_nodes.add(marker)
        _, definitions, imports, _, node_refs = load(relative)
        # Cached facts contain only immutable names/declarations. Resolution
        # remains local to this reader and rechecks every actual source byte.
        facts = _node_facts(relative, contents[relative], *node_refs[id(node)])
        local_imports, dynamic_modules, delegated, used, attributes = facts
        nested_imports = dict(imports)
        nested_imports.update({name: (module, symbol, None)
                               for name, module, symbol in local_imports})
        for module in dynamic_modules:
            destination = local_module(module)
            if destination is not None:
                select(destination, "*")
        for declared, function in delegated:
            if declared.endswith(".py") and declared.split("/")[0] in _LOCAL_ROOTS:
                select(declared, function)
            elif declared.split(".")[0] in _LOCAL_ROOTS:
                destination = local_module(declared)
                if destination is not None:
                    select(destination, function)
        attribute_uses = dict(attributes)
        for name in used:
            if name in definitions:
                select(relative, name)
            elif name in nested_imports:
                module, symbol, imported = nested_imports[name]
                if name in imports:
                    selected[relative].add(name)
                follow_import(relative, module, symbol, attribute_uses.get(name))

    parts = {relative: _project_module(relative, contents[relative], tuple(sorted(selected[relative])))
             for relative in modules}
    # Native loading is indirect (ctypes/path literals); its numerical owner
    # is declared by the same source capability graph as Python execution.
    native = "core/native_optimizer_hotpath.c"
    parts[native] = read_source(native).decode("utf-8-sig")
    projection_owner = "services/research/oracle_numerical_sources.py"
    projection_bytes = read_optional(projection_owner)
    if projection_bytes is not None:
        parts[projection_owner] = _whole_source_projection(projection_owner, projection_bytes)
    return dict(sorted(parts.items()))


def oracle_build_source_parts(project_root: Path | None = None) -> dict[str, str]:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    return oracle_build_source_parts_from_reader(lambda relative: (root / relative).read_bytes())
