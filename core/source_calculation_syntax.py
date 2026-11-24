"""Canonical structural syntax used to attest calculation source code."""
from __future__ import annotations

import ast

from core.file_integrity import canonical_json_sha256


class _CalculationSyntax(ast.NodeTransformer):
    """Ignore comments/docstrings and inactive target-ID branches, not math.

    Only literal comparisons with the bound target ID are folded. No source
    expression is executed, and every unknown condition remains in the proof.
    Registering another target branch therefore cannot revise an existing
    target's calculation, while changing its selected arithmetic does.
    """
    def __init__(self, literals=None):
        self.literals = dict(literals or {})

    def _literal(self, node):
        if isinstance(node, ast.Name) and node.id in self.literals:
            return self.literals[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
            return tuple(self._literal(value) for value in node.elts)
        raise ValueError("not a bound literal")

    def _condition(self, node):
        if isinstance(node, (ast.Name, ast.Constant)):
            value = self._literal(node)
            if isinstance(value, bool):
                return value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self._literal(node.left), self._literal(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.In):
                return left in right
            if isinstance(op, ast.NotIn):
                return left not in right
            if isinstance(op, ast.Eq):
                return left == right
            if isinstance(op, ast.NotEq):
                return left != right
            if isinstance(op, ast.Is):
                return left is right
            if isinstance(op, ast.IsNot):
                return left is not right
        if isinstance(node, ast.BoolOp):
            known = []
            for value in node.values:
                try:
                    known.append(self._condition(value))
                except ValueError:
                    known.append(None)
            if isinstance(node.op, ast.And) and False in known:
                return False
            if isinstance(node.op, ast.Or) and True in known:
                return True
            if None not in known:
                return all(known) if isinstance(node.op, ast.And) else any(known)
        raise ValueError("not a bound comparison")

    def visit_If(self, node):
        try:
            branch = node.body if self._condition(node.test) else node.orelse
        except ValueError:
            before = dict(self.literals)
            node.body = self._statements(node.body)
            self.literals = dict(before)
            node.orelse = self._statements(node.orelse)
            self.literals = before
            return node
        return self._statements(branch)

    def _statements(self, statements):
        result = []
        for statement in statements:
            transformed = self.visit(statement)
            if transformed is not None:
                result.extend(transformed if isinstance(transformed, list) else [transformed])
        return result

    def visit_IfExp(self, node):
        try:
            selected = node.body if self._condition(node.test) else node.orelse
        except ValueError:
            return self.generic_visit(node)
        return self.visit(selected)

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    self.literals[target.id] = self._literal(node.value)
                except ValueError:
                    self.literals.pop(target.id, None)
        return node

    def _local_scope(self, node):
        before = dict(self.literals)
        node = self.generic_visit(node)
        self.literals = before
        return node

    visit_For = _local_scope
    visit_While = _local_scope
    visit_Try = _local_scope
    visit_ClassDef = _local_scope

    def visit_FunctionDef(self, node):
        # Annotations and docstrings do not execute under future annotations.
        node.returns = None
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            argument.annotation = None
        return self._local_scope(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def _syntax_payload(value):
    # ast.dump's treatment of optional/empty fields changed in Python 3.13.
    # This structural encoding is stable across the supported interpreters,
    # including the user's Python 3.14. Locations/empty new AST fields are not
    # execution semantics and must not invalidate an existing result.
    if isinstance(value, ast.AST):
        return {"node": type(value).__name__, **{
            name: _syntax_payload(item) for name, item in ast.iter_fields(value)
            if item is not None and item != []}}
    if isinstance(value, list):
        return [_syntax_payload(item) for item in value]
    return value


def calculation_syntax_attestation(tree):
    """Hash calculation structure with the same projection for every consumer."""
    payload = _syntax_payload(_CalculationSyntax().visit(tree))
    return payload, canonical_json_sha256(payload)
