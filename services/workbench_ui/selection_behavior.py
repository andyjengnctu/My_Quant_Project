"""Shared Workbench row-selection UX contract.

All selectable row controls use the same toggle rule: clicking the currently
selected row again clears the selection.  Consumers may attach an ``on_clear``
callback to reset an edit form; the helper never mutates domain state.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from tkinter import ttk


def toggled_row_selection(current_id: str | None, clicked_id: str | None) -> str | None:
    """Canonical Workbench row-toggle rule for both custom tables and Treeviews."""

    current = None if current_id is None else str(current_id)
    clicked = None if clicked_id is None else str(clicked_id)
    if not clicked:
        return None
    return None if current == clicked else clicked


def clear_treeview_selection(tree: ttk.Treeview, *, on_clear: Callable[[], None] | None = None) -> bool:
    selected = tuple(tree.selection())
    changed = bool(selected)
    if selected:
        tree.selection_remove(*selected)
    tree.focus("")
    if changed and callable(on_clear):
        on_clear()
    return changed


def handle_treeview_toggle_click(
    tree: ttk.Treeview,
    event,
    *,
    ignore_columns: Iterable[str] = (),
    clear_on_empty: bool = True,
    on_clear: Callable[[], None] | None = None,
):
    """Return ``"break"`` when the click is consumed as an explicit deselect.

    Call this from a panel's existing ``<Button-1>`` handler before normal row
    actions.  Heading clicks are ignored so sorting/navigation contracts remain
    untouched.  ``ignore_columns`` is useful for explicit action cells such as
    the single-stock inspector icon.
    """

    region = str(tree.identify_region(event.x, event.y) or "")
    if region == "heading":
        return None
    column = str(tree.identify_column(event.x) or "")
    if column in {str(value) for value in ignore_columns}:
        return None
    row_id = str(tree.identify_row(event.y) or "")
    selected = set(str(value) for value in tree.selection())
    if row_id and row_id in selected:
        clear_treeview_selection(tree, on_clear=on_clear)
        return "break"
    if not row_id and selected and clear_on_empty:
        clear_treeview_selection(tree, on_clear=on_clear)
        return "break"
    return None


def bind_treeview_toggle_selection(
    tree: ttk.Treeview,
    *,
    ignore_columns: Iterable[str] = (),
    clear_on_empty: bool = True,
    on_clear: Callable[[], None] | None = None,
) -> None:
    """Bind the shared toggle-selection behavior to a plain ttk.Treeview."""

    def _handler(event):
        return handle_treeview_toggle_click(
            tree,
            event,
            ignore_columns=ignore_columns,
            clear_on_empty=clear_on_empty,
            on_clear=on_clear,
        )

    tree.bind("<Button-1>", _handler, add="+")


__all__ = [
    "bind_treeview_toggle_selection",
    "clear_treeview_selection",
    "handle_treeview_toggle_click",
    "toggled_row_selection",
]
