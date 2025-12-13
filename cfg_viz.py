"""Graphviz visualization helpers for CFGs.

Provides `render_cfg_dot(cfg, liveness=None, collapse_liveness=False)` which
returns a `graphviz.Digraph` object (not rendered). Optionally `write_and_render`
can write the file to disk.

Block layout: each basic block is rendered as an HTML-like table node. The
statements inside a block are laid out side-by-side as table cells. Liveness
information may be displayed per-statement and highlighted with background
colors.
"""

from typing import Dict, Any, Optional
from graphviz import Digraph
from pretty_printer import PrettyPrinter
import html


def _stmt_html(stmt_str: str, live_in=None, live_out=None, highlight=False) -> str:
    # Convert multiline stmt string into HTML with <br/>
    escaped = html.escape(stmt_str)
    escaped = escaped.replace("\n", "<br/>")
    live_html = ""
    if live_in is not None or live_out is not None:
        lin = ", ".join(sorted(live_in)) if live_in else ""
        lout = ", ".join(sorted(live_out)) if live_out else ""
        # Use FONT to render smaller text (SMALL is not supported in Graphviz HTML-like labels)
        live_html = f'<BR/><FONT POINT-SIZE="8">in: {html.escape(lin)} out: {html.escape(lout)}</FONT>'
    # cell with optional highlight background
    bgcolor = ' BGCOLOR="#ffefef"' if highlight else ""
    # Use FONT for the main statement text as well
    return f'<TD{bgcolor}><FONT POINT-SIZE="10">{escaped}</FONT>{live_html}</TD>'


def render_cfg_dot(
    cfg: Dict[str, Any],
    liveness: Optional[Dict[str, Any]] = None,
    collapse_liveness: bool = False,
    use_surface: bool = False,
    include_liveness: bool = True,
) -> Digraph:
    """Return a graphviz.Digraph for the given CFG and optional liveness mapping.

    The caller may call `dot.source` to inspect the dot text, or call
    `dot.render(filename, format=...)` to write files (requires Graphviz installed).
    """
    dot = Digraph(format="svg")
    dot.attr("graph", rankdir="LR")

    # Create nodes
    for b in cfg.get("blocks", []):
        lbl = b["label"]
        stmts = b.get("statements", [])
        # header: label and block-level live sets (optional)
        live_in = []
        live_out = []
        if include_liveness and liveness and lbl in liveness:
            live_in = sorted(list(liveness[lbl].get("live_in", [])))
            live_out = sorted(list(liveness[lbl].get("live_out", [])))
        # Use quoted attributes and uppercase tags to satisfy Graphviz HTML-like label requirements
        header = f"<TR><TD COLSPAN=\"999\"><B>{html.escape(lbl)}</B> &nbsp; <FONT POINT-SIZE=\"8\">in: {html.escape(', '.join(live_in))} out: {html.escape(', '.join(live_out))}</FONT></TD></TR>"
        # statements row: each cell is a statement
        cells = []
        instr_l = []
        if include_liveness and liveness and lbl in liveness:
            instr_l = liveness[lbl].get("instr_liveness", [])
        for i, stmt in enumerate(stmts):
            stmt_str = (
                PrettyPrinter.print_surface(stmt) if use_surface else PrettyPrinter.print_ast(stmt)
            )
            live_in_i, live_out_i = (None, None)
            highlight = False
            if instr_l and i < len(instr_l):
                live_in_i, live_out_i = instr_l[i]
                # highlight if any variable is live_out
                highlight = bool(live_out_i)

            # If include_liveness is False, suppress per-statement live info
            if not include_liveness:
                live_in_i = None
                live_out_i = None
                highlight = False

            cells.append(
                _stmt_html(stmt_str, live_in_i, live_out_i, highlight=highlight)
            )

        # assemble table: header + statements row
        stmts_row = ""
        if cells:
            stmts_row = "<TR>" + "".join(cells) + "</TR>"
        # Use uppercase TABLE and quoted attributes
        table_html = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">{header}{stmts_row}</TABLE>>'
        dot.node(lbl, label=table_html, shape="plaintext")

    # Add edges
    for b in cfg.get("blocks", []):
        for succ in b.get("out_edges", []):
            dot.edge(b["label"], succ)

    return dot


def write_and_render(
    cfg: Dict[str, Any],
    out_path: str,
    liveness: Optional[Dict[str, Any]] = None,
    collapse_liveness: bool = False,
    fmt: str = "svg",
    use_surface: bool = False,
    include_liveness: bool = True,
) -> None:
    """Write and render the CFG to the given path (without extension). Returns when rendered.

    Example: write_and_render(cfg, 'out/cfg', ... , fmt='png') will create out/cfg.png
    (requires Graphviz)."""
    dot = render_cfg_dot(
        cfg,
        liveness=liveness,
        collapse_liveness=collapse_liveness,
        use_surface=use_surface,
        include_liveness=include_liveness,
    )
    dot.format = fmt
    # Note: render will append extension automatically
    dot.render(out_path, cleanup=True)
