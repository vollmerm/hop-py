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
import re
from ast_nodes import NodeType, ExpressionStatementNode, IdentifierNode
from graphviz import Digraph
from pretty_printer import PrettyPrinter
import html


def _format_instr(instr: Dict[str, Any]) -> str:
    """Return a short string for an instruction dict."""
    if not isinstance(instr, dict):
        return html.escape(str(instr))
    op = instr.get("op", "")
    # Build arg list
    parts = [op]
    for k in (
        "rd",
        "rs1",
        "rs2",
        "rs",
        "base",
        "offset",
        "imm",
        "func",
        "target",
        "name",
    ):
        if k in instr:
            parts.append(f"{k}={instr[k]}")
    return html.escape(" ".join(map(str, parts)))


def _stmt_html(stmt_str: str, live_in=None, live_out=None, highlight=False) -> str:
    # Convert multiline stmt string into HTML with <br/>
    escaped = html.escape(stmt_str)
    escaped = escaped.replace("\n", "<br/>")
    # Avoid empty FONT elements which some Graphviz versions reject
    if not escaped.strip():
        escaped = "&nbsp;"
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

    # Compute predecessor (incoming edges) map so we can detect isolated blocks
    preds = {b["label"]: [] for b in cfg.get("blocks", [])}
    for b in cfg.get("blocks", []):
        for succ in b.get("out_edges", []):
            if succ in preds:
                preds[succ].append(b["label"])

    # First pass: determine function ownership for each block. Prefer explicit
    # `function` metadata on the block (added by `build_cfg`); fall back to
    # detecting a nearby `FUNC_LABEL` instruction. We walk blocks in order and
    # propagate the most recently seen function label so that all blocks
    # belonging to a function are grouped together even if metadata is missing.
    block_infos = []
    current_func = None
    for b in cfg.get("blocks", []):
        lbl = b["label"]
        stmts = b.get("statements", [])
        func_name = b.get("function")
        if func_name:
            current_func = func_name
        else:
            # fallback detection inside instruction lists
            for s in stmts:
                if isinstance(s, list):
                    for instr in s:
                        if isinstance(instr, dict) and instr.get("op") == "FUNC_LABEL":
                            func_name = instr.get("name")
                            break
                    if func_name:
                        break
                else:
                    try:
                        if getattr(s, "type", None) and s.type == NodeType.FUNC_DECL:
                            func_name = getattr(s, "func_name", None)
                            break
                    except Exception:
                        pass
            # if still no explicit func found, inherit the last-seen function
            if not func_name:
                func_name = current_func
            else:
                current_func = func_name

        block_infos.append(
            {
                "label": lbl,
                "stmts": stmts,
                "func": func_name,
                "out_edges": list(b.get("out_edges", [])),
                "in_edges": list(preds.get(lbl, [])),
            }
        )

    # Group blocks by function (None -> global)
    groups = {}
    for info in block_infos:
        groups.setdefault(info["func"], []).append(info)

    # Filter out trivial empty isolated blocks (no statements, no in_edges, no out_edges)
    included_labels = set()
    filtered_infos = []
    for info in block_infos:
        if (not info["stmts"]) and (not info["in_edges"]) and (not info["out_edges"]):
            # skip this isolated empty block
            continue
        filtered_infos.append(info)
        included_labels.add(info["label"])

    # Emit nodes grouped into subgraphs per function
    for func, infos in groups.items():
        # filter infos to only included ones
        infos = [i for i in infos if i["label"] in included_labels]
        if not infos:
            continue
        cluster_name = (
            f"cluster_{re.sub(r'[^0-9A-Za-z_]', '_', func) if func else 'global'}"
        )
        with dot.subgraph(name=cluster_name) as c:
            if func:
                c.attr(label=f"function: {func}")
            else:
                c.attr(label="")
            c.attr(style="rounded")
            for info in infos:
                lbl = info["label"]
                stmts = info["stmts"]

                # header: label and optional block-level live sets (only when requested)
                live_in = []
                live_out = []
                if include_liveness and liveness and lbl in liveness:
                    live_in = sorted(list(liveness[lbl].get("live_in", [])))
                    live_out = sorted(list(liveness[lbl].get("live_out", [])))
                    # include live sets in header when liveness requested
                    if stmts:
                        header = f"<TR><TD COLSPAN=\"999\"><B>{html.escape(lbl)}</B> &nbsp; <FONT POINT-SIZE=\"8\">in: {html.escape(', '.join(live_in))} out: {html.escape(', '.join(live_out))}</FONT></TD></TR>"
                    else:
                        # For empty blocks avoid large colspan which can force wide layout
                        header = f"<TR><TD><B>{html.escape(lbl)}</B> &nbsp; <FONT POINT-SIZE=\"8\">in: {html.escape(', '.join(live_in))} out: {html.escape(', '.join(live_out))}</FONT></TD></TR>"
                else:
                    # omit the in/out text when liveness is not included
                    if stmts:
                        header = (
                            f'<TR><TD COLSPAN="999"><B>{html.escape(lbl)}</B></TD></TR>'
                        )
                    else:
                        header = f"<TR><TD><B>{html.escape(lbl)}</B></TD></TR>"

                # statements row: each cell is a statement
                cells = []
                instr_l = []
                if include_liveness and liveness and lbl in liveness:
                    instr_l = liveness[lbl].get("instr_liveness", [])
                for i, stmt in enumerate(stmts):
                    # Statement may be a list of instruction dicts (after instrsel), a single
                    # instruction dict, or an AST node. Handle all cases.
                    if isinstance(stmt, list):
                        # Emit one cell per instruction in the statement list.
                        # This avoids concatenating multiple instructions into a
                        # single cell and skips empty instruction lists.
                        if not stmt:
                            continue
                        # For statements that are lists, we'll create multiple
                        # cells and treat the statement-level liveness the same
                        # for each instruction (we don't have per-instr liveness
                        # granularity here).
                        for instr in stmt:
                            stmt_str = _format_instr(instr)
                            live_in_i, live_out_i = (None, None)
                            highlight = False
                            if instr_l and i < len(instr_l):
                                live_in_i, live_out_i = instr_l[i]
                                highlight = bool(live_out_i)
                            if not include_liveness:
                                live_in_i = None
                                live_out_i = None
                                highlight = False
                            cells.append(
                                _stmt_html(
                                    stmt_str, live_in_i, live_out_i, highlight=highlight
                                )
                            )
                        # we've handled this statement (which produced multiple cells)
                        continue
                    elif isinstance(stmt, dict):
                        stmt_str = _format_instr(stmt)
                    else:
                        # Skip trivial temporary-only expression statements like `_tmp8;`
                        if (
                            isinstance(stmt, ExpressionStatementNode)
                            and isinstance(
                                getattr(stmt, "expression", None), IdentifierNode
                            )
                            and getattr(stmt.expression, "name", "").startswith("_tmp")
                        ):
                            # render as empty cell (skip)
                            continue
                        stmt_str = (
                            PrettyPrinter.print_surface(stmt)
                            if use_surface
                            else PrettyPrinter.print_ast(stmt)
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

                # assemble table: header + optional func row + statements row
                func_row = ""
                if func:
                    func_row = f'<TR><TD COLSPAN="999"><FONT POINT-SIZE="8"><I>function: {html.escape(func)}</I></FONT></TD></TR>'

                stmts_row = ""
                if cells:
                    stmts_row = "<TR>" + "".join(cells) + "</TR>"
                # Use uppercase TABLE and quoted attributes
                table_html = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">{header}{func_row}{stmts_row}</TABLE>>'
                c.node(lbl, label=table_html, shape="plaintext")

    # Add edges between included blocks only
    for b in cfg.get("blocks", []):
        src = b["label"]
        if src not in included_labels:
            continue
        for succ in b.get("out_edges", []):
            if succ in included_labels:
                dot.edge(src, succ)

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
