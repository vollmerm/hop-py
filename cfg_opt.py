"""CFG-level optimizations and small peepholes.

This module provides a set of conservative transformations that operate on
the AST-level CFG (before instruction selection). The goal is to remove
common temporaries produced by the flattener and reduce unnecessary
instructions emitted by lowering, which also helps later register
allocation.

Implemented passes:
- copy-propagation (`copy_propagation_cfg`): a forward dataflow analysis
    that computes available identifier-to-identifier copies and rewrites
    uses where safe (merge by intersection at joins). Conservative by
    design: it only tracks pure id->id copies and avoids aliasing issues.
- peephole tmp-call elimination (`peephole_tmp_call_elim`): collapses
    the common pattern `_tmp = call(...); var x = _tmp;` into
    `var x = call(...)` and the three-statement pattern ending in a return
    into `return call(...)`.
- collapse `var` then `return` (`_collapse_vardecl_then_return`): turns
    `var x = call(...); return x;` into `return call(...)` when safe.
- dead-code-elimination (`dead_code_elim`): a conservative block-local
    eliminator that removes definitions whose results are not live and
    whose RHS has no side-effects. It accepts an aggressiveness `level`
    parameter (e.g. `"conservative"` vs `"aggressive"`).

The passes are intentionally conservative to avoid incorrect removal of
user-visible code. Typically the driver (`main.py`) controls when to run
these optimizations (via CLI flags), and `copy_propagation_cfg` will
invoke the peepholes and a conservative DCE internally.
"""

from typing import Dict, List, Optional, Set, Tuple
from ast_nodes import *
from liveness import analyze_liveness


def _compute_preds(cfg: Dict[str, any]) -> Dict[str, List[str]]:
    preds: Dict[str, List[str]] = {}
    for b in cfg.get("blocks", []):
        lbl = b["label"]
        preds.setdefault(lbl, [])
    for b in cfg.get("blocks", []):
        lbl = b["label"]
        for succ in b.get("out_edges", []):
            preds.setdefault(succ, []).append(lbl)
    return preds


def _merge_maps(maps: List[Dict[str, str]]) -> Dict[str, str]:
    if not maps:
        return {}
    # Intersection: keep mapping k->v only if all maps have same v for k
    keys = set(maps[0].keys())
    for m in maps[1:]:
        keys &= set(m.keys())
    out: Dict[str, str] = {}
    for k in keys:
        v = maps[0][k]
        if all(m.get(k) == v for m in maps[1:]):
            out[k] = v
    return out


def _replace_in_node(node: ASTNode, mapping: Dict[str, str]) -> ASTNode:
    # Mutate identifier uses in place according to mapping; returns node
    # For compound nodes, recurse.
    match node:
        case IdentifierNode(name=n):
            if n in mapping and mapping[n] != n:
                node.name = mapping[n]
            return node
        case BinaryOpNode(left=l, right=r):
            node.left = _replace_in_node(node.left, mapping)
            node.right = _replace_in_node(node.right, mapping)
            return node
        case UnaryOpNode(right=rv):
            node.right = _replace_in_node(node.right, mapping)
            return node
        case ArrayIndexNode(array=arr, index=idx):
            node.array = _replace_in_node(node.array, mapping)
            node.index = _replace_in_node(node.index, mapping)
            return node
        case FunctionCallNode(function=fn, arguments=args):
            node.function = _replace_in_node(node.function, mapping)
            node.arguments = [_replace_in_node(a, mapping) for a in node.arguments]
            return node
        case AssignmentNode(left=l, right=r):
            node.left = _replace_in_node(node.left, mapping)
            node.right = _replace_in_node(node.right, mapping)
            return node
        case ExpressionStatementNode(expression=expr):
            node.expression = _replace_in_node(node.expression, mapping)
            return node
        case IfStatementNode(condition=cond, then_block=then_b, else_block=else_b):
            node.condition = _replace_in_node(node.condition, mapping)
            # blocks will be processed top-level; keep structure
            return node
        case WhileStatementNode(condition=cond, body=body):
            node.condition = _replace_in_node(node.condition, mapping)
            return node
        case ReturnStatementNode(expression=expr) if expr is not None:
            node.expression = _replace_in_node(node.expression, mapping)
            return node
        case VariableDeclarationNode(var_name=vn, init_value=init) if init is not None:
            node.init_value = _replace_in_node(node.init_value, mapping)
            return node
        case BlockNode(statements=stmts):
            node.statements = [_replace_in_node(s, mapping) for s in stmts]
            return node
        case ProgramNode(statements=stmts):
            node.statements = [_replace_in_node(s, mapping) for s in stmts]
            return node
        case _:
            return node


def copy_propagation_cfg(cfg: Dict[str, any]) -> Dict[str, any]:
    """Perform conservative copy propagation on the CFG in-place and return it.

    cfg: dict with keys 'blocks' (list of blocks with 'label' and 'statements')
    This modifies the AST nodes in the blocks to replace identifier uses
    with their propagated originals where safe.
    """
    blocks = {b["label"]: b for b in cfg.get("blocks", [])}
    preds = _compute_preds(cfg)

    # Initialize in/out maps
    in_map: Dict[str, Dict[str, str]] = {lbl: {} for lbl in blocks}
    out_map: Dict[str, Dict[str, str]] = {lbl: {} for lbl in blocks}

    changed = True
    # Iterative forward dataflow
    while changed:
        changed = False
        for lbl, block in blocks.items():
            # compute in as intersection of predecessors' outs
            pred_maps = [out_map[p] for p in preds.get(lbl, [])]
            if pred_maps:
                new_in = _merge_maps(pred_maps)
            else:
                new_in = {}
            if new_in != in_map[lbl]:
                in_map[lbl] = new_in
                changed = True
            # simulate block
            cur = dict(in_map[lbl])
            for stmt in block.get("statements", []):
                # handle assignment x = y and var decl patterns using structural matching
                match stmt:
                    case AssignmentNode(
                        left=IdentifierNode(name=lhs), right=IdentifierNode(name=rhs)
                    ):
                        canonical = cur.get(rhs, rhs)
                        cur[lhs] = canonical
                    case AssignmentNode(left=IdentifierNode(name=lhs), right=_):
                        # non-copy assignment kills lhs
                        if lhs in cur:
                            del cur[lhs]
                    case AssignmentNode():
                        # assignment to non-simple LHS conservative: do nothing
                        pass
                    case VariableDeclarationNode(
                        var_name=vn, init_value=IdentifierNode(name=iname)
                    ):
                        cur[vn] = cur.get(iname, iname)
                    case VariableDeclarationNode(var_name=vn, init_value=None):
                        if vn in cur:
                            del cur[vn]
                    case VariableDeclarationNode(var_name=vn):
                        # initializer exists but not a simple identifier: kill mapping
                        if vn in cur:
                            del cur[vn]
                    case _:
                        # For other nodes we don't change the mapping
                        pass
            if cur != out_map[lbl]:
                out_map[lbl] = cur
                changed = True

    # Now rewrite uses using computed in_map and walking statements
    for lbl, block in blocks.items():
        mapping = dict(in_map[lbl])
        new_stmts: List[ASTNode] = []
        for stmt in block.get("statements", []):
            # First, rewrite uses inside the stmt according to current mapping
            new_stmt = _replace_in_node(stmt, mapping)
            # Then update mapping as we encounter defs in this stmt using matching
            match new_stmt:
                case AssignmentNode(
                    left=IdentifierNode(name=lhs), right=IdentifierNode(name=rhs)
                ):
                    mapping[lhs] = mapping.get(rhs, rhs)
                case AssignmentNode(left=IdentifierNode(name=lhs), right=_):
                    mapping.pop(lhs, None)
                case VariableDeclarationNode(
                    var_name=vn, init_value=IdentifierNode(name=iname)
                ):
                    mapping[vn] = mapping.get(iname, iname)
                case VariableDeclarationNode(var_name=vn):
                    mapping.pop(vn, None)
                case _:
                    pass

            new_stmts.append(new_stmt)
        block["statements"] = new_stmts

    # Run a small peephole pass to eliminate tmp-call patterns produced by
    # the flattener which are safe to collapse and lead to redundant
    # MV/ADDI sequences after lowering.
    try:
        cfg = peephole_tmp_call_elim(cfg)
    except Exception:
        pass

    try:
        cfg = _collapse_vardecl_then_return(cfg)
    except Exception:
        pass
    try:
        cfg = dead_code_elim(cfg, level="conservative")
    except Exception:
        pass

    return cfg


def peephole_tmp_call_elim(cfg: Dict[str, any]) -> Dict[str, any]:
    """Simple peephole to eliminate patterns produced by the flattener:

    `_tmp = call(...); var x = _tmp; return x;`  -> `return call(...);`
    `_tmp = call(...); var x = _tmp;` -> `var x = call(...);`

    This removes an unnecessary temporary that would otherwise become an
    MV/ADDI sequence in lowered instructions.
    """
    for b in cfg.get("blocks", []):
        stmts = b.get("statements", [])
        i = 0
        new_stmts: List[ASTNode] = []
        while i < len(stmts):
            s = stmts[i]
            # match assignment of temp from call
            match s:
                case AssignmentNode(
                    left=IdentifierNode(name=tmpname), right=FunctionCallNode() as fc
                ) if tmpname.startswith("_tmp"):
                    # lookahead for var decl that initializes from tmp
                    if i + 1 < len(stmts):
                        next_s = stmts[i + 1]
                        match next_s:
                            case VariableDeclarationNode(
                                var_name=vn, init_value=IdentifierNode(name=iname)
                            ) if (iname == tmpname):
                                # check for return after that
                                if i + 2 < len(stmts):
                                    third = stmts[i + 2]
                                    match third:
                                        case ReturnStatementNode(
                                            expression=IdentifierNode(name=rname)
                                        ) if (rname == vn or rname == tmpname):
                                            new_stmts.append(
                                                ReturnStatementNode(expression=fc)
                                            )
                                            i += 3
                                            continue
                                # replace with vardecl initialized by call
                                new_stmts.append(
                                    VariableDeclarationNode(
                                        var_name=vn,
                                        var_type=next_s.var_type,
                                        init_value=fc,
                                    )
                                )
                                i += 2
                                continue
                case _:
                    pass
            new_stmts.append(s)
            i += 1
        b["statements"] = new_stmts
    return cfg


def _collapse_vardecl_then_return(cfg: Dict[str, any]) -> Dict[str, any]:
    """Collapse `var x = call(...); return x;` into `return call(...);`.
    This handles the case where earlier passes already transformed the
    assignment into a vardecl with a call initializer.
    """
    for b in cfg.get("blocks", []):
        stmts = b.get("statements", [])
        new: List[ASTNode] = []
        i = 0
        while i < len(stmts):
            s = stmts[i]
            match s:
                case VariableDeclarationNode(
                    init_value=FunctionCallNode() as fc, var_name=vn
                ) if (
                    i + 1 < len(stmts)
                    and isinstance(stmts[i + 1], ReturnStatementNode)
                    and isinstance(stmts[i + 1].expression, IdentifierNode)
                    and stmts[i + 1].expression.name == vn
                ):
                    new.append(ReturnStatementNode(expression=fc))
                    i += 2
                    continue
                case _:
                    new.append(s)
                    i += 1
        b["statements"] = new
    return cfg


def dead_code_elim(cfg: Dict[str, any], level: str = "aggressive") -> Dict[str, any]:
    """Dead code elimination at CFG level.

    level: aggressiveness of elimination. Supported values:
      - "conservative": only remove temporaries produced by the flattener
        (names starting with `_tmp`) and only when the RHS has no side-effects.
      - "aggressive": allow removing any dead definition (subject to RHS
        having no side-effects). Use with care.

    Removes assignments/var-decls whose defined variable is not live after
    the statement and whose RHS has no side-effects (function calls).
    """
    try:
        liv = analyze_liveness(cfg)
    except Exception:
        return cfg

    def _has_call(n: ASTNode) -> bool:
        if n is None:
            return False
        if isinstance(n, FunctionCallNode):
            return True
        # recurse
        for attr in getattr(n, "__dict__", {}).values():
            if isinstance(attr, ASTNode) and _has_call(attr):
                return True
            if isinstance(attr, list):
                for x in attr:
                    if isinstance(x, ASTNode) and _has_call(x):
                        return True
        return False

    for b in cfg.get("blocks", []):
        lbl = b["label"]
        blk_liv = liv.get(lbl, {}).get("instr_liveness", [])
        new_stmts: List[ASTNode] = []
        stmts_list = b.get("statements", [])
        for i, stmt in enumerate(stmts_list):
            # Determine live_out for this statement
            if i < len(blk_liv):
                _, live_out = blk_liv[i]
            else:
                live_out = set()

            removable = False
            # Determine removability based on level and side-effects.
            if isinstance(stmt, AssignmentNode) and isinstance(
                stmt.left, IdentifierNode
            ):
                lhs = stmt.left.name
                rhs_node = stmt.right
                # If RHS is an identifier that refers to a temp defined earlier,
                # inspect its defining statement for side-effects.
                if isinstance(rhs_node, IdentifierNode) and rhs_node.name.startswith(
                    "_tmp"
                ):
                    # search backwards for its definition in this block
                    def_rhs = None
                    for j in range(i - 1, -1, -1):
                        s2 = stmts_list[j]
                        if (
                            isinstance(s2, AssignmentNode)
                            and isinstance(s2.left, IdentifierNode)
                            and s2.left.name == rhs_node.name
                        ):
                            def_rhs = s2.right
                            break
                    if def_rhs is not None:
                        rhs_node = def_rhs

                if not _has_call(rhs_node):
                    if level == "aggressive":
                        if lhs not in live_out:
                            removable = True
                    else:
                        # conservative: only remove temporaries introduced by flattening
                        if lhs.startswith("_tmp") and lhs not in live_out:
                            removable = True
            # Variable declaration with init: similar policy as assignments
            elif (
                isinstance(stmt, VariableDeclarationNode)
                and stmt.init_value is not None
            ):
                vn = stmt.var_name
                init_node = stmt.init_value
                # If initializer is an identifier pointing to a temp, look for its definition
                if isinstance(init_node, IdentifierNode) and init_node.name.startswith(
                    "_tmp"
                ):
                    def_rhs = None
                    for j in range(i - 1, -1, -1):
                        s2 = stmts_list[j]
                        if (
                            isinstance(s2, AssignmentNode)
                            and isinstance(s2.left, IdentifierNode)
                            and s2.left.name == init_node.name
                        ):
                            def_rhs = s2.right
                            break
                    if def_rhs is not None:
                        init_node = def_rhs

                if not _has_call(init_node):
                    if level == "aggressive":
                        if vn not in live_out:
                            removable = True
                    else:
                        if vn.startswith("_tmp") and vn not in live_out:
                            removable = True

            if removable:
                # skip stmt
                continue
            new_stmts.append(stmt)
        b["statements"] = new_stmts
    return cfg
