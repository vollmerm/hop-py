"""Liveness analysis for the CFG.

This module computes liveness information per basic block and per instruction
for the CFG structure used in this project. The CFG representation expected is
as produced by `cfg.build_cfg`:

  cfg = {
    'blocks': [ { 'label': str, 'statements': [ASTNode...], 'out_edges': [labels...] }, ... ],
    'entry': str,
    'exit': Optional[str]
  }

Output format:
  {
    block_label: {
      'live_in': set(str),         # live at block entry
      'live_out': set(str),        # live at block exit
      'instr_liveness': [         # list of (live_in, live_out) for each statement in order
          (set(str), set(str)),
      ]
    },
    ...
  }

The analysis is a standard backward dataflow (liveness) computed to fixed point
at block granularity, then refined to per-instruction liveness by scanning
instructions in reverse within each block.
"""

from typing import Dict, Set, List, Tuple, Any
from ast_nodes import *


def _expr_uses(expr: ASTNode) -> Set[str]:
    """Return set of identifier names used by expression (conservative)."""
    uses: Set[str] = set()
    if expr is None:
        return uses
    t = expr.type
    if t == NodeType.IDENTIFIER:
        if isinstance(expr, IdentifierNode):
            uses.add(expr.name)
    elif t in (NodeType.INT_LITERAL, NodeType.BOOL_LITERAL):
        return uses
    elif t == NodeType.BINARY_OP:
        if isinstance(expr, BinaryOpNode):
            uses |= _expr_uses(expr.left)
            uses |= _expr_uses(expr.right)
    elif t == NodeType.UNARY_OP:
        if isinstance(expr, UnaryOpNode):
            uses |= _expr_uses(expr.right)
    elif t == NodeType.ARRAY_INDEX:
        if isinstance(expr, ArrayIndexNode):
            uses |= _expr_uses(expr.array)
            uses |= _expr_uses(expr.index)
    elif t == NodeType.FUNC_CALL:
        if isinstance(expr, FunctionCallNode):
            # function expression may be identifier
            uses |= _expr_uses(expr.function)
            for a in expr.arguments:
                uses |= _expr_uses(a)
    elif t == NodeType.ASSIGNMENT:
        if isinstance(expr, AssignmentNode):
            uses |= _expr_uses(expr.left)
            uses |= _expr_uses(expr.right)
    else:
        # Conservative walk over attributes
        for attr in getattr(expr, "__dict__", {}).values():
            if isinstance(attr, ASTNode):
                uses |= _expr_uses(attr)
            elif isinstance(attr, list):
                for x in attr:
                    if isinstance(x, ASTNode):
                        uses |= _expr_uses(x)
    return uses


def _stmt_defs(stmt: ASTNode) -> Set[str]:
    """Return the names defined (assigned) by a statement (conservative)."""
    defs: Set[str] = set()
    if stmt is None:
        return defs
    t = stmt.type
    if t == NodeType.ASSIGNMENT and isinstance(stmt, AssignmentNode):
        left = stmt.left
        if isinstance(left, IdentifierNode):
            defs.add(left.name)
        elif isinstance(left, ArrayIndexNode):
            # array write: conservatively treat array variable as defined
            if isinstance(left.array, IdentifierNode):
                defs.add(left.array.name)
    elif t == NodeType.VAR_DECL and isinstance(stmt, VariableDeclarationNode):
        defs.add(stmt.var_name)
    elif t == NodeType.EXPR_STMT and isinstance(stmt, ExpressionStatementNode):
        # Some expression statements can define temps via assignments; handle nested assignment
        if stmt.expression and stmt.expression.type == NodeType.ASSIGNMENT:
            defs |= _stmt_defs(stmt.expression)
    elif t == NodeType.RETURN_STMT:
        # return does not define local variables
        pass
    # Other statements (if/while) do not directly define at statement level
    return defs


def _stmt_uses(stmt: ASTNode) -> Set[str]:
    """Return the set of variable names used by a statement (conservative)."""
    uses: Set[str] = set()
    if stmt is None:
        return uses
    t = stmt.type
    if t == NodeType.ASSIGNMENT and isinstance(stmt, AssignmentNode):
        uses |= _expr_uses(stmt.right)
        # left may contain uses (array index)
        if isinstance(stmt.left, ArrayIndexNode):
            uses |= _expr_uses(stmt.left.array)
            uses |= _expr_uses(stmt.left.index)
        elif isinstance(stmt.left, IdentifierNode):
            # assigning to plain identifier does not use it (except in some languages), skip
            pass
    elif t == NodeType.VAR_DECL and isinstance(stmt, VariableDeclarationNode):
        if stmt.init_value:
            uses |= _expr_uses(stmt.init_value)
    elif t == NodeType.EXPR_STMT and isinstance(stmt, ExpressionStatementNode):
        uses |= _expr_uses(stmt.expression)
    elif t == NodeType.RETURN_STMT and isinstance(stmt, ReturnStatementNode):
        if stmt.expression:
            uses |= _expr_uses(stmt.expression)
    elif t == NodeType.IF_STMT and isinstance(stmt, IfStatementNode):
        uses |= _expr_uses(stmt.condition)
    elif t == NodeType.WHILE_STMT and isinstance(stmt, WhileStatementNode):
        uses |= _expr_uses(stmt.condition)
    elif t == NodeType.FUNC_DECL and isinstance(stmt, FunctionDeclarationNode):
        # function declaration: uses inside body
        pass
    else:
        # Conservative walk
        for attr in getattr(stmt, "__dict__", {}).values():
            if isinstance(attr, ASTNode):
                uses |= _expr_uses(attr)
            elif isinstance(attr, list):
                for x in attr:
                    if isinstance(x, ASTNode):
                        uses |= _expr_uses(x)
    return uses


def analyze_liveness(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Perform liveness analysis on the CFG and return per-block and per-instruction liveness.

    Returns a mapping from block label to a dict containing:
      - 'live_in': set of variables live at block entry
      - 'live_out': set of variables live at block exit
      - 'instr_liveness': list of (live_in, live_out) for each statement in order
    """
    # Map label -> block dict
    blocks_by_label = {b["label"]: b for b in cfg.get("blocks", [])}
    # Successors map
    succs = {b["label"]: list(b.get("out_edges", [])) for b in cfg.get("blocks", [])}

    # Initialize empty live_in/live_out
    analysis: Dict[str, Dict[str, Any]] = {}
    for lbl, b in blocks_by_label.items():
        analysis[lbl] = {
            "live_in": set(),
            "live_out": set(),
            "instr_liveness": [(set(), set()) for _ in b.get("statements", [])],
        }

    changed = True
    iteration = 0
    while changed:
        iteration += 1
        changed = False
        # Process blocks in reverse order of appearance for typically faster convergence
        for b in reversed(cfg.get("blocks", [])):
            lbl = b["label"]
            old_in = analysis[lbl]["live_in"].copy()
            old_out = analysis[lbl]["live_out"].copy()
            # live_out of block = union of live_in of successors
            new_out: Set[str] = set()
            for s in succs.get(lbl, []):
                new_out |= analysis[s]["live_in"]
            # Now compute per-instruction liveness by walking instructions backwards starting from new_out
            instrs = b.get("statements", [])
            live_after = new_out.copy()
            instr_liv: List[Tuple[Set[str], Set[str]]] = []
            # reverse iterate
            for stmt in reversed(instrs):
                uses = _stmt_uses(stmt)
                defs = _stmt_defs(stmt)
                live_in = (live_after - defs) | uses
                live_out = live_after.copy()
                instr_liv.append((live_in, live_out))
                live_after = live_in
            # instr_liv is reversed (last to first), reverse it back
            instr_liv.reverse()
            analysis[lbl]["instr_liveness"] = instr_liv
            new_in = instr_liv[0][0] if instr_liv else new_out.copy()
            # Update
            analysis[lbl]["live_out"] = new_out
            analysis[lbl]["live_in"] = new_in
            if new_in != old_in or new_out != old_out:
                changed = True
    return analysis


if __name__ == "__main__":
    # quick smoke test if run directly
    from cfg import build_cfg
    from ast_flatten import flatten_program

    # Build trivial program AST
    a = BlockNode(
        statements=[
            AssignmentNode(
                left=IdentifierNode(name="x", symbol_type=None),
                right=IntLiteralNode(value=1),
            ),
            AssignmentNode(
                left=IdentifierNode(name="y", symbol_type=None),
                right=BinaryOpNode(
                    left=IdentifierNode(name="x", symbol_type=None),
                    operator="+",
                    right=IntLiteralNode(value=2),
                ),
            ),
            ReturnStatementNode(expression=IdentifierNode(name="y", symbol_type=None)),
        ]
    )
    cfg = build_cfg(a)
    analysis = analyze_liveness(cfg)
    for lbl, info in analysis.items():
        print(lbl, info)
