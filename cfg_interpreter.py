"""Interpreter that executes the CFG produced by `cfg.build_cfg`.

This interpreter is primarily for testing: it executes blocks following
control-flow edges and evaluates AST expressions/statements contained in
blocks. It supports functions by locating the FunctionDeclarationNode that
build_cfg places in function entry blocks.

The interpreter returns a globals environment dict with variable names
mapped to values and `__functions__` mapping function names to their
FunctionDeclarationNode for inspection.
"""

from typing import Any, Dict, Optional
from ast_nodes import *
from ast_interpreter import ReturnException


class CFGReturn(Exception):
    def __init__(self, value: Any):
        self.value = value


def interpret_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    blocks = cfg.get("blocks", [])
    # Build label->block map
    block_map: Dict[str, Dict] = {b["label"]: b for b in blocks}

    # Discover functions: function entry blocks contain a FunctionDeclarationNode
    functions: Dict[str, FunctionDeclarationNode] = {}
    func_entries: Dict[str, str] = {}
    for b in blocks:
        stmts = b.get("statements", [])
        if stmts and isinstance(stmts[0], FunctionDeclarationNode):
            fdecl = stmts[0]
            functions[fdecl.func_name] = fdecl
            func_entries[fdecl.func_name] = b["label"]

    globals_env: Dict[str, Any] = {}

    def eval_expr(node: ASTNode, local_env: Dict[str, Any]) -> Any:
        match node:
            case IntLiteralNode(value=v):
                return v
            case BoolLiteralNode(value=v):
                return bool(v)
            case IdentifierNode(name=n):
                if n in local_env:
                    return local_env[n]
                return globals_env.get(n)
            case BinaryOpNode(left=l, operator=op, right=r):
                lv = eval_expr(l, local_env)
                rv = eval_expr(r, local_env)
                match op:
                    case "+":
                        return lv + rv
                    case "-":
                        return lv - rv
                    case "*":
                        return lv * rv
                    case "/":
                        return (
                            lv // rv
                            if isinstance(lv, int) and isinstance(rv, int)
                            else lv / rv
                        )
                    case "%":
                        return lv % rv
                    case "==":
                        return lv == rv
                    case "<":
                        return lv < rv
                    case ">":
                        return lv > rv
                    case "<=":
                        return lv <= rv
                    case ">=":
                        return lv >= rv
                    case _:
                        raise RuntimeError(f"Unsupported binary op: {op}")
            case UnaryOpNode(operator=op, right=right):
                v = eval_expr(right, local_env)
                match op:
                    case "-":
                        return -v
                    case "!":
                        return not v
                    case _:
                        raise RuntimeError(f"Unsupported unary op: {op}")
            case ArrayIndexNode(array=arr, index=idx):
                arrv = eval_expr(arr, local_env)
                idxv = eval_expr(idx, local_env)
                return arrv[idxv]
            case FunctionCallNode(function=IdentifierNode(name=fname), arguments=args):
                evaled = [eval_expr(a, local_env) for a in args]
                if fname not in functions:
                    raise RuntimeError(f"Unknown function: {fname}")
                fdecl = functions[fname]
                entry_label = func_entries[fname]
                local = {}
                if fdecl.arg_names:
                    for name, val in zip(fdecl.arg_names, evaled):
                        local[name] = val
                try:
                    exec_block(entry_label, local)
                except CFGReturn as r:
                    return r.value
                return None
            case AssignmentNode(left=IdentifierNode(name=n), right=right):
                rhs = eval_expr(right, local_env)
                local_env[n] = rhs
                return rhs
            case AssignmentNode(left=ArrayIndexNode(array=arr, index=idx), right=right):
                rhs = eval_expr(right, local_env)
                arrv = eval_expr(arr, local_env)
                idxv = eval_expr(idx, local_env)
                arrv[idxv] = rhs
                return rhs
            case _:
                raise RuntimeError(
                    f"Unhandled expr node type in CFG interpreter: {node}"
                )

    def exec_stmt(stmt: ASTNode, local_env: Dict[str, Any]):
        match stmt:
            case VariableDeclarationNode(var_name=name, init_value=init):
                if init is not None:
                    local_env[name] = eval_expr(init, local_env)
                else:
                    local_env[name] = None
                return None
            case AssignmentNode(left=IdentifierNode(name=n), right=right):
                rhs = eval_expr(right, local_env)
                local_env[n] = rhs
                return None
            case AssignmentNode(left=ArrayIndexNode(array=arr, index=idx), right=right):
                rhs = eval_expr(right, local_env)
                arrv = eval_expr(arr, local_env)
                idxv = eval_expr(idx, local_env)
                arrv[idxv] = rhs
                return None
            case ExpressionStatementNode(expression=expr):
                eval_expr(expr, local_env)
                return None
            case ReturnStatementNode(expression=None):
                raise CFGReturn(None)
            case ReturnStatementNode(expression=expr):
                val = eval_expr(expr, local_env)
                raise CFGReturn(val)
            case IfStatementNode(
                condition=cond, then_block=then_block, else_block=else_block
            ):
                if eval_expr(cond, local_env):
                    if isinstance(then_block, BlockNode):
                        exec_block_local(then_block, local_env)
                    else:
                        exec_stmt(then_block, local_env)
                elif else_block:
                    if isinstance(else_block, BlockNode):
                        exec_block_local(else_block, local_env)
                    else:
                        exec_stmt(else_block, local_env)
                return None
            case WhileStatementNode(condition=cond, body=body):
                while eval_expr(cond, local_env):
                    exec_block_local(body, local_env)
                return None
            case FunctionCallNode():
                # direct function call as statement
                eval_expr(stmt, local_env)
                return None
            case FunctionDeclarationNode():
                # ignore
                return None
            case BlockNode():
                exec_block_local(stmt, local_env)
                return None
            case _:
                raise RuntimeError(f"Unhandled statement in CFG interpreter: {stmt}")

    def exec_block_local(block_node: BlockNode, local_env: Dict[str, Any]):
        for s in block_node.statements:
            exec_stmt(s, local_env)

    def exec_block(start_label: str, local_env: Dict[str, Any] = None):
        # local_env defaults to globals_env for top-level execution
        if local_env is None:
            local_env = globals_env
        cur = start_label
        visited = set()
        while True:
            if cur not in block_map:
                break
            blk = block_map[cur]
            stmts = blk.get("statements", [])
            # execute statements in the block
            for s in stmts:
                # Skip function-decl header when starting a function
                if isinstance(s, FunctionDeclarationNode):
                    continue
                exec_stmt(s, local_env)
            # Determine next block
            if not stmts:
                # fall-through if single successor
                if len(blk.get("out_edges", [])) == 1:
                    cur = blk["out_edges"][0]
                    continue
                else:
                    break
            last = stmts[-1]
            if isinstance(last, IfStatementNode):
                cond = eval_expr(last.condition, local_env)
                outs = blk.get("out_edges", [])
                if len(outs) < 2:
                    raise RuntimeError("Malformed CFG: if without two targets")
                cur = outs[0] if cond else outs[1]
                continue
            if isinstance(last, WhileStatementNode):
                cond = eval_expr(last.condition, local_env)
                outs = blk.get("out_edges", [])
                if len(outs) < 2:
                    raise RuntimeError("Malformed CFG: while without two targets")
                cur = outs[0] if cond else outs[1]
                continue
            outs = blk.get("out_edges", [])
            if len(outs) == 1:
                cur = outs[0]
                continue
            # no successors -> end
            break

    # Determine sensible start block. Some CFGs produced by `build_cfg`
    # create an initially-empty `entry` block and place real global code
    # in subsequent blocks. If the entry block is empty, prefer the first
    # non-function block that contains statements.
    start = cfg.get("entry")
    entry_blk = block_map.get(start)
    if entry_blk is not None and not entry_blk.get("statements"):
        # find first non-function block with statements
        for b in blocks:
            if not b.get("function") and b.get("statements"):
                start = b["label"]
                break

    # Execute top-level by starting at resolved start label
    exec_block(start)

    # If a `main` function exists, execute it and capture its return value
    # under `__return__` in the globals env for test convenience.
    if "main" in functions:
        entry = func_entries.get("main")
        if entry:
            try:
                exec_block(entry, {})
            except CFGReturn as r:
                globals_env["__return__"] = r.value
            else:
                globals_env.setdefault("__return__", None)

    globals_env["__functions__"] = functions
    return globals_env
