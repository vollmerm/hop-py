"""Small interpreter for ASTs used by tests.

This interpreter is intentionally simple and intended for test harnesses.
It supports `ProgramNode`, `VariableDeclarationNode`, `AssignmentNode`,
`ExpressionStatementNode`, `ReturnStatementNode`, `IfStatementNode`,
`WhileStatementNode`, `FunctionDeclarationNode`, `FunctionCallNode`, and
basic expressions: `BinaryOpNode`, `UnaryOpNode`, `IntLiteralNode`,
`BoolLiteralNode`, `IdentifierNode`, and `ArrayIndexNode` (simple arrays).
"""

from typing import Any, Dict, Optional
from ast_nodes import *


class ReturnException(Exception):
    def __init__(self, value: Any):
        self.value = value


def _eval_expr(
    node: ASTNode, env: Dict[str, Any], functions: Dict[str, FunctionDeclarationNode]
) -> Any:
    match node:
        case IntLiteralNode(value=v):
            return v
        case BoolLiteralNode(value=v):
            return bool(v)
        case IdentifierNode(name=n):
            return env.get(n)
        case BinaryOpNode(left=l, operator=op, right=r):
            lv = _eval_expr(l, env, functions)
            rv = _eval_expr(r, env, functions)
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
                    raise RuntimeError(f"Unsupported binary operator: {op}")
        case UnaryOpNode(operator=op, right=right):
            val = _eval_expr(right, env, functions)
            match op:
                case "-":
                    return -val
                case "!":
                    return not val
                case _:
                    raise RuntimeError(f"Unsupported unary operator: {op}")
        case ArrayIndexNode(array=arr, index=idx):
            arrv = _eval_expr(arr, env, functions)
            idxv = _eval_expr(idx, env, functions)
            return arrv[idxv]
        case FunctionCallNode(function=IdentifierNode(name=fname), arguments=args):
            evaled = [_eval_expr(a, env, functions) for a in args]
            if fname in functions:
                fdecl = functions[fname]
                local_env: Dict[str, Any] = {}
                for name, val in zip(fdecl.arg_names or [], evaled):
                    local_env[name] = val
                try:
                    _exec_block(fdecl.body, local_env, functions)
                except ReturnException as re:
                    return re.value
                return None
            raise RuntimeError(f"Unknown function: {fname}")
        case AssignmentNode(left=IdentifierNode(name=n), right=right):
            rhs = _eval_expr(right, env, functions)
            env[n] = rhs
            return rhs
        case AssignmentNode(left=ArrayIndexNode(array=arr, index=idx), right=right):
            rhs = _eval_expr(right, env, functions)
            arrv = _eval_expr(arr, env, functions)
            idxv = _eval_expr(idx, env, functions)
            arrv[idxv] = rhs
            return rhs
        case _:
            raise RuntimeError(f"Unhandled expression node type: {node}")


def _exec_stmt(
    stmt: ASTNode, env: Dict[str, Any], functions: Dict[str, FunctionDeclarationNode]
):
    match stmt:
        case VariableDeclarationNode(var_name=name, init_value=init):
            if init is not None:
                val = _eval_expr(init, env, functions)
                env[name] = val
            else:
                env[name] = None
            return None
        case AssignmentNode(left=IdentifierNode(name=n), right=right):
            rhs = _eval_expr(right, env, functions)
            env[n] = rhs
            return None
        case AssignmentNode(left=ArrayIndexNode(array=arr, index=idx), right=right):
            rhs = _eval_expr(right, env, functions)
            arrv = _eval_expr(arr, env, functions)
            idxv = _eval_expr(idx, env, functions)
            arrv[idxv] = rhs
            return None
        case ExpressionStatementNode(expression=expr):
            _eval_expr(expr, env, functions)
            return None
        case ReturnStatementNode(expression=None):
            raise ReturnException(None)
        case ReturnStatementNode(expression=expr):
            val = _eval_expr(expr, env, functions)
            raise ReturnException(val)
        case IfStatementNode(
            condition=cond, then_block=then_block, else_block=else_block
        ):
            if _eval_expr(cond, env, functions):
                _exec_block(then_block, env, functions)
            elif else_block:
                _exec_block(else_block, env, functions)
            return None
        case WhileStatementNode(condition=cond, body=body):
            while _eval_expr(cond, env, functions):
                _exec_block(body, env, functions)
            return None
        case BlockNode():
            _exec_block(stmt, env, functions)
            return None
        case FunctionDeclarationNode():
            # Function declarations handled at program level
            return None
        case ProgramNode(statements=stmts):
            for s in stmts:
                _exec_stmt(s, env, functions)
            return None
        case _:
            raise RuntimeError(f"Unhandled statement node: {stmt}")


def _exec_block(
    block: BlockNode, env: Dict[str, Any], functions: Dict[str, FunctionDeclarationNode]
):
    for s in block.statements:
        _exec_stmt(s, env, functions)


def interpret_program(prog: ProgramNode) -> Dict[str, Any]:
    """Interpret a (flattened) ProgramNode and return the global environment mapping.

    Functions are collected in the returned environment under key '__functions__'
    and also kept in the local functions dict for calling.
    """
    env: Dict[str, Any] = {}
    functions: Dict[str, FunctionDeclarationNode] = {}
    # Collect function declarations first
    for s in prog.statements:
        if isinstance(s, FunctionDeclarationNode):
            functions[s.func_name] = s
    # Execute top-level statements (skipping function decls)
    for s in prog.statements:
        if isinstance(s, FunctionDeclarationNode):
            continue
        try:
            _exec_stmt(s, env, functions)
        except ReturnException as re:
            # Top-level return: store under '__return__'
            env["__return__"] = re.value
            break
    env["__functions__"] = functions
    return env
