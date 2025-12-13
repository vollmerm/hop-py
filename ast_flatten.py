"""AST flattener for three-address code preparation.

This module provides a function to transform an AST so that all expressions
are flattened: every binary operation, function call, or return has only
variables or literals as operands. Nested expressions are extracted into
assignments to fresh temporary variables (tmp1, tmp2, ...).

Example:
    (x + y) + z  ==>  tmp1 = x + y; tmp1 + z
    return f(a + b, c * d)  ==>  tmp1 = a + b; tmp2 = c * d; return f(tmp1, tmp2)

Intended for use before generating three-address code or a control flow graph.
"""

from ast_nodes import *
from typing import List, Tuple


class TempVarGenerator:
    def __init__(self):
        self.counter = 0

    def new(self) -> str:
        self.counter += 1
        return f"_tmp{self.counter}"


def flatten_block(block: BlockNode, tempgen: TempVarGenerator) -> BlockNode:
    stmts: List[ASTNode] = []
    for stmt in block.statements:
        flat_stmts = flatten_statement(stmt, tempgen)
        if isinstance(flat_stmts, list):
            stmts.extend(flat_stmts)
        else:
            stmts.append(flat_stmts)
    return BlockNode(statements=stmts)


def flatten_lhs_expr(expr: ASTNode, tempgen: TempVarGenerator) -> Tuple[List[ASTNode], ASTNode]:
    """Flatten an expression when it's used as an lvalue (left-hand side).

    Unlike `flatten_expr`, this preserves `ArrayIndex` as an lvalue node
    (with flattened children) instead of turning it into a temporary that
    reads from the array.
    """
    if expr.type == NodeType.ARRAY_INDEX:
        stmts_arr, arr = flatten_expr(expr.array, tempgen)
        stmts_idx, idx = flatten_expr(expr.index, tempgen)
        return stmts_arr + stmts_idx, ArrayIndexNode(array=arr, index=idx)
    # Fallback: for identifiers and other lvalues, reuse flatten_expr
    return flatten_expr(expr, tempgen)


def flatten_statement(stmt: ASTNode, tempgen: TempVarGenerator):
    t = stmt.type
    if t == NodeType.BLOCK:
        return flatten_block(stmt, tempgen)
    elif t == NodeType.EXPR_STMT:
        flat_stmts, expr = flatten_expr(stmt.expression, tempgen)
        return flat_stmts + [ExpressionStatementNode(expression=expr)]
    elif t == NodeType.ASSIGNMENT:
        # For the left-hand side of an assignment we must preserve lvalue
        # shape (e.g. ArrayIndex) rather than converting it into a temporary
        # used for rvalue expressions. Use the globally-defined lhs flattener.
        flat_stmts_left, left = flatten_lhs_expr(stmt.left, tempgen)
        flat_stmts_right, right = flatten_expr(stmt.right, tempgen)
        assign = AssignmentNode(left=left, right=right)
        return flat_stmts_left + flat_stmts_right + [assign]
    elif t == NodeType.RETURN_STMT:
        if stmt.expression is None:
            return [stmt]
        flat_stmts, expr = flatten_expr(stmt.expression, tempgen)
        return flat_stmts + [ReturnStatementNode(expression=expr)]
    elif t == NodeType.IF_STMT:
        flat_stmts_cond, cond = flatten_expr(stmt.condition, tempgen)
        then_block = flatten_statement(stmt.then_block, tempgen)
        else_block = (
            flatten_statement(stmt.else_block, tempgen) if stmt.else_block else None
        )
        if isinstance(then_block, list):
            then_block = BlockNode(statements=then_block)
        if else_block and isinstance(else_block, list):
            else_block = BlockNode(statements=else_block)
        return flat_stmts_cond + [
            IfStatementNode(
                condition=cond, then_block=then_block, else_block=else_block
            )
        ]
    elif t == NodeType.WHILE_STMT:
        flat_stmts_cond, cond = flatten_expr(stmt.condition, tempgen)
        body = flatten_statement(stmt.body, tempgen)
        if isinstance(body, list):
            body = BlockNode(statements=body)
        return flat_stmts_cond + [WhileStatementNode(condition=cond, body=body)]
    elif t == NodeType.VAR_DECL:
        if stmt.init_value:
            flat_stmts, expr = flatten_expr(stmt.init_value, tempgen)
            decl = VariableDeclarationNode(
                var_name=stmt.var_name, var_type=stmt.var_type, init_value=expr
            )
            return flat_stmts + [decl]
        else:
            return [stmt]
    elif t == NodeType.FUNC_DECL:
        if stmt.body:
            new_body = flatten_block(stmt.body, tempgen)
            return [
                FunctionDeclarationNode(
                    func_name=stmt.func_name,
                    return_type=stmt.return_type,
                    arg_types=stmt.arg_types,
                    arg_names=stmt.arg_names,
                    body=new_body,
                )
            ]
        else:
            return [stmt]
    else:
        return [stmt]


def flatten_expr(
    expr: ASTNode, tempgen: TempVarGenerator
) -> Tuple[List[ASTNode], ASTNode]:
    t = expr.type
    # Base cases: literals and identifiers
    if t in (NodeType.INT_LITERAL, NodeType.BOOL_LITERAL, NodeType.IDENTIFIER):
        return [], expr
    # Binary op: flatten both sides, assign to temp if needed
    elif t == NodeType.BINARY_OP:
        stmts_left, left = flatten_expr(expr.left, tempgen)
        stmts_right, right = flatten_expr(expr.right, tempgen)
        tmp_name = tempgen.new()
        tmp_var = IdentifierNode(name=tmp_name, symbol_type=None)
        assign = AssignmentNode(
            left=tmp_var,
            right=BinaryOpNode(left=left, operator=expr.operator, right=right),
        )
        return stmts_left + stmts_right + [assign], tmp_var
    # Unary op: flatten right, assign to temp
    elif t == NodeType.UNARY_OP:
        stmts_right, right = flatten_expr(expr.right, tempgen)
        tmp_name = tempgen.new()
        tmp_var = IdentifierNode(name=tmp_name, symbol_type=None)
        assign = AssignmentNode(
            left=tmp_var, right=UnaryOpNode(operator=expr.operator, right=right)
        )
        return stmts_right + [assign], tmp_var
    # Array index: flatten array and index, assign to temp
    elif t == NodeType.ARRAY_INDEX:
        stmts_arr, arr = flatten_expr(expr.array, tempgen)
        stmts_idx, idx = flatten_expr(expr.index, tempgen)
        tmp_name = tempgen.new()
        tmp_var = IdentifierNode(name=tmp_name, symbol_type=None)
        assign = AssignmentNode(
            left=tmp_var, right=ArrayIndexNode(array=arr, index=idx)
        )
        return stmts_arr + stmts_idx + [assign], tmp_var
    # Function call: flatten all arguments, assign to temp
    elif t == NodeType.FUNC_CALL:
        stmts = []
        flat_args = []
        for arg in expr.arguments:
            s, a = flatten_expr(arg, tempgen)
            stmts.extend(s)
            flat_args.append(a)
        tmp_name = tempgen.new()
        tmp_var = IdentifierNode(name=tmp_name, symbol_type=None)
        assign = AssignmentNode(
            left=tmp_var,
            right=FunctionCallNode(function=expr.function, arguments=flat_args),
        )
        return stmts + [assign], tmp_var
    # Assignment: flatten both sides, assign to temp
    elif t == NodeType.ASSIGNMENT:
        # When flattening an assignment used as an expression, preserve
        # lvalue shape for left-hand side (so stores like `arr[i] = v` are
        # represented correctly). Use `flatten_lhs_expr` to avoid turning
        # ArrayIndex lvalues into temporaries.
        stmts_left, left = flatten_lhs_expr(expr.left, tempgen)
        stmts_right, right = flatten_expr(expr.right, tempgen)
        tmp_name = tempgen.new()
        tmp_var = IdentifierNode(name=tmp_name, symbol_type=None)
        assign = AssignmentNode(left=left, right=right)
        assign_tmp = AssignmentNode(left=tmp_var, right=assign)
        return stmts_left + stmts_right + [assign, assign_tmp], tmp_var
    else:
        # For any other node, return as is
        return [], expr


def flatten_program(prog: ProgramNode) -> ProgramNode:
    tempgen = TempVarGenerator()
    stmts: List[ASTNode] = []
    for stmt in prog.statements:
        flat = flatten_statement(stmt, tempgen)
        if isinstance(flat, list):
            stmts.extend(flat)
        else:
            stmts.append(flat)
    return ProgramNode(statements=stmts)
