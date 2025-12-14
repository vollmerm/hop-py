"""Unit tests for ast_flatten.py: checks that expressions are flattened and temporaries are introduced correctly."""

import pytest
from ast_nodes import *
from ast_flatten import flatten_program


def make_add(x, y):
    return BinaryOpNode(
        left=IdentifierNode(name=x, symbol_type=None),
        operator="+",
        right=IdentifierNode(name=y, symbol_type=None),
    )


def test_flatten_simple_add():
    prog = ProgramNode(
        statements=[ExpressionStatementNode(expression=make_add("a", "b"))]
    )
    flat = flatten_program(prog)
    # Both operands are simple identifiers, so no temporary is necessary.
    # The expression statement should remain a BinaryOp expression.
    assert len(flat.statements) == 1
    expr_stmt = flat.statements[0]
    assert isinstance(expr_stmt, ExpressionStatementNode)
    assert isinstance(expr_stmt.expression, BinaryOpNode)
    assert expr_stmt.expression.operator == "+"
    assert expr_stmt.expression.left.name == "a"
    assert expr_stmt.expression.right.name == "b"


def test_flatten_nested_add():
    # (a + b) + c
    inner = make_add("a", "b")
    outer = BinaryOpNode(
        left=inner, operator="+", right=IdentifierNode(name="c", symbol_type=None)
    )
    prog = ProgramNode(statements=[ExpressionStatementNode(expression=outer)])
    flat = flatten_program(prog)
    # Inner (a+b) is already simple, so only a single temporary for the
    # outer expression is produced: `_tmp = (a + b) + c`.
    assert len(flat.statements) == 1
    assign = flat.statements[0]
    assert isinstance(assign, AssignmentNode)
    assert assign.left.name.startswith("_tmp")
    assert isinstance(assign.right, BinaryOpNode)
    # The right-hand side should be (a + b) + c
    inner_bin = assign.right.left
    assert isinstance(inner_bin, BinaryOpNode)
    assert inner_bin.left.name == "a"
    assert inner_bin.right.name == "b"
    assert assign.right.right.name == "c"


def test_flatten_func_call():
    # f(a + b, c)
    call = FunctionCallNode(
        function=IdentifierNode(name="f", symbol_type=None),
        arguments=[make_add("a", "b"), IdentifierNode(name="c", symbol_type=None)],
    )
    prog = ProgramNode(statements=[ExpressionStatementNode(expression=call)])
    flat = flatten_program(prog)
    # The inner argument (a + b) is simple, so it should be passed as a
    # BinaryOpNode directly to the function. The call's side-effects are
    # preserved by emitting an ExpressionStatement containing the call.
    assert len(flat.statements) == 1
    expr_stmt = flat.statements[0]
    assert isinstance(expr_stmt, ExpressionStatementNode)
    assert isinstance(expr_stmt.expression, FunctionCallNode)
    call_args = expr_stmt.expression.arguments
    assert isinstance(call_args[0], BinaryOpNode)
    assert call_args[0].left.name == "a"
    assert call_args[0].right.name == "b"
    assert call_args[1].name == "c"


def test_flatten_return():
    # return (a + b)
    ret = ReturnStatementNode(expression=make_add("a", "b"))
    prog = ProgramNode(statements=[ret])
    flat = flatten_program(prog)
    # With simple operands the binary expression is returned directly.
    assert len(flat.statements) == 1
    ret_stmt = flat.statements[0]
    assert isinstance(ret_stmt, ReturnStatementNode)
    assert isinstance(ret_stmt.expression, BinaryOpNode)
    assert ret_stmt.expression.left.name == "a"
    assert ret_stmt.expression.right.name == "b"


def test_flatten_if():
    # if (a + b) { return c; }
    cond = make_add("a", "b")
    then_block = BlockNode(
        statements=[
            ReturnStatementNode(expression=IdentifierNode(name="c", symbol_type=None))
        ]
    )
    if_stmt = IfStatementNode(condition=cond, then_block=then_block, else_block=None)
    prog = ProgramNode(statements=[if_stmt])
    flat = flatten_program(prog)
    # The condition (a + b) is simple; it should be embedded directly in
    # the IfStatement's condition.
    assert len(flat.statements) == 1
    if_flat = flat.statements[0]
    assert isinstance(if_flat, IfStatementNode)
    assert isinstance(if_flat.condition, BinaryOpNode)
    assert if_flat.condition.left.name == "a"
    assert if_flat.condition.right.name == "b"
    assert isinstance(if_flat.then_block, BlockNode)
    assert isinstance(if_flat.then_block.statements[0], ReturnStatementNode)


def test_flatten_assignment_expression_no_nested():
    # An expression-statement whose expression is an assignment: `result = sum`
    # After flattening we should get a single AssignmentNode `result = sum`
    # (no `_tmp = result = sum` or other nested assignment forms).
    assignment = AssignmentNode(
        left=IdentifierNode(name="result", symbol_type=None),
        right=IdentifierNode(name="sum", symbol_type=None),
    )
    prog = ProgramNode(statements=[ExpressionStatementNode(expression=assignment)])
    flat = flatten_program(prog)
    assert len(flat.statements) == 1
    a = flat.statements[0]
    assert isinstance(a, AssignmentNode)
    assert isinstance(a.left, IdentifierNode)
    assert isinstance(a.right, IdentifierNode)
    assert a.left.name == "result"
    assert a.right.name == "sum"
