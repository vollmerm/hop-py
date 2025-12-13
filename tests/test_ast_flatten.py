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
    # Should introduce a temp assignment and then use the temp in the expr stmt
    assert len(flat.statements) == 2
    assign = flat.statements[0]
    expr_stmt = flat.statements[1]
    assert isinstance(assign, AssignmentNode)
    assert isinstance(expr_stmt, ExpressionStatementNode)
    assert isinstance(expr_stmt.expression, IdentifierNode)
    assert assign.left.name.startswith("_tmp")
    assert assign.right.operator == "+"
    assert assign.right.left.name == "a"
    assert assign.right.right.name == "b"
    assert expr_stmt.expression.name == assign.left.name


def test_flatten_nested_add():
    # (a + b) + c
    inner = make_add("a", "b")
    outer = BinaryOpNode(
        left=inner, operator="+", right=IdentifierNode(name="c", symbol_type=None)
    )
    prog = ProgramNode(statements=[ExpressionStatementNode(expression=outer)])
    flat = flatten_program(prog)
    # Should introduce two temporaries
    assign1 = flat.statements[0]
    assign2 = flat.statements[1]
    expr_stmt = flat.statements[2]
    assert isinstance(assign1, AssignmentNode)
    assert isinstance(assign2, AssignmentNode)
    assert isinstance(expr_stmt, ExpressionStatementNode)
    # assign1: _tmp1 = a + b
    # assign2: _tmp2 = _tmp1 + c
    assert assign1.left.name.startswith("_tmp")
    assert assign2.left.name.startswith("_tmp")
    assert assign2.right.left.name == assign1.left.name
    assert assign2.right.right.name == "c"
    assert expr_stmt.expression.name == assign2.left.name


def test_flatten_func_call():
    # f(a + b, c)
    call = FunctionCallNode(
        function=IdentifierNode(name="f", symbol_type=None),
        arguments=[make_add("a", "b"), IdentifierNode(name="c", symbol_type=None)],
    )
    prog = ProgramNode(statements=[ExpressionStatementNode(expression=call)])
    flat = flatten_program(prog)
    # Should introduce a temp for a+b, a temp for the call, and use the call temp in the expr stmt
    assign1 = flat.statements[0]
    assign2 = flat.statements[1]
    expr_stmt = flat.statements[2]
    assert isinstance(assign1, AssignmentNode)
    assert isinstance(assign2, AssignmentNode)
    assert isinstance(expr_stmt, ExpressionStatementNode)
    # assign1: _tmp1 = a + b
    # assign2: _tmp2 = f(_tmp1, c)
    assert assign1.left.name.startswith("_tmp")
    assert assign2.left.name.startswith("_tmp")
    call_args = assign2.right.arguments
    assert call_args[0].name == assign1.left.name
    assert call_args[1].name == "c"
    assert expr_stmt.expression.name == assign2.left.name


def test_flatten_return():
    # return (a + b)
    ret = ReturnStatementNode(expression=make_add("a", "b"))
    prog = ProgramNode(statements=[ret])
    flat = flatten_program(prog)
    # Should introduce a temp for a+b, then return the temp
    assign = flat.statements[0]
    ret_stmt = flat.statements[1]
    assert isinstance(assign, AssignmentNode)
    assert isinstance(ret_stmt, ReturnStatementNode)
    assert ret_stmt.expression.name == assign.left.name


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
    # Should introduce a temp for a+b, then the if
    assign = flat.statements[0]
    if_flat = flat.statements[1]
    assert isinstance(assign, AssignmentNode)
    assert isinstance(if_flat, IfStatementNode)
    assert if_flat.condition.name == assign.left.name
    assert isinstance(if_flat.then_block, BlockNode)
    assert isinstance(if_flat.then_block.statements[0], ReturnStatementNode)
