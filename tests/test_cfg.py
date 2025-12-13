"""Unit tests for cfg.py: checks CFG construction from flattened AST blocks and functions."""

from ast_nodes import *
from ast_flatten import flatten_program
from cfg import build_cfg


def make_flat_block():
    # a = 1; b = 2; if (a) { b = 3; } else { b = 4; } return b;
    stmts = [
        AssignmentNode(
            left=IdentifierNode(name="a", symbol_type=None),
            right=IntLiteralNode(value=1),
        ),
        AssignmentNode(
            left=IdentifierNode(name="b", symbol_type=None),
            right=IntLiteralNode(value=2),
        ),
        IfStatementNode(
            condition=IdentifierNode(name="a", symbol_type=None),
            then_block=BlockNode(
                statements=[
                    AssignmentNode(
                        left=IdentifierNode(name="b", symbol_type=None),
                        right=IntLiteralNode(value=3),
                    )
                ]
            ),
            else_block=BlockNode(
                statements=[
                    AssignmentNode(
                        left=IdentifierNode(name="b", symbol_type=None),
                        right=IntLiteralNode(value=4),
                    )
                ]
            ),
        ),
        ReturnStatementNode(expression=IdentifierNode(name="b", symbol_type=None)),
    ]
    return BlockNode(statements=stmts)


def test_cfg_block():
    block = make_flat_block()
    cfg = build_cfg(block)
    # There should be multiple blocks: entry, then, else, after-if, etc.
    labels = set(b["label"] for b in cfg["blocks"])
    assert cfg["entry"] in labels
    assert cfg["exit"] in labels
    # Check that each block has statements and out_edges
    for b in cfg["blocks"]:
        assert "label" in b
        assert "statements" in b
        assert "out_edges" in b
    # There should be a block with the return statement
    found_return = any(
        any(s.type == NodeType.RETURN_STMT for s in b["statements"])
        for b in cfg["blocks"]
    )
    assert found_return


def test_cfg_function():
    # function foo() { a = 1; return a; }
    body = BlockNode(
        statements=[
            AssignmentNode(
                left=IdentifierNode(name="a", symbol_type=None),
                right=IntLiteralNode(value=1),
            ),
            ReturnStatementNode(expression=IdentifierNode(name="a", symbol_type=None)),
        ]
    )
    func = FunctionDeclarationNode(
        func_name="foo", return_type="int", arg_types=[], arg_names=[], body=body
    )
    cfg = build_cfg(func)
    labels = set(b["label"] for b in cfg["blocks"])
    assert cfg["entry"] in labels
    assert cfg["exit"] in labels
    # There should be a block with the return statement
    found_return = any(
        any(s.type == NodeType.RETURN_STMT for s in b["statements"])
        for b in cfg["blocks"]
    )
    assert found_return
