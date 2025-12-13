"""Unit tests for liveness analysis."""

from ast_nodes import *
from cfg import build_cfg
from ast_flatten import flatten_program
from liveness import analyze_liveness


def test_liveness_simple_assignment():
    # x = 1; y = x + 2; return y;
    block = BlockNode(
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
    cfg = build_cfg(block)
    analysis = analyze_liveness(cfg)
    # There should be a single block
    lbl = cfg["blocks"][0]["label"]
    instrs = cfg["blocks"][0]["statements"]
    instr_liv = analysis[lbl]["instr_liveness"]
    # After return, nothing live
    assert instr_liv[-1][1] == set()
    # Before return, y must be live
    assert "y" in instr_liv[-1][0]
    # For the assignment y = x + 2; x must be live after that assign
    assert "x" in instr_liv[1][0] or "x" in instr_liv[1][1]


def test_liveness_if():
    # if (a) { b = 1; } else { b = 2; } return b;
    block = BlockNode(
        statements=[
            IfStatementNode(
                condition=IdentifierNode(name="a", symbol_type=None),
                then_block=BlockNode(
                    statements=[
                        AssignmentNode(
                            left=IdentifierNode(name="b", symbol_type=None),
                            right=IntLiteralNode(value=1),
                        )
                    ]
                ),
                else_block=BlockNode(
                    statements=[
                        AssignmentNode(
                            left=IdentifierNode(name="b", symbol_type=None),
                            right=IntLiteralNode(value=2),
                        )
                    ]
                ),
            ),
            ReturnStatementNode(expression=IdentifierNode(name="b", symbol_type=None)),
        ]
    )
    cfg = build_cfg(block)
    analysis = analyze_liveness(cfg)
    # The return should require b to be live at merge
    # find the block that contains the ReturnStatement and assert b is live at its entry
    ret_block_label = None
    for b in cfg["blocks"]:
        if any(s.type == NodeType.RETURN_STMT for s in b["statements"]):
            ret_block_label = b["label"]
            break
    assert ret_block_label is not None
    assert "b" in analysis[ret_block_label]["live_in"]
