"""Tests for cfg_viz: ensure a Digraph is produced and contains block labels."""

from ast_nodes import *
from cfg import build_cfg
from ast_flatten import flatten_program
from cfg_viz import render_cfg_dot


def test_cfg_viz_dot_source():
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
        ]
    )
    cfg = build_cfg(block)
    dot = render_cfg_dot(cfg)
    src = dot.source
    assert "block_1" in src
    assert "Assignment" in src or "Identifier" in src
