"""Additional CFG tests to catch issues with loop back-edges and branch merges."""

from tests.utils import parse_text
from ast_flatten import flatten_program
from cfg import build_cfg
from ast_nodes import NodeType


def build_cfg_from_source(src: str):
    ast = parse_text(src)
    flat = flatten_program(ast)
    return build_cfg(flat)


def test_while_body_loops_back_complex():
    # Use the complex example in examples/complex.hop to verify at least one
    # block in the loop body has an edge back to the loop condition block.
    src = open("examples/complex.hop").read()
    cfg = build_cfg_from_source(src)
    blocks = cfg["blocks"]
    labels = {b["label"]: b for b in blocks}

    cond_labels = [
        b["label"]
        for b in blocks
        if any(s.type == NodeType.WHILE_STMT for s in b["statements"])
    ]
    assert cond_labels, "Expected at least one while condition block"

    # For each condition block, ensure some other block points back to it
    for cond in cond_labels:
        has_back = any(
            cond in blk.get("out_edges", []) and blk["label"] != cond for blk in blocks
        )
        assert has_back, f"No back-edge to while condition {cond} found"


def test_if_branches_merge():
    src = "int x; if (x > 0) { x = 1; } else { x = 2; }"
    cfg = build_cfg_from_source(src)
    blocks = cfg["blocks"]
    labels = {b["label"]: b for b in blocks}

    # find the block containing IF_STMT
    if_blocks = [
        b for b in blocks if any(s.type == NodeType.IF_STMT for s in b["statements"])
    ]
    assert if_blocks, "Expected an if-statement block"
    if_block = if_blocks[0]
    outs = if_block.get("out_edges", [])
    assert len(outs) >= 2, "If block should have two outgoing edges (then/else)"

    # The two successor blocks should both have an out_edge to a common successor (the merge)
    succs_outs = [set(labels[o].get("out_edges", [])) for o in outs if o in labels]
    assert succs_outs, "Expected successor blocks for if"
    # Check intersection
    common = set.intersection(*succs_outs) if len(succs_outs) > 1 else set()
    assert common, f"Then/else branches do not merge: {succs_outs}"


def test_nested_while_loops_back_edges():
    src = "int i; while (i < 3) { while (i < 2) { i = i + 1; } i = i + 1; }"
    cfg = build_cfg_from_source(src)
    blocks = cfg["blocks"]

    cond_labels = [
        b["label"]
        for b in blocks
        if any(s.type == NodeType.WHILE_STMT for s in b["statements"])
    ]
    # Expect two while conditions (outer and inner)
    assert len(cond_labels) >= 2
    for cond in cond_labels:
        # ensure some block different from cond points back to cond
        assert any(
            cond in blk.get("out_edges", []) and blk["label"] != cond for blk in blocks
        ), f"No back-edge to while condition {cond}"
