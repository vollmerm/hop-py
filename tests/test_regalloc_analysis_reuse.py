from cfg_instrsel import Register
from regalloc import build_interference


def test_build_interference_uses_provided_analysis():
    """When instr_cfg contains an `analysis` mapping, build_interference should
    use it rather than recomputing a global instruction-level fixed-point.

    We create an instr_cfg for a single block with one instruction and attach
    an artificial `analysis` payload. The block has no successors, so a
    recomputed analysis would typically yield empty live-out sets; if
    `build_interference` reuses the provided analysis, the returned analysis
    will reflect our supplied values.
    """
    # Create some virtual registers
    v1 = Register("v1", is_virtual=True)
    v2 = Register("v2", is_virtual=True)
    v3 = Register("v3", is_virtual=True)

    # Single instruction that defines v1 and uses v2,v3
    instr = {"op": "ADD", "rd": v1, "rs1": v2, "rs2": v3}

    instr_cfg = {
        "blocks": [
            {
                "label": "block_1",
                "statements": [[instr]],
                "out_edges": [],
            }
        ],
        "entry": "block_1",
        "exit": None,
        # Provide a deliberately non-empty analysis that would differ from
        # what a recompute would produce for a block with no successors.
        "analysis": {
            "block_1": {
                "live_in": [v2],
                "live_out": [v3],
                "instr_liveness": [([v2], [v3])],
            }
        },
    }

    ig, analysis, moves = build_interference(instr_cfg)

    # The returned analysis should reflect the provided analysis (converted to sets)
    assert "block_1" in analysis
    blk = analysis["block_1"]
    assert blk["live_in"] == set([v2])
    assert blk["live_out"] == set([v3])
    # instr_liveness should be a list with one per-statement pair of sets
    assert isinstance(blk["instr_liveness"], list)
    assert len(blk["instr_liveness"]) == 1
    li, lo = blk["instr_liveness"][0]
    assert li == set([v2])
    assert lo == set([v3])
