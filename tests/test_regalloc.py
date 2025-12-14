from cfg_instrsel import Register
from regalloc import allocate_registers


def make_cfg(instrs_per_stmt):
    """Helper: build a single-block instr_cfg with given list of statement-instr lists."""
    return {
        "blocks": [
            {
                "label": "entry",
                "statements": instrs_per_stmt,
                "out_edges": [],
            }
        ]
    }


def test_allocate_without_spill():
    v1 = Register("v1", True)
    v2 = Register("v2", True)

    # two temporaries defined and then not live at the same time
    instr1 = {"op": "ADDI", "rd": v1, "rs1": Register("x0"), "imm": 1}
    instr2 = {"op": "ADDI", "rd": v2, "rs1": Register("x0"), "imm": 2}

    cfg = make_cfg([[instr1], [instr2]])

    assign, new_cfg, spilled = allocate_registers(cfg)

    assert assign.get(v1) is not None, "v1 should be assigned a physical register"
    assert assign.get(v2) is not None, "v2 should be assigned a physical register"
    assert spilled == set(), "No spilled registers expected"


def test_spill_with_insufficient_physical_registers():
    # Construct a small program where v1 and v2 are live at the same time
    v1 = Register("v1", True)
    v2 = Register("v2", True)
    v3 = Register("v3", True)

    instr1 = {"op": "ADDI", "rd": v1, "rs1": Register("x0"), "imm": 1}
    instr2 = {"op": "ADDI", "rd": v2, "rs1": Register("x0"), "imm": 2}
    # both v1 and v2 used together -> interference
    instr3 = {"op": "ADD", "rd": v3, "rs1": v1, "rs2": v2}

    cfg = make_cfg([[instr1], [instr2], [instr3]])

    # restrict to a single physical register and ensure allocation completes
    assign, new_cfg, spilled = allocate_registers(cfg, phys_reg_names=["t0"], K=1)

    # Allocation should produce an assignment mapping (may coalesce/alias nodes)
    assert isinstance(assign, dict)
    assert any(getattr(k, "is_virtual", False) for k in assign.keys()), "Assignment should include at least one virtual register"
    # result CFG should be well-formed
    assert isinstance(new_cfg, dict) and "blocks" in new_cfg
