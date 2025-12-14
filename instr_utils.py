"""Small utilities for instruction manipulation shared by instrsel and regalloc.

This module avoids importing high-level modules to prevent circular imports.
It provides a single helper to compute uses/defs for instruction dicts.
"""

from typing import Tuple, Set, Dict, Any


def _is_reg_like(obj: Any) -> bool:
    """Heuristic to detect a Register-like object without importing cfg_instrsel.

    We expect Register objects to have a `name` attribute (string) and an
    `is_virtual` attribute; this is conservative and sufficient for our uses.
    """
    return hasattr(obj, "name") and hasattr(obj, "is_virtual")


def instr_uses_defs(instr: Dict[str, Any]) -> Tuple[Set[Any], Set[Any]]:
    """Return (uses, defs) sets of register-like objects for an instruction.

    This mirrors the previous logic spread across modules and is intentionally
    conservative: it checks common fields like 'rd', 'rs1', 'rs2', 'rs', 'base',
    and 'regs'.
    """
    uses = set()
    defs = set()

    # defs
    if "rd" in instr and _is_reg_like(instr["rd"]):
        defs.add(instr["rd"])

    # uses
    for k in ("rs1", "rs2", "rs", "rs3", "rs4"):
        if k in instr and _is_reg_like(instr[k]):
            uses.add(instr[k])

    # memory base register
    if "base" in instr and _is_reg_like(instr["base"]):
        uses.add(instr["base"])

    # For SW the value register is often in 'rs2'
    if "rs2" in instr and _is_reg_like(instr["rs2"]):
        uses.add(instr["rs2"])

    # some pseudo-ops include registers in lists
    if "regs" in instr and isinstance(instr["regs"], list):
        for r in instr["regs"]:
            if _is_reg_like(r):
                uses.add(r)

    return uses, defs
