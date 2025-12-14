"""Simple register-allocation helpers: build interference graph from instr-CFG.

This module computes register-level liveness for an instruction-level CFG
and produces an interference graph suitable as input to a graph-coloring
register allocator.

The input expected is the output of `cfg_instrsel.select_instructions(cfg)`,
where each block's `statements` is a list of lists of instruction dicts.

The interference graph is represented as a mapping `Register -> set(Register)`
where edges are undirected (both directions present).
"""

from typing import Dict, Set, List, Tuple, Any
from collections import defaultdict
from cfg_instrsel import Register
from instr_utils import instr_uses_defs as _instr_uses_defs


# instr_uses_defs provided by instr_utils


def build_interference(
    instr_cfg: Dict[str, Any],
) -> Tuple[
    Dict[Register, Set[Register]],
    Dict[str, Dict[str, Any]],
    List[Tuple[Register, Register]],
]:
    """Compute interference graph, instruction-level liveness, and move list.

    Returns (interference_graph, liveness, moves) where `liveness` maps block label
    to { 'live_in': set(Register), 'live_out': set(Register),
    'instr_liveness': [(live_in, live_out), ...] } and `moves` is a list of
    `(dst, src)` register pairs corresponding to `MV` (move) instructions.
    The move list can be used by a subsequent allocator to attempt move
    coalescing. This function also avoids adding interference edges between
    direct move pairs when building the interference graph (optimistic
    coalescing), but a fuller coalescing algorithm can be implemented in the
    allocator using the returned `moves` list.
    """
    blocks = instr_cfg.get("blocks", [])
    # map label->block
    block_map = {b["label"]: b for b in blocks}
    succs = {b["label"]: list(b.get("out_edges", [])) for b in blocks}

    # If the instr_cfg already contains precomputed per-statement/instruction
    # liveness (produced by instruction selection using the CFG-level analysis),
    # reuse it and skip the expensive global fixed-point. Expected shape:
    # instr_cfg["analysis"][label] -> { 'live_in': [...], 'live_out': [...], 'instr_liveness': [ (live_in, live_out), ... ] }
    if isinstance(instr_cfg.get("analysis"), dict):
        # When instruction selection attached `analysis`, it may contain names
        # (strings) or Register objects. Convert names -> Register objects and
        # ignore any unexpected items (e.g., AST nodes) to avoid leaking them
        # into the interference analysis.
        from cfg_instrsel import temp_to_reg

        def _to_reg_set(iterable):
            out = set()
            for x in iterable:
                if isinstance(x, str):
                    out.add(temp_to_reg(x))
                else:
                    # already a Register-like object?
                    try:
                        if getattr(x, "name", None) is not None:
                            out.add(x)
                    except Exception:
                        # skip unexpected types like AST nodes
                        continue
            return out

        analysis = {}
        for b in blocks:
            lbl = b["label"]
            blk = instr_cfg["analysis"].get(lbl)
            if blk and "instr_liveness" in blk:
                per_stmt = []
                for pair in blk.get("instr_liveness", []):
                    # pair expected to be (live_in, live_out) where elements are lists
                    try:
                        li, lo = pair
                    except Exception:
                        # malformed entry: skip
                        continue
                    per_stmt.append((_to_reg_set(li), _to_reg_set(lo)))
                live_in = _to_reg_set(
                    blk.get("live_in", per_stmt[0][0] if per_stmt else [])
                )
                live_out = _to_reg_set(
                    blk.get("live_out", per_stmt[-1][1] if per_stmt else [])
                )
                analysis[lbl] = {
                    "live_in": live_in,
                    "live_out": live_out,
                    "instr_liveness": per_stmt,
                }
            else:
                # fallback to empty structure for this block
                analysis[lbl] = {
                    "live_in": set(),
                    "live_out": set(),
                    "instr_liveness": [(set(), set()) for _ in b.get("statements", [])],
                }
    else:
        # initialize analysis and run a global fixed-point over instruction sequences
        analysis: Dict[str, Dict[str, Any]] = {}
        for b in blocks:
            analysis[b["label"]] = {
                "live_in": set(),
                "live_out": set(),
                "instr_liveness": [(set(), set()) for _ in b.get("statements", [])],
            }

        changed = True
        while changed:
            changed = False
            # iterate blocks in reverse for faster convergence
            for b in reversed(blocks):
                lbl = b["label"]
                old_in = analysis[lbl]["live_in"].copy()
                old_out = analysis[lbl]["live_out"].copy()

                # live_out = union of live_in of successors
                new_out: Set[Register] = set()
                for s in succs.get(lbl, []):
                    new_out |= analysis[s]["live_in"]

                # compute per-instruction liveness by scanning statements in reverse
                instr_lists: List[List[Dict[str, Any]]] = b.get("statements", [])
                live_after: Set[Register] = new_out.copy()
                instr_liv_rev: List[Tuple[Set[Register], Set[Register]]] = []
                # iterate over statement-instruction groups in reverse
                for stmt_instrs in reversed(instr_lists):
                    # each statement may emit a sequence of instructions; process each instr backwards
                    for instr in reversed(stmt_instrs):
                        uses, defs = _instr_uses_defs(instr)
                        live_in = (live_after - defs) | uses
                        live_out = live_after.copy()
                        instr_liv_rev.append((live_in.copy(), live_out.copy()))
                        live_after = live_in
                # instr_liv_rev currently contains per-instruction entries across all statements;
                # for simplicity we store per-statement liveness by grouping the per-instrs back into
                # per-statement pairs using the last instr in each statement as representative.
                instr_liv_rev.reverse()
                # Build a per-statement list: for each statement (list of instrs), we find the
                # corresponding live_in/live_out by consuming as many instr entries as instrs len
                per_stmt_liveness: List[Tuple[Set[Register], Set[Register]]] = []
                idx = 0
                # flatten lengths
                stmt_lens = [len(s) for s in instr_lists]
                for length in stmt_lens:
                    if length == 0:
                        # no instrs -> liveness equals current live_after (new_out at end)
                        per_stmt_liveness.append((set(), new_out.copy()))
                        continue
                    # take next `length` entries
                    # the per-instr entries are in order; we want the first instr's live_in and last's live_out
                    first_live_in = instr_liv_rev[idx][0]
                    last_live_out = instr_liv_rev[idx + length - 1][1]
                    per_stmt_liveness.append(
                        (first_live_in.copy(), last_live_out.copy())
                    )
                    idx += length

                analysis[lbl]["instr_liveness"] = per_stmt_liveness
                new_in = (
                    per_stmt_liveness[0][0] if per_stmt_liveness else new_out.copy()
                )
                analysis[lbl]["live_out"] = new_out
                analysis[lbl]["live_in"] = new_in

                if new_in != old_in or new_out != old_out:
                    changed = True

    # Build interference graph
    ig: Dict[Register, Set[Register]] = defaultdict(set)
    moves: Set[Tuple[Register, Register]] = set()
    for b in blocks:
        lbl = b["label"]
        stmt_instrs_list = b.get("statements", [])
        per_stmt_liv = analysis[lbl]["instr_liveness"]
        for stmt_instrs, (s_live_in, s_live_out) in zip(stmt_instrs_list, per_stmt_liv):
            # For each instruction within the statement, recompute uses/defs and live sets by walking forward
            # We'll approximate: use the statement-level live_out as the live set after the statement,
            # and for each def in the statement add interference with all regs live_out.
            live_out_regs = set(s_live_out)
            # collect all defs within the statement
            defs_in_stmt: Set[Register] = set()
            for instr in stmt_instrs:
                # detect simple move instructions for coalescing
                op = instr.get("op")
                if op == "MV":
                    rd = instr.get("rd")
                    rs1 = instr.get("rs1")
                    if isinstance(rd, Register) and isinstance(rs1, Register):
                        # record move as (dst, src)
                        moves.add((rd, rs1))

                _, defs = _instr_uses_defs(instr)
                defs_in_stmt |= defs
            for d in defs_in_stmt:
                for live_r in live_out_regs:
                    if live_r == d:
                        continue
                    # optimistic: if this def-live pair is exactly a move pair, skip adding
                    # interference now to allow coalescing; a later allocator should
                    # validate legality before actually merging nodes.
                    if (d, live_r) in moves or (live_r, d) in moves:
                        continue
                    ig[d].add(live_r)
                    ig[live_r].add(d)

    # ensure all registers seen appear in ig
    for b in blocks:
        for stmt_instrs in b.get("statements", []):
            for instr in stmt_instrs:
                uses, defs = _instr_uses_defs(instr)
                for r in uses | defs:
                    ig.setdefault(r, set())

    # ensure all registers seen appear in ig
    for b in blocks:
        for stmt_instrs in b.get("statements", []):
            for instr in stmt_instrs:
                uses, defs = _instr_uses_defs(instr)
                for r in uses | defs:
                    ig.setdefault(r, set())

    return dict(ig), analysis, list(moves)


def allocate_registers(
    instr_cfg: Dict[str, Any], phys_reg_names: List[str] = None, K: int = None
):
    """Attempt to allocate registers using an iterative Briggs/George approach.

    Returns (assignment, rewritten_cfg, spilled) where `assignment` maps
    virtual `Register` -> physical `Register` (or None for spilled),
    `rewritten_cfg` is the possibly rewritten instruction CFG (with spill stores/loads inserted),
    and `spilled` is a set of registers that were spilled.

    This implementation is a pragmatic, simplified form of Briggs/George:
      - builds interference graph and move list
      - performs simplify/coalesce/spill selection using Briggs' conservative
        criterion for coalescing (degree(u)+degree(v) < K)
      - assigns colors (physical registers) by popping the stack
      - when spills occur, performs a simple rewrite using per-spill stack slots
        and retries allocation (iterative)

    Note: This allocator is intentionally simple and conservative; it may
    not produce optimal coalescing or minimal spills.
    """
    if phys_reg_names is None:
        phys_reg_names = [f"t{i}" for i in range(7)]  # t0..t6
    if K is None:
        K = len(phys_reg_names)

    # convert names to Register objects
    phys_regs = [Register(n) for n in phys_reg_names]

    def run_once(cfg):
        ig, liveness, moves = build_interference(cfg)

        # Separate precolored (physical) and virtual nodes
        nodes = set(ig.keys())
        precolored = {r for r in nodes if not getattr(r, "is_virtual", False)}

        # Work structures
        adj = {n: set(neighs) for n, neighs in ig.items()}
        degree = {n: len(adj[n]) for n in adj}

        # move set
        move_set = set(moves)

        stack: List[Register] = []
        removed: Set[Register] = set()

        # coalesced nodes alias mapping
        alias: Dict[Register, Register] = {}

        # Helper to find representative
        def find(u):
            while alias.get(u, u) != u:
                u = alias[u]
            return u

        # simplify/coalesce/spill loop
        while True:
            progressed = False

            # Simplify: remove any non-precolored node with degree < K
            simple_nodes = [
                n for n in nodes - removed - precolored if degree.get(n, 0) < K
            ]
            if simple_nodes:
                for n in simple_nodes:
                    removed.add(n)
                    stack.append(n)
                    # remove from neighbors
                    for m in list(adj.get(n, [])):
                        adj[m].discard(n)
                        degree[m] = len(adj[m])
                    adj[n].clear()
                    degree[n] = 0
                progressed = True
                continue

            # Coalesce: attempt to merge a move pair satisfying Briggs' rule
            coalesced = False
            for d, s in list(move_set):
                if d in removed or s in removed:
                    continue
                ud = find(d)
                us = find(s)
                if ud == us:
                    move_set.discard((d, s))
                    continue
                deg_ud = degree.get(ud, 0)
                deg_us = degree.get(us, 0)
                # Briggs conservative test
                if (deg_ud + deg_us) < K:
                    # merge us into ud
                    alias[us] = ud
                    # union neighbors
                    neighs = (adj.get(ud, set()) | adj.get(us, set())) - {ud, us}
                    adj[ud] = neighs
                    degree[ud] = len(neighs)
                    # update neighbors to point to ud
                    for n in neighs:
                        adj[n].discard(ud)
                        adj[n].discard(us)
                        adj[n].add(ud)
                        degree[n] = len(adj[n])
                    removed.add(us)
                    coalesced = True
                    progressed = True
                    # remove any moves involving us
                    move_set = {mv for mv in move_set if us not in mv}
                    break
            if coalesced:
                continue

            # If nothing to do, select a spill candidate: highest degree virtual node
            spill_candidates = [n for n in nodes - removed - precolored]
            if spill_candidates:
                # pick node with max degree
                spill_node = max(spill_candidates, key=lambda x: degree.get(x, 0))
                removed.add(spill_node)
                stack.append(spill_node)
                for m in list(adj.get(spill_node, [])):
                    adj[m].discard(spill_node)
                    degree[m] = len(adj[m])
                adj[spill_node].clear()
                degree[spill_node] = 0
                progressed = True

            if not progressed:
                break

        # Assign colors by popping stack
        assignment: Dict[Register, Register] = {}
        spilled: Set[Register] = set()

        # Pre-color physical regs to themselves
        for pr in phys_regs:
            assignment[pr] = pr

        while stack:
            n = stack.pop()
            # representative
            rep = find(n)
            if rep in assignment:
                assignment[n] = assignment[rep]
                continue
            neighbor_colors = {
                assignment.get(find(nb))
                for nb in ig.get(n, set())
                if assignment.get(find(nb))
            }
            avail = [r for r in phys_regs if r not in neighbor_colors]
            if avail:
                assignment[n] = avail[0]
            else:
                # spill
                assignment[n] = None
                spilled.add(n)

        return assignment, spilled, liveness

    # Iteratively try allocation, rewriting if spills occur
    current_cfg = instr_cfg
    max_iters = 5
    for _ in range(max_iters):
        assign, spilled, liveness = run_once(current_cfg)
        if not spilled:
            # successful allocation: produce rewritten cfg with replaced regs
            new_cfg = _apply_assignment(instr_cfg, assign)
            return assign, new_cfg, set()

        # simple spill rewriting: allocate stack slot per spilled reg and insert loads/stores
        # For each spilled register `r`, create a stack slot name and replace uses/defs
        spill_slots: Dict[Register, str] = {
            r: f"spill_{abs(hash(r)) % 100000}" for r in spilled
        }
        current_cfg = _rewrite_spills(current_cfg, spill_slots, phys_regs)

    # if still spilling after iterations, return last assignment and spilled set
    assign, spilled, _ = run_once(current_cfg)
    new_cfg = _apply_assignment(current_cfg, assign)
    return assign, new_cfg, spilled


def _apply_assignment(
    instr_cfg: Dict[str, Any], assignment: Dict[Register, Register]
) -> Dict[str, Any]:
    """Return a copy of instr_cfg with Register operands replaced by assigned physical regs (or left as-is)."""
    import copy

    cfg_copy = copy.deepcopy(instr_cfg)
    for b in cfg_copy.get("blocks", []):
        for stmt_instrs in b.get("statements", []):
            for instr in stmt_instrs:
                for k, v in list(instr.items()):
                    if isinstance(v, Register):
                        mapped = assignment.get(v)
                        if mapped is None:
                            # spilled; leave as original register to indicate unresolved
                            instr[k] = v
                        else:
                            instr[k] = mapped
                    elif isinstance(v, list):
                        instr[k] = [
                            assignment.get(x, x) if isinstance(x, Register) else x
                            for x in v
                        ]
    return cfg_copy


def _find_unused_phys_reg(
    instr_cfg: Dict[str, Any], phys_regs: List[Register]
) -> Register:
    """Return a physical Register from phys_regs that does not appear in instr_cfg.

    If none is unused, return the first phys_reg as a fallback.
    """
    seen = set()
    for b in instr_cfg.get("blocks", []):
        for stmt_instrs in b.get("statements", []):
            for instr in stmt_instrs:
                for v in instr.values():
                    if isinstance(v, Register):
                        seen.add(v.name)
                    elif isinstance(v, list):
                        for x in v:
                            if isinstance(x, Register):
                                seen.add(x.name)
    # ABI-reserved registers to avoid when selecting a scratch reg
    ABI_RESERVED = set(
        [
            "sp",
            "ra",
            "gp",
            "tp",
            "fp",
        ]
        + [f"a{i}" for i in range(8)]
        + [f"s{i}" for i in range(12)]
    )

    # Prefer a phys reg that is unused and not ABI-reserved
    for pr in phys_regs:
        if pr.name not in seen and pr.name not in ABI_RESERVED:
            return pr, False

    # If none found, try any unused phys reg (even if ABI)
    for pr in phys_regs:
        if pr.name not in seen:
            return pr, False

    # If all are used, return the first phys reg and indicate reservation needed
    if phys_regs:
        return phys_regs[0], True
    return Register("t0"), True


def _rewrite_spills(
    instr_cfg: Dict[str, Any],
    spill_slots: Dict[Register, str],
    phys_regs: List[Register],
) -> Dict[str, Any]:
    """Rewrite instr_cfg to spill registers to memory slots and return new cfg.

    This is a simple rewriter that, for each instruction referencing a spilled
    Register `r`, inserts loads/stores around the instruction that move the
    spilled value to/from a temporary physical register (`t0`). This is not
    optimal but allows iterative allocation to reduce virtual register pressure.
    """
    import copy

    cfg_copy = copy.deepcopy(instr_cfg)
    tmp_reg, needs_reserve = _find_unused_phys_reg(cfg_copy, phys_regs)

    # If tmp_reg was in use, reserve it by replacing its occurrences with
    # a fresh virtual register and inserting save/restore moves only where
    # tmp_reg is live (liveness-aware).
    if needs_reserve:
        reserve_name = f"v_reserve_{abs(hash(tmp_reg.name)) % 100000}"
        v_reserve = Register(reserve_name, is_virtual=True)

        # compute liveness for cfg_copy to decide where tmp_reg is live
        _, analysis, _ = build_interference(cfg_copy)

        for b in cfg_copy.get("blocks", []):
            lbl = b.get("label")
            block_info = analysis.get(lbl, {"live_in": set(), "live_out": set()})
            need_save = tmp_reg in block_info.get("live_in", set())
            need_restore = tmp_reg in block_info.get("live_out", set())

            # replace occurrences of tmp_reg with v_reserve in all instructions
            for stmt_instrs in b.get("statements", []):
                for instr in stmt_instrs:
                    for k, v in list(instr.items()):
                        if isinstance(v, Register) and v.name == tmp_reg.name:
                            instr[k] = v_reserve
                        elif isinstance(v, list):
                            instr[k] = [
                                (
                                    v_reserve
                                    if isinstance(x, Register)
                                    and x.name == tmp_reg.name
                                    else x
                                )
                                for x in v
                            ]

            stmts = b.get("statements", [])
            # Insert save at block entry if tmp_reg is live-in
            if need_save:
                save_instr = {"op": "MV", "rd": v_reserve, "rs1": tmp_reg}
                stmts = [[save_instr]] + stmts

            # Insert restore at block exit if tmp_reg is live-out
            if need_restore:
                restore_instr = {"op": "MV", "rd": tmp_reg, "rs1": v_reserve}
                stmts = stmts + [[restore_instr]]

            b["statements"] = stmts

    for b in cfg_copy.get("blocks", []):
        new_stmt_list: List[List[Dict[str, Any]]] = []
        for stmt_instrs in b.get("statements", []):
            new_instrs: List[Dict[str, Any]] = []
            for instr in stmt_instrs:
                # For each spilled reg used in this instr, insert a load before
                uses, defs = _instr_uses_defs(instr)
                pre: List[Dict[str, Any]] = []
                post: List[Dict[str, Any]] = []
                # loads for uses
                for u in list(uses):
                    if u in spill_slots:
                        slot = spill_slots[u]
                        # load into tmp_reg
                        pre.append(
                            {
                                "op": "LW",
                                "rd": tmp_reg,
                                "offset": 0,
                                "base": Register(slot),
                            }
                        )
                        # replace uses of u in instr with tmp_reg
                        for k, v in list(instr.items()):
                            if v == u:
                                instr[k] = tmp_reg
                # stores for defs
                for d in list(defs):
                    if d in spill_slots:
                        slot = spill_slots[d]
                        # write result from tmp_reg after instr
                        post.append(
                            {
                                "op": "SW",
                                "rs2": tmp_reg,
                                "offset": 0,
                                "base": Register(slot),
                            }
                        )
                        # replace defs in instr to write to tmp_reg
                        for k, v in list(instr.items()):
                            if v == d and k == "rd":
                                instr[k] = tmp_reg

                new_instrs.extend(pre)
                new_instrs.append(instr)
                new_instrs.extend(post)

            new_stmt_list.append(new_instrs)
        b["statements"] = new_stmt_list

    return cfg_copy
