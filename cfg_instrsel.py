"""Instruction selection pass for RISC-V.

Takes a CFG (as produced by cfg.py) and returns a new CFG where each statement
is replaced by a list of RISC-V-like instructions. This is a simple pattern-
based selector, not a full tree matcher. The output CFG is suitable for input
to register allocation and code emission passes.

Instruction format:
- Each instruction is a dict: { 'op': 'ADD', 'rd': 'x1', 'rs1': 'x2', 'rs2': 'x3' }
- For immediates: { 'op': 'ADDI', 'rd': 'x1', 'rs1': 'x2', 'imm': 42 }
- For loads/stores: { 'op': 'LW', 'rd': 'x1', 'offset': 0, 'base': 'x2' }
- For branches: { 'op': 'BEQ', 'rs1': 'x1', 'rs2': 'x2', 'target': 'block_3' }
- For jumps: { 'op': 'JAL', 'rd': 'x0', 'target': 'block_4' }

This pass is intentionally simple and does not optimize instruction selection.
"""

from typing import Dict, Any, List, Optional
from ast_nodes import *
from liveness import analyze_liveness
from instr_utils import instr_uses_defs as _instr_uses_defs


# Register representation
class Register:
    """Represents a RISC-V or virtual register."""

    def __init__(self, name: str, is_virtual: bool = False):
        self.name = name  # e.g. 'x10', 'a0', 'v1', ...
        self.is_virtual = is_virtual

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Reg({self.name}{' (v)' if self.is_virtual else ''})"

    def __eq__(self, other):
        return (
            isinstance(other, Register)
            and self.name == other.name
            and self.is_virtual == other.is_virtual
        )

    def __hash__(self):
        return hash((self.name, self.is_virtual))


# RISC-V register mapping for temporaries (for now, just use symbolic names)
def temp_to_reg(temp: str) -> Register:
    # Map _tmpN to virtual registers vN, otherwise to named RISC-V regs
    if temp.startswith("_tmp") and temp[4:].isdigit():
        return Register(f"v{int(temp[4:])}", is_virtual=True)
    # Recognize RISC-V ABI names
    if temp in {f"x{i}" for i in range(32)} or temp in {
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
        "s9",
        "s10",
        "s11",
        "ra",
        "sp",
        "gp",
        "tp",
        "fp",
        "zero",
    }:
        return Register(temp)
    return Register(temp, is_virtual=True)


# Named register used to hold expression results produced by
# `select_instructions_for_stmt` for non-call expressions. This is a
# symbolic placeholder (currently mapped to RISC-V `x31`) so later
# passes (register allocation) can identify and handle it specially.
EXPR_RESULT_REG = Register("x31")


def select_instructions_for_stmt(stmt) -> List[Dict[str, Any]]:
    instrs = []

    match stmt:
        case AssignmentNode(
            left=ArrayIndexNode(array=arr, index=idx), right=FunctionCallNode() as fc
        ):
            base = (
                temp_to_reg(arr.name)
                if isinstance(arr, IdentifierNode)
                else Register(str(arr), is_virtual=True)
            )
            offset = idx.value * 4 if isinstance(idx, IntLiteralNode) else 0
            instrs.extend(select_instructions_for_stmt(fc))
            instrs.append(
                {"op": "SW", "rs2": Register("a0"), "offset": offset, "base": base}
            )
            return instrs

        case AssignmentNode(left=ArrayIndexNode(array=arr, index=idx), right=right):
            base = (
                temp_to_reg(arr.name)
                if isinstance(arr, IdentifierNode)
                else Register(str(arr), is_virtual=True)
            )
            offset = idx.value * 4 if isinstance(idx, IntLiteralNode) else 0
            if isinstance(right, IdentifierNode):
                rs2 = temp_to_reg(right.name)
            else:
                rs2 = Register(str(right), is_virtual=True)
            instrs.append({"op": "SW", "rs2": rs2, "offset": offset, "base": base})
            return instrs

        case AssignmentNode(left=left, right=FunctionCallNode() as fc):
            rd = (
                temp_to_reg(left.name)
                if isinstance(left, IdentifierNode)
                else Register(str(left), is_virtual=True)
            )
            instrs.extend(select_instructions_for_stmt(fc))
            instrs.append({"op": "MV", "rd": rd, "rs1": Register("a0")})
            return instrs

        case AssignmentNode(
            left=left, right=BinaryOpNode(left=rl, operator=op, right=rr)
        ):
            op_map = {
                "+": "ADD",
                "-": "SUB",
                "*": "MUL",
                "/": "DIV",
                "%": "REM",
                "==": "SUB",
                "<": "SLT",
                ">": "SGT",
                "<=": "SLE",
                ">=": "SGE",
            }
            opcode = op_map.get(op, "ADD")
            rd = (
                temp_to_reg(left.name)
                if isinstance(left, IdentifierNode)
                else Register(str(left), is_virtual=True)
            )
            rs1 = (
                temp_to_reg(rl.name)
                if isinstance(rl, IdentifierNode)
                else Register(str(rl), is_virtual=True)
            )
            rs2 = (
                temp_to_reg(rr.name)
                if isinstance(rr, IdentifierNode)
                else Register(str(rr), is_virtual=True)
            )
            instrs.append({"op": opcode, "rd": rd, "rs1": rs1, "rs2": rs2})

        case AssignmentNode(left=left, right=UnaryOpNode(operator="-", right=rv)):
            rd = (
                temp_to_reg(left.name)
                if isinstance(left, IdentifierNode)
                else Register(str(left), is_virtual=True)
            )
            rs1 = (
                temp_to_reg(rv.name)
                if isinstance(rv, IdentifierNode)
                else Register(str(rv), is_virtual=True)
            )
            instrs.append({"op": "SUB", "rd": rd, "rs1": Register("x0"), "rs2": rs1})

        case AssignmentNode(left=left, right=UnaryOpNode(operator="!", right=rv)):
            rd = (
                temp_to_reg(left.name)
                if isinstance(left, IdentifierNode)
                else Register(str(left), is_virtual=True)
            )
            rs1 = (
                temp_to_reg(rv.name)
                if isinstance(rv, IdentifierNode)
                else Register(str(rv), is_virtual=True)
            )
            instrs.append({"op": "SEQZ", "rd": rd, "rs": rs1})

        case AssignmentNode(left=left, right=IntLiteralNode(value=v)):
            rd = (
                temp_to_reg(left.name)
                if isinstance(left, IdentifierNode)
                else Register(str(left), is_virtual=True)
            )
            instrs.append({"op": "ADDI", "rd": rd, "rs1": Register("x0"), "imm": v})

        case AssignmentNode(left=left, right=BoolLiteralNode(value=b)):
            rd = (
                temp_to_reg(left.name)
                if isinstance(left, IdentifierNode)
                else Register(str(left), is_virtual=True)
            )
            imm = 1 if b else 0
            instrs.append({"op": "ADDI", "rd": rd, "rs1": Register("x0"), "imm": imm})

        case AssignmentNode(left=left, right=IdentifierNode(name=rname)):
            rd = (
                temp_to_reg(left.name)
                if isinstance(left, IdentifierNode)
                else Register(str(left), is_virtual=True)
            )
            rs1 = temp_to_reg(rname)
            instrs.append({"op": "ADDI", "rd": rd, "rs1": rs1, "imm": 0})

        case AssignmentNode(left=left, right=ArrayIndexNode(array=arr, index=idx)):
            rd = (
                temp_to_reg(left.name)
                if isinstance(left, IdentifierNode)
                else Register(str(left), is_virtual=True)
            )
            base = (
                temp_to_reg(arr.name)
                if isinstance(arr, IdentifierNode)
                else Register(str(arr), is_virtual=True)
            )
            offset = idx.value * 4 if isinstance(idx, IntLiteralNode) else 0
            instrs.append({"op": "LW", "rd": rd, "offset": offset, "base": base})

        case AssignmentNode(left=left, right=right):
            rd = (
                temp_to_reg(left.name)
                if isinstance(left, IdentifierNode)
                else Register(str(left), is_virtual=True)
            )
            instrs.append(
                {"op": "MV", "rd": rd, "rs1": Register(str(right), is_virtual=True)}
            )

        case VariableDeclarationNode(var_name=varname, init_value=None):
            # No initializer: nothing to emit for declaration alone
            return instrs

        case VariableDeclarationNode(var_name=varname, init_value=init) if (
            init is not None
        ):
            rd = temp_to_reg(varname)
            match init:
                case IntLiteralNode(value=v):
                    instrs.append(
                        {"op": "ADDI", "rd": rd, "rs1": Register("x0"), "imm": v}
                    )
                case IdentifierNode(name=iname):
                    instrs.append(
                        {"op": "ADDI", "rd": rd, "rs1": temp_to_reg(iname), "imm": 0}
                    )
                case _:
                    # If the initializer is a function call, its return
                    # value will be in `a0`; otherwise non-call expressions
                    # place their result into `x31`.
                    if isinstance(init, FunctionCallNode):
                        sub = select_instructions_for_stmt(init)
                        instrs.extend(sub)
                        instrs.append({"op": "MV", "rd": rd, "rs1": Register("a0")})
                    else:
                        sub = select_instructions_for_stmt(init)
                        instrs.extend(sub)
                        instrs.append({"op": "MV", "rd": rd, "rs1": EXPR_RESULT_REG})

        case ExpressionStatementNode(expression=expr):
            instrs.extend(select_instructions_for_stmt(expr))

        case FunctionCallNode(arguments=args, function=fn):
            arg_regs = [
                Register(r) for r in ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]
            ]
            for i, arg in enumerate(args):
                reg = arg_regs[i] if i < len(arg_regs) else Register(f"a{i}")
                match arg:
                    case IdentifierNode(name=an):
                        instrs.append(
                            {"op": "ADDI", "rd": reg, "rs1": temp_to_reg(an), "imm": 0}
                        )
                    case IntLiteralNode(value=v):
                        instrs.append(
                            {"op": "ADDI", "rd": reg, "rs1": Register("x0"), "imm": v}
                        )
                    case _:
                        # Evaluate the argument into `x31`, then move it into the
                        # proper argument register. `select_instructions_for_stmt`
                        # emits its result into `x31` for non-assign contexts.
                        instrs.extend(select_instructions_for_stmt(arg))
                        instrs.append({"op": "MV", "rd": reg, "rs1": EXPR_RESULT_REG})
            fname = fn.name if isinstance(fn, IdentifierNode) else str(fn)
            instrs.append({"op": "CALL", "func": fname})
            # Place return value into `EXPR_RESULT_REG` so callers that
            # expect an expression result can read it there.
            instrs.append({"op": "MV", "rd": EXPR_RESULT_REG, "rs1": Register("a0")})

        case ReturnStatementNode(expression=expr) if expr is not None:
            instrs.extend(select_instructions_for_stmt(expr))
            instrs.append(
                {
                    "op": "MV",
                    "rd": Register("a0"),
                    "rs1": (
                        EXPR_RESULT_REG
                        if not isinstance(expr, IdentifierNode)
                        else temp_to_reg(expr.name)
                    ),
                }
            )
            instrs.append({"op": "RET"})

        case BinaryOpNode(left=rl, operator=op, right=rr):
            op_map = {
                "+": "ADD",
                "-": "SUB",
                "*": "MUL",
                "/": "DIV",
                "%": "REM",
                "==": "SUB",
                "<": "SLT",
                ">": "SGT",
                "<=": "SLE",
                ">=": "SGE",
            }
            opcode = op_map.get(op, "ADD")
            rs1 = (
                temp_to_reg(rl.name)
                if isinstance(rl, IdentifierNode)
                else Register(str(rl), is_virtual=True)
            )
            rs2 = (
                temp_to_reg(rr.name)
                if isinstance(rr, IdentifierNode)
                else Register(str(rr), is_virtual=True)
            )
            instrs.append({"op": opcode, "rd": EXPR_RESULT_REG, "rs1": rs1, "rs2": rs2})
            return instrs

        case UnaryOpNode(operator="-", right=rv):
            rs1 = (
                temp_to_reg(rv.name)
                if isinstance(rv, IdentifierNode)
                else Register(str(rv), is_virtual=True)
            )
            instrs.append({"op": "SUB", "rd": EXPR_RESULT_REG, "rs1": Register("x0"), "rs2": rs1})
            return instrs

        case UnaryOpNode(operator="!", right=rv):
            rs1 = (
                temp_to_reg(rv.name)
                if isinstance(rv, IdentifierNode)
                else Register(str(rv), is_virtual=True)
            )
            instrs.append({"op": "SEQZ", "rd": EXPR_RESULT_REG, "rs": rs1})
            return instrs

        case ArrayIndexNode(array=arr, index=idx):
            base = (
                temp_to_reg(arr.name)
                if isinstance(arr, IdentifierNode)
                else Register(str(arr), is_virtual=True)
            )
            offset = idx.value * 4 if isinstance(idx, IntLiteralNode) else 0
            instrs.append({"op": "LW", "rd": EXPR_RESULT_REG, "offset": offset, "base": base})
            return instrs

        case ReturnStatementNode():
            instrs.append({"op": "RET"})

        case FunctionDeclarationNode(func_name=name, body=body) if body is not None:
            # Emit only a function label here. The CFG already places the
            # function body statements into subsequent blocks; inlining the
            # body here would duplicate instructions in the instruction-level
            # CFG and visualization.
            instrs.append({"op": "FUNC_LABEL", "name": name})

        case IfStatementNode(condition=cond, then_block=then_b, else_block=else_b):
            cond_reg = EXPR_RESULT_REG
            instrs.extend(select_instructions_for_stmt(cond))
            instrs.append({"op": "BNEZ", "rs": cond_reg, "target": "then_block"})
            if else_b:
                instrs.append(
                    {"op": "JAL", "rd": Register("x0"), "target": "else_block"}
                )

        case WhileStatementNode(condition=cond, body=body):
            cond_reg = EXPR_RESULT_REG
            instrs.extend(select_instructions_for_stmt(cond))
            instrs.append({"op": "BNEZ", "rs": cond_reg, "target": "body_block"})
            instrs.append({"op": "JAL", "rd": Register("x0"), "target": "exit_block"})

        case BlockNode(statements=stmts):
            for s in stmts:
                instrs.extend(select_instructions_for_stmt(s))

        case ProgramNode(statements=stmts):
            for s in stmts:
                instrs.extend(select_instructions_for_stmt(s))

        case IntLiteralNode(value=v):
            instrs.append({"op": "ADDI", "rd": EXPR_RESULT_REG, "rs1": Register("x0"), "imm": v})

        case BoolLiteralNode(value=b):
            imm = 1 if b else 0
            instrs.append({"op": "ADDI", "rd": EXPR_RESULT_REG, "rs1": Register("x0"), "imm": imm})

        case IdentifierNode(name=n):
            instrs.append({"op": "MV", "rd": EXPR_RESULT_REG, "rs1": temp_to_reg(n)})

        case _:
            # Unknown node type: fail loudly to avoid silent compiler errors
            node_type = type(stmt).__name__
            raise RuntimeError(
                f"cfg_instrsel: unhandled AST node type in instruction selection: {node_type}: {stmt}"
            )

    return instrs


def select_instructions(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Given a CFG, return a new CFG with statements replaced by lists of RISC-V instructions.

    This function also computes and attaches per-block and per-statement
    instruction-level liveness into the returned `instr_cfg["analysis"]` so
    downstream passes (notably register allocation) can reuse it and avoid
    recomputing a global instruction-level fixed-point.
    """
    new_blocks = []
    for block in cfg.get("blocks", []):
        new_stmts = []
        for stmt in block.get("statements", []):
            instrs = select_instructions_for_stmt(stmt)
            new_stmts.append(instrs)
        new_blocks.append(
            {
                "label": block["label"],
                "statements": new_stmts,
                "out_edges": list(block.get("out_edges", [])),
            }
        )

    instr_cfg = {
        "blocks": new_blocks,
        "entry": cfg.get("entry"),
        "exit": cfg.get("exit"),
    }

    # Compute CFG-level liveness (by variable name) and convert to per-statement
    # per-instruction liveness expressed with `Register` objects. This allows
    # `regalloc` to consume `instr_cfg["analysis"]` directly.
    try:
        cfg_liv = analyze_liveness(cfg)
    except Exception:
        cfg_liv = None

    analysis: Dict[str, Dict[str, Any]] = {}
    for b in instr_cfg.get("blocks", []):
        lbl = b["label"]
        stmts = b.get("statements", [])
        # Default empty per-statement pairs
        per_stmt_pairs: List[List[tuple]] = [ ([], []) for _ in stmts ]

        if cfg_liv and lbl in cfg_liv:
            blk_liv = cfg_liv[lbl].get("instr_liveness", [])
            # blk_liv is a list of (live_in_names, live_out_names) per statement
            per_stmt_pairs = []
            for i, stmt_instrs in enumerate(stmts):
                # get statement-level live_out (names) from CFG liveness; fall back to empty set
                if i < len(blk_liv):
                    stmt_li_names, stmt_lo_names = blk_liv[i]
                else:
                    stmt_li_names, stmt_lo_names = set(), set()

                # map name-based live_out to Register objects
                live_after = set()
                for name in stmt_lo_names:
                    live_after.add(temp_to_reg(name) if isinstance(name, str) else Register(str(name), is_virtual=True))

                # Compute per-instruction liveness for this statement by scanning backwards
                instr_pairs: List[tuple] = []
                # Work on a copy since we'll mutate live_after
                la = set(live_after)
                for instr in reversed(stmt_instrs):
                    uses, defs = _instr_uses_defs(instr)
                    live_in = (la - defs) | uses
                    live_out = la.copy()
                    instr_pairs.append((list(live_in), list(live_out)))
                    la = live_in
                instr_pairs.reverse()
                # If the statement had no instructions, still record the stmt-level pair
                if not instr_pairs:
                    per_stmt_pairs.append((list(live_after), list(live_after)))
                else:
                    # For compatibility with `build_interference` which expects a
                    # per-statement pair (live_in of first instr, live_out of last instr)
                    first_in = instr_pairs[0][0]
                    last_out = instr_pairs[-1][1]
                    per_stmt_pairs.append((first_in, last_out))
        else:
            # No prior liveness available: emit conservative empty pairs
            per_stmt_pairs = [ ([], []) for _ in stmts ]

        # Store in analysis: lists (so JSON-ification still works if needed)
        # live_in/live_out for block approximate from per_stmt_pairs
        blk_live_in = set(per_stmt_pairs[0][0]) if per_stmt_pairs else set()
        blk_live_out = set(per_stmt_pairs[-1][1]) if per_stmt_pairs else set()
        analysis[lbl] = {
            "live_in": list(blk_live_in),
            "live_out": list(blk_live_out),
            "instr_liveness": per_stmt_pairs,
        }

    instr_cfg["analysis"] = analysis
    return instr_cfg
