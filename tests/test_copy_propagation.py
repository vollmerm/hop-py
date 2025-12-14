"""Tests for CFG-level copy-propagation and the tmp-removal optimization."""

from ast_nodes import *
from ast_flatten import flatten_program
from cfg import build_cfg
from cfg_opt import copy_propagation_cfg
from cfg_instrsel import select_instructions


def test_copy_propagation_simple_copy():
    # _tmp = y; var x = _tmp  => var x = y after propagation
    assign = AssignmentNode(
        left=IdentifierNode(name="_tmp"),
        right=IdentifierNode(name="y"),
    )
    vardecl = VariableDeclarationNode(
        var_name="x", init_value=IdentifierNode(name="_tmp")
    )
    prog = ProgramNode(statements=[assign, vardecl])
    flat = flatten_program(prog)
    cfg = build_cfg(flat)
    cfg2 = copy_propagation_cfg(cfg)
    # Find the block that contains the var decl and ensure its init is 'y'
    found = False
    for b in cfg2.get("blocks", []):
        for s in b.get("statements", []):
            if isinstance(s, VariableDeclarationNode) and s.var_name == "x":
                assert isinstance(s.init_value, IdentifierNode)
                assert s.init_value.name == "y"
                found = True
    assert found


def test_regression_tmp_call_removed():
    # Regression test on the real example file which previously emitted
    # `_tmp = call(...); var f5 = _tmp; return f5;` and thus produced
    # CALL + MV + ADDI + ADDI + RET. The pipeline should now produce
    # CALL followed immediately by RET in `main`.
    text = open("examples/functions_recursive.hop", "r", encoding="utf-8").read()
    from lexer import Lexer
    from parser import Parser

    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    flat = flatten_program(ast)
    cfg = build_cfg(flat)
    instr_cfg = select_instructions(cfg)

    # Find the main-like block and inspect its instruction sequences: there
    # should be no MV or ADDI-with-imm-0 between the CALL and the RET.
    # Ensure the stale pattern CALL; MV; ADDI (copy) does not occur. This
    # pattern indicates a tmp was materialized for the call and then copied
    # into a variable which is immediately returned; our pass should remove
    # that temporary.
    for b in instr_cfg.get("blocks", []):
        stmts = b.get("statements", [])
        instrs = [instr for s in stmts for instr in s]
        for i in range(len(instrs) - 2):
            a, b1, c = instrs[i : i + 3]
            if a.get("op") == "CALL" and b1.get("op") == "MV" and c.get("op") == "ADDI":
                # check ADDI is a zero-imm copy
                if c.get("imm") == 0:
                    assert False, "Found stale tmp-copy pattern CALL; MV; ADDI"

    # No bad pattern found
    assert True
