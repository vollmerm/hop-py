from cfg import build_cfg
from ast_flatten import flatten_program
from parser import Parser
from lexer import Lexer
from cfg_instrsel import select_instructions
from ast_nodes import *
from pretty_printer import PrettyPrinter


def test_simple_add():
    src = "int x = 1; int y = 2; int z = x + y;"
    ast = Parser(Lexer(src).tokenize()).parse()
    flat = flatten_program(ast)
    cfg = build_cfg(flat)
    new_cfg = select_instructions(cfg)
    # Check that at least one block contains an ADD instruction
    found = False
    for block in new_cfg["blocks"]:
        for instrs in block["statements"]:
            for instr in instrs:
                if instr.get("op") == "ADD":
                    found = True
    assert found


def test_assign_immediate():
    src = "int x = 42;"
    ast = Parser(Lexer(src).tokenize()).parse()
    flat = flatten_program(ast)
    cfg = build_cfg(flat)
    new_cfg = select_instructions(cfg)
    found = False
    for block in new_cfg["blocks"]:
        for instrs in block["statements"]:
            for instr in instrs:
                if instr.get("op") == "ADDI" and instr.get("imm") == 42:
                    found = True
    assert found


def test_array_load_store():
    src = "int arr[4]; arr[2] = 5; int x = arr[2];"
    ast = Parser(Lexer(src).tokenize()).parse()
    flat = flatten_program(ast)
    cfg = build_cfg(flat)
    new_cfg = select_instructions(cfg)
    # Pretty print for debug
    print(PrettyPrinter.print_instr_cfg(new_cfg))
    found_lw = found_sw = False
    for block in new_cfg["blocks"]:
        for instrs in block["statements"]:
            for instr in instrs:
                if instr.get("op") == "LW":
                    found_lw = True
                if instr.get("op") == "SW":
                    found_sw = True
    assert found_lw and found_sw


def test_function_call_and_return():
    src = "int foo(int a) { return a + 1; } int x = foo(41);"
    ast = Parser(Lexer(src).tokenize()).parse()
    flat = flatten_program(ast)
    cfg = build_cfg(flat)
    new_cfg = select_instructions(cfg)
    # Pretty print for debug
    print(PrettyPrinter.print_instr_cfg(new_cfg))
    found_call = found_ret = False
    for block in new_cfg["blocks"]:
        for instrs in block["statements"]:
            for instr in instrs:
                if instr.get("op") == "CALL":
                    found_call = True
                if instr.get("op") == "RET":
                    found_ret = True
    assert found_call and found_ret


def test_if_while_branch():
    src = "int x = 0; if (x < 1) { x = 2; } while (x < 10) { x = x + 1; }"
    ast = Parser(Lexer(src).tokenize()).parse()
    flat = flatten_program(ast)
    cfg = build_cfg(flat)
    new_cfg = select_instructions(cfg)
    # Pretty print for debug
    print(PrettyPrinter.print_instr_cfg(new_cfg))
    found_bnez = found_jal = False
    for block in new_cfg["blocks"]:
        for instrs in block["statements"]:
            for instr in instrs:
                if instr.get("op") == "BNEZ":
                    found_bnez = True
                if instr.get("op") == "JAL":
                    found_jal = True
    assert found_bnez and found_jal
