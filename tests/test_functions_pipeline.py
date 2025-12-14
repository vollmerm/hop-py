from parser import Parser
from lexer import Lexer
from ast_flatten import flatten_program
from cfg import build_cfg
from cfg_instrsel import select_instructions
from pretty_printer import PrettyPrinter


def test_single_function_declaration():
    src = "int foo() { int x = 1; return x; }"
    ast = Parser(Lexer(src).tokenize()).parse()
    flat = flatten_program(ast)
    cfg = build_cfg(flat)
    new_cfg = select_instructions(cfg)

    # Ensure a function label and a return are produced
    found_func = found_ret = False
    for block in new_cfg["blocks"]:
        for instrs in block["statements"]:
            for instr in instrs:
                if instr.get("op") == "FUNC_LABEL" and instr.get("name") == "foo":
                    found_func = True
                if instr.get("op") == "RET":
                    found_ret = True

    assert found_func and found_ret


def test_function_call_between_functions():
    src = "int foo() { return 7; }" " int bar() { return foo(); }"
    ast = Parser(Lexer(src).tokenize()).parse()
    flat = flatten_program(ast)
    cfg = build_cfg(flat)
    new_cfg = select_instructions(cfg)

    # Expect two function labels and a call
    funcs = set()
    found_call = False
    for block in new_cfg["blocks"]:
        for instrs in block["statements"]:
            for instr in instrs:
                if instr.get("op") == "FUNC_LABEL":
                    funcs.add(instr.get("name"))
                if instr.get("op") == "CALL":
                    found_call = True

    assert "foo" in funcs and "bar" in funcs and found_call


def test_function_with_param_and_call_assign():
    src = "int inc(int a) { return a + 1; }" " int x = inc(5);"
    ast = Parser(Lexer(src).tokenize()).parse()
    flat = flatten_program(ast)
    cfg = build_cfg(flat)
    new_cfg = select_instructions(cfg)

    found_call = found_mv = found_func = False
    for block in new_cfg["blocks"]:
        for instrs in block["statements"]:
            for instr in instrs:
                if instr.get("op") == "CALL":
                    found_call = True
                if instr.get("op") == "MV" and instr.get("rd") and instr.get("rs1"):
                    # MV used after calls to move a0 into result
                    found_mv = True
                if instr.get("op") == "FUNC_LABEL" and instr.get("name") == "inc":
                    found_func = True

    # Basic sanity: function defined, call emitted, and move from a0 to result
    assert found_func and found_call and found_mv
