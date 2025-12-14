"""Tests for the CFG interpreter."""
from lexer import Lexer
from parser import Parser
from ast_flatten import flatten_program
from cfg import build_cfg
from cfg_interpreter import interpret_cfg


def run_cfg_src(src: str):
    ast = Parser(Lexer(src).tokenize()).parse()
    flat = flatten_program(ast)
    cfg = build_cfg(flat)
    env = interpret_cfg(cfg)
    return env


def test_cfg_simple_sequence():
    src = "int x = 1; int y = 2; int z = x + y;"
    env = run_cfg_src(src)
    assert env["z"] == 3


def test_cfg_function_call():
    src = "int inc(int a) { return a + 1; } int x = inc(7);"
    env = run_cfg_src(src)
    assert env["x"] == 8

def test_cfg_complex():
    src = open('examples/complex.hop').read()
    env = run_cfg_src(src)
    assert env.get('result') == 20
