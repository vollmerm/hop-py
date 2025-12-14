"""Tests for the simple AST interpreter used for integration testing."""

from lexer import Lexer
from parser import Parser
from ast_flatten import flatten_program
from ast_interpreter import interpret_program


def run_src(src: str):
    ast = Parser(Lexer(src).tokenize()).parse()
    flat = flatten_program(ast)
    env = interpret_program(flat)
    return env


def test_simple_arithmetic():
    src = "int x = 1 + 2;"
    env = run_src(src)
    assert env["x"] == 3


def test_function_call():
    src = "int inc(int a) { return a + 1; } int x = inc(5);"
    env = run_src(src)
    assert env["x"] == 6


def test_complex_example():
    src = open('examples/complex.hop').read()
    env = run_src(src)
    # complex.hop stores final result in `result`
    assert env.get('result') == 20
