from lexer import Lexer
from parser import Parser
from ast_interpreter import interpret_program


def test_interpret_unflattened_simple_assignment():
    src = """
    int a = 3;
    int b = 4;
    int c = a + b;
    """
    prog = Parser(Lexer(src).tokenize()).parse()
    env = interpret_program(prog)
    assert env.get("c") == 7


def test_interpret_unflattened_function_call():
    src = """
    int inc(int x) { return x + 1; }
    int r = inc(5);
    """
    prog = Parser(Lexer(src).tokenize()).parse()
    env = interpret_program(prog)
    assert env.get("r") == 6
