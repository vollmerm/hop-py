import pytest
from main import lex, parse_tokens
from pretty_printer import PrettyPrinter
from type_checker import TypeChecker
from ast_nodes import NodeType


def _type_check_program(ast):
    # Run type checker over top-level program or single node
    if ast.type == NodeType.PROGRAM:
        for stmt in ast.statements:
            TypeChecker.check_statement(stmt)
    else:
        TypeChecker.check_statement(ast)


def test_type_checker_accepts_valid_functions():
    src = "int add(int a, int b) { return a + b; }"
    ast = parse_tokens(lex(src))
    _type_check_program(ast)


def test_type_checker_rejects_void_returning_value():
    src = "void foo() { return 1; }"
    ast = parse_tokens(lex(src))
    with pytest.raises(SyntaxError):
        _type_check_program(ast)


def test_type_checker_rejects_missing_return_value():
    src = "int bad() { return; }"
    ast = parse_tokens(lex(src))
    with pytest.raises(SyntaxError):
        _type_check_program(ast)


def test_type_checker_checks_call_signatures():
    src = "int add(int a, int b); int call() { return add(1); }"
    ast = parse_tokens(lex(src))
    with pytest.raises(SyntaxError):
        _type_check_program(ast)


def test_prototype_call_is_allowed_without_body():
    src = "int prot(); int call() { return prot(); }"
    ast = parse_tokens(lex(src))
    # Should not raise because prototype provides signature
    _type_check_program(ast)
