from tests.utils import parse_text
from ast_nodes import NodeType, FunctionDeclarationNode
from symbols import SymbolType


def _find_funcs(program_node):
    return [s for s in program_node.statements if s.type == NodeType.FUNC_DECL]


def test_parser_parses_variable_and_assignment():
    src = "int x = 5; x = x + 1;"
    ast = parse_text(src)
    assert ast.type == NodeType.PROGRAM
    assert len(ast.statements) == 2


def test_parser_parses_function_and_prototype():
    src = "int add(int a, int b) { return a + b; } void foo();"
    ast = parse_text(src)
    funcs = _find_funcs(ast)

    names = [f.func_name for f in funcs]
    assert "add" in names
    assert "foo" in names

    add_node = next(f for f in funcs if f.func_name == "add")
    assert isinstance(add_node, FunctionDeclarationNode)
    assert add_node.return_type == SymbolType.INT
    assert add_node.body is not None

    foo_node = next(f for f in funcs if f.func_name == "foo")
    assert foo_node.return_type == SymbolType.VOID
    assert foo_node.body is None
