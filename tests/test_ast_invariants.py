from main import lex, parse_tokens
from ast_nodes import *
from tokens import TokenType
from pretty_printer import PrettyPrinter


def test_function_declaration_arg_lengths_and_body():
    src = "int f(int a, int b); int g(int x) { return x; }"
    ast = parse_tokens(lex(src))
    funcs = [s for s in ast.statements if isinstance(s, FunctionDeclarationNode)]

    # Prototype f: arg lengths should match and body is None
    fnode = next(f for f in funcs if f.func_name == "f")
    assert len(fnode.arg_names) == len(fnode.arg_types)
    assert fnode.body is None

    # Definition g: body present and arg name/type lengths match
    gnode = next(f for f in funcs if f.func_name == "g")
    assert gnode.body is not None
    assert len(gnode.arg_names) == len(gnode.arg_types)


def test_identifier_function_flag_and_parameters_propagated_from_symbols():
    src = "int add(int a, int b); int call() { return add(1, 2); }"
    ast = parse_tokens(lex(src))

    # Find the function call inside call's body
    call_fn = next(
        s
        for s in ast.statements
        if isinstance(s, FunctionDeclarationNode) and s.func_name == "call"
    )
    body = call_fn.body
    assert isinstance(body, BlockNode)
    # First statement should be Return with FunctionCall
    ret = body.statements[0]
    assert isinstance(ret, ReturnStatementNode)
    assert isinstance(ret.expression, FunctionCallNode)
    func_ident = ret.expression.function
    # The identifier used for function call should be flagged as function and have parameters
    assert isinstance(func_ident, IdentifierNode)
    assert func_ident.is_function is True
    assert len(func_ident.parameters) == 2


def test_array_index_and_assignment_ast_shape():
    src = "int[] arr; arr[0] = 10;"
    ast = parse_tokens(lex(src))
    prog = ast
    assert isinstance(prog, ProgramNode)
    assign_stmt = prog.statements[1]
    assert isinstance(assign_stmt, ExpressionStatementNode)
    assign = assign_stmt.expression
    assert isinstance(assign, AssignmentNode)
    assert isinstance(assign.left, ArrayIndexNode)
    assert isinstance(assign.left.array, IdentifierNode)
    assert isinstance(assign.left.index, IntLiteralNode)


def test_pretty_printer_outputs_non_empty_strings():
    src = "int x = 5; int add(int a, int b) { return a + b; }"
    ast = parse_tokens(lex(src))
    s = PrettyPrinter.print_ast(ast)
    assert isinstance(s, str)
    assert len(s) > 0
