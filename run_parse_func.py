from main import lex, parse_tokens
from pretty_printer import PrettyPrinter

s = "int add(int a, int b) { return a + b; }"
print("INPUT:", s)

tokens = lex(s)
print("TOKENS:", tokens)
ast = parse_tokens(tokens)
print("\nAST:")
print(PrettyPrinter.print_ast(ast))
