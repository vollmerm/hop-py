from __future__ import annotations
from typing import List
from lexer import Lexer
from tokens import Token
from ast_nodes import ASTNode, NodeType
from parser import Parser
from type_checker import TypeChecker
from pretty_printer import PrettyPrinter


def lex(text: str) -> List[Token]:
    """Tokenize input string."""
    lexer = Lexer(text)
    return lexer.tokenize()


def parse_tokens(tokens: List[Token]) -> ASTNode:
    """Parse tokens into AST."""
    parser = Parser(tokens)
    return parser.parse()


def parse_string(text: str) -> ASTNode:
    """Parse string directly into AST."""
    tokens = lex(text)
    return parse_tokens(tokens)


def test_compiler_pipeline() -> None:
    """Test the complete compiler pipeline."""
    test_cases = [
        # Simple assignment statement
        "int x = 5;",
        # While loop
        "int i; while (i < 10) { i = i + 1; }",
        # If statement
        "int x; int y; if (x > 0) { y = 1; } else { y = -1; }",
        # Complex program
        """
        int i = 0;
        int sum = 0;
        while (i < 10) {
            if (i % 2 == 0) {
                sum = sum + i;
            }
            i = i + 1;
        }
        """,
        # Array operations
        "int[] arr; arr[0] = 5;",
        # Boolean operations
        """
        bool flag = true;
        int i = 5;
        bool result = flag && (i < 10);
        """,
        # Nested if-else
        """
        int a; int b; int max = 0;
        if (a > b) {
            max = a;
        } else if (a < b) {
            max = b;
        } else {
            max = 0;
        }
        """,
    ]

    print("Testing compiler pipeline:")
    print("=" * 80)

    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print("-" * 40)
        print(f"Input:\n{test.strip()}")
        print("-" * 40)

        try:
            # Tokenize
            tokens = lex(test)
            print(f"Tokens ({len(tokens)}):")
            for j, token in enumerate(tokens[:10]):  # Show first 10 tokens
                print(f"  {j:2}: {token}")
            if len(tokens) > 10:
                print(f"  ... and {len(tokens) - 10} more")

            # Parse
            ast = parse_tokens(tokens)
            print("\nAST:")
            print(PrettyPrinter.print_ast(ast))

            # Type check
            try:
                if ast.type == NodeType.PROGRAM:
                    for stmt in ast.statements:
                        TypeChecker.check_statement(stmt)
                else:
                    TypeChecker.check_statement(ast)
                print("\n✓ Type check passed")
            except SyntaxError as e:
                print(f"\n✗ Type error: {e}")

        except SyntaxError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback

            traceback.print_exc()


def interactive_mode() -> None:
    """Run interactive compiler mode."""
    print("\nInteractive Compiler Mode (type 'quit' to exit)")
    print("=" * 80)

    while True:
        try:
            text = input("\nEnter program or expression: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not text:
                continue

            # Tokenize
            tokens = lex(text)
            print(f"\nTokens ({len(tokens)}):")
            for i, token in enumerate(tokens):
                print(f"  {i:3}: {token}")

            # Parse
            ast = parse_tokens(tokens)
            print("\nAST:")
            print(PrettyPrinter.print_ast(ast))

            # Type check
            try:
                TypeChecker.check_statement(ast)
                print("\n✓ Type check passed")
            except SyntaxError as e:
                print(f"\n✗ Type error: {e}")

        except SyntaxError as e:
            print(f"Syntax error: {e}")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    test_compiler_pipeline()
    # interactive_mode()
