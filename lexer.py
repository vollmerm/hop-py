"""
Lexer for the simple C-like language.

Overview:
- This module implements a small hand-written lexical analyzer (scanner) that
    transforms an input source string into a stream of `Token` objects defined
    in `tokens.py`.
- It recognizes keywords (e.g. `int`, `bool`, `void`, `return`, `if`,
    `while`), identifiers, integer literals, single- and two-character
    operators (e.g. `==`, `!=`, `<=`, `>=`, `&&`, `||`), punctuation (commas,
    semicolons, parentheses/braces/brackets) and skips whitespace and
    single-line comments starting with `//`.

Examples:
    Input:  "int add(int a, int b) { return a + b; }"
    Tokens: [INT_TYPE, IDENTIFIER('add'), LPAREN, INT_TYPE, IDENTIFIER('a'), ...]

Implementation notes:
- The lexer is a simple stateful scanner using `self.pos` and `self.current_char`.
- Two-character operators are checked first (e.g. `==`, `!=`, `<=`, `>=`,
    `&&`, `||`) to avoid splitting them into two tokens.
- Identifiers are scanned and then mapped to keywords using `self.keywords`.
- Integer literals are parsed by consuming consecutive digits.

This design favors clarity and small size over extreme performance.
"""

from __future__ import annotations
from typing import Optional, List
from tokens import Token, TokenType


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.current_char = self.text[self.pos] if self.text else None

        self.keywords = {
            "while": TokenType.WHILE,
            "if": TokenType.IF,
            "else": TokenType.ELSE,
            "return": TokenType.RETURN,
            "void": TokenType.VOID_TYPE,
            "int": TokenType.INT_TYPE,
            "bool": TokenType.BOOL_TYPE,
            "true": TokenType.TRUE,
            "false": TokenType.FALSE,
        }

    def error(self, message: str = "") -> SyntaxError:
        msg = f"Lexical error at line {self.line}, column {self.column}: {message}"
        return SyntaxError(msg)

    def advance(self) -> None:
        """Advance to next character."""
        # Maintain `self.pos`, `self.current_char`, and update `line`/`column`
        # counters. Newlines reset the column and increment the line number.
        if self.current_char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def peek_char(self) -> Optional[str]:
        """Look at next character without consuming it."""
        next_pos = self.pos + 1
        if next_pos < len(self.text):
            return self.text[next_pos]
        return None

    def skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self) -> None:
        """Skip single-line comments (// ...)."""
        # We know current_char is '/' and peek_char is '/'
        # Called when the lexer sees `//`. Consumes characters until the end of
        # the line (or EOF). This lexer only supports single-line comments.

        while self.current_char is not None and self.current_char != "\n":
            self.advance()

        if self.current_char == "\n":
            self.advance()

    def integer(self) -> int:
        """Parse a multi-digit integer."""
        result = []
        start_col = self.column

        # Consume a run of digits to form the integer literal.
        while self.current_char is not None and self.current_char.isdigit():
            result.append(self.current_char)
            self.advance()

        if not result:
            raise self.error("Expected integer")

        return int("".join(result))

    def identifier(self) -> str:
        """Parse an identifier or keyword."""
        result = []

        # First character must be a letter or underscore (C-like rule)
        if self.current_char is not None and (
            self.current_char.isalpha() or self.current_char == "_"
        ):
            result.append(self.current_char)
            self.advance()
        else:
            raise self.error("Expected identifier")

        # Following characters can be letters, digits, or underscores.
        while self.current_char is not None and (
            self.current_char.isalnum() or self.current_char == "_"
        ):
            result.append(self.current_char)
            self.advance()

        return "".join(result)

    def get_next_token(self) -> Token:
        """Lexical analyzer that returns tokens one at a time."""
        while self.current_char is not None:
            # Skip whitespace
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            # Skip comments
            if self.current_char == "/" and self.peek_char() == "/":
                self.skip_comment()
                continue

            # Handle two-character operators first so `==` is not lexed as `=` `=`.
            if self.current_char == "=" and self.peek_char() == "=":
                self.advance()
                self.advance()
                return Token(TokenType.EQ, "==")

            if self.current_char == "!" and self.peek_char() == "=":
                self.advance()
                self.advance()
                return Token(TokenType.NEQ, "!=")

            if self.current_char == "<" and self.peek_char() == "=":
                self.advance()
                self.advance()
                return Token(TokenType.LTE, "<=")

            if self.current_char == ">" and self.peek_char() == "=":
                self.advance()
                self.advance()
                return Token(TokenType.GTE, ">=")

            if self.current_char == "&" and self.peek_char() == "&":
                self.advance()
                self.advance()
                return Token(TokenType.AND, "&&")

            if self.current_char == "|" and self.peek_char() == "|":
                self.advance()
                self.advance()
                return Token(TokenType.OR, "||")

            # Single character tokens handled directly via structural matching.
            # This covers operators, punctuation and grouping symbols.
            match self.current_char:
                case "+":
                    self.advance()
                    return Token(TokenType.PLUS, "+")
                case "-":
                    self.advance()
                    return Token(TokenType.MINUS, "-")
                case "*":
                    self.advance()
                    return Token(TokenType.STAR, "*")
                case "%":
                    self.advance()
                    return Token(TokenType.MOD, "%")
                case "/":
                    self.advance()
                    return Token(TokenType.SLASH, "/")
                case "(":
                    self.advance()
                    return Token(TokenType.LPAREN, "(")
                case ")":
                    self.advance()
                    return Token(TokenType.RPAREN, ")")
                case "[":
                    self.advance()
                    return Token(TokenType.LBRACKET, "[")
                case "]":
                    self.advance()
                    return Token(TokenType.RBRACKET, "]")
                case "{":
                    self.advance()
                    return Token(TokenType.LBRACE, "{")
                case "}":
                    self.advance()
                    return Token(TokenType.RBRACE, "}")
                case ",":
                    self.advance()
                    return Token(TokenType.COMMA, ",")
                case ";":
                    self.advance()
                    return Token(TokenType.SEMICOLON, ";")
                case "=":
                    self.advance()
                    return Token(TokenType.ASSIGN, "=")
                case "<":
                    self.advance()
                    return Token(TokenType.LT, "<")
                case ">":
                    self.advance()
                    return Token(TokenType.GT, ">")
                case "!":
                    self.advance()
                    return Token(TokenType.NOT, "!")

            # Numbers: integer literals
            if self.current_char.isdigit():
                value = self.integer()
                return Token(TokenType.INTEGER, value)

            # Identifiers and keywords: scan an identifier and map to a
            # keyword token if present in `self.keywords`.
            if self.current_char.isalpha() or self.current_char == "_":
                ident = self.identifier()
                token_type = self.keywords.get(ident, TokenType.IDENTIFIER)
                return Token(token_type, ident)

            # If we reach here, the character is not recognized.
            raise self.error(f"Unexpected character '{self.current_char}'")

        return Token(TokenType.EOF, None)

    def tokenize(self) -> List[Token]:
        """Return all tokens from the input string."""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens
