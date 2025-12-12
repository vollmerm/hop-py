"""Token definitions for the lexer.

This module defines the `TokenType` enum for all token kinds recognized by
the lexer and a small `Token` dataclass that holds a token type and an
optional lexeme/value. Tokens are the atomic units produced by the lexer and
consumed by the parser.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class TokenType(Enum):
    # Literals
    INTEGER = auto()
    IDENTIFIER = auto()

    # Arithmetic operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    MOD = auto()
    SLASH = auto()

    # Parentheses and brackets
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()

    # Punctuation
    COMMA = auto()
    ASSIGN = auto()
    SEMICOLON = auto()

    # Comparison operators
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()

    # Logical operators
    AND = auto()
    OR = auto()
    NOT = auto()

    # Keywords
    WHILE = auto()
    IF = auto()
    ELSE = auto()
    RETURN = auto()
    INT_TYPE = auto()
    BOOL_TYPE = auto()
    VOID_TYPE = auto()
    TRUE = auto()
    FALSE = auto()

    # Special
    EOF = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class Token:
    type: TokenType
    value: Optional[str | int | bool] = None

    def __repr__(self) -> str:
        return f"Token({self.type}, {repr(self.value)})"

    @property
    def lexeme(self) -> str:
        if self.value is None:
            return str(self.type)
        return str(self.value)
