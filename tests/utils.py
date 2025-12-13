import sys
import os

from lexer import Lexer
from parser import Parser
from tokens import Token


def lex(text: str):
    """Return a list of tokens for the given source text."""
    return Lexer(text).tokenize()


def parse_tokens(tokens):
    """Parse a list of tokens into an AST node."""
    return Parser(tokens).parse()


def parse_text(text: str):
    """Convenience: lex+parse a source text into an AST."""
    return Parser(Lexer(text).tokenize()).parse()
