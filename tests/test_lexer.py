from main import lex
from tokens import TokenType


def test_lexer_recognizes_keywords_and_punctuation():
    src = "int x = 5; void foo(); return;"
    tokens = lex(src)
    types = [t.type for t in tokens]

    assert TokenType.INT_TYPE in types
    assert TokenType.VOID_TYPE in types
    assert TokenType.IDENTIFIER in types
    assert TokenType.ASSIGN in types
    assert TokenType.SEMICOLON in types
    assert TokenType.RETURN in types
