"""Symbol table and symbol representations.

This module defines the `SymbolType` enum, a `Symbol` dataclass to represent
variables and functions, and `SymbolTable` which supports nested scopes via
an optional parent link. The `SymbolTable` API provides `declare`, `lookup`,
and existence checks used by the parser and type checker.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict


class SymbolType(Enum):
    INT = auto()
    BOOL = auto()
    INT_ARRAY = auto()
    VOID = auto()

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class Symbol:
    name: str
    type: SymbolType
    is_array: bool = False
    is_function: bool = False
    value: Optional[int | bool] = None
    parameters: Optional[list[SymbolType]] = None

    def __repr__(self) -> str:
        return f"Symbol({self.name}, {self.type}, is_array={self.is_array}, is_function={self.is_function})"


class SymbolTable:
    def __init__(self, parent: Optional[SymbolTable] = None):
        self.symbols: Dict[str, Symbol] = {}
        self.parent = parent

    def declare(
        self,
        name: str,
        type_: SymbolType,
        is_array: bool = False,
        is_function: bool = False,
        value: Optional[int | bool] = None,
    ) -> Symbol:
        """Declare a new variable in the current scope."""
        if name in self.symbols:
            raise SyntaxError(f"Variable '{name}' already declared in this scope")

        symbol = Symbol(name, type_, is_array, is_function, value)
        self.symbols[name] = symbol
        return symbol

    def lookup(self, name: str) -> Symbol:
        """Look up a variable in the current and parent scopes."""
        if name in self.symbols:
            return self.symbols[name]
        elif self.parent:
            return self.parent.lookup(name)
        else:
            raise SyntaxError(f"Undeclared variable '{name}'")

    def exists_in_current_scope(self, name: str) -> bool:
        """Check if variable is declared in current scope only."""
        return name in self.symbols

    def exists(self, name: str) -> bool:
        """Check if variable is declared in any scope."""
        if name in self.symbols:
            return True
        elif self.parent:
            return self.parent.exists(name)
        return False
