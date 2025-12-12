"""AST node definitions for the simple C-like language.

This module defines the concrete AST node dataclasses used by the parser and
subsequent compiler phases. Each node is represented by a dataclass that
carries the relevant information (e.g. an operator, child nodes, names,
types). The `NodeType` enum identifies node kinds and is used by the
pretty-printer and type checker.

Conventions:
- All AST node dataclasses inherit from `ASTNode` which records the node
    kind (`NodeType`) and optional source `line`/`column` information.
- Expression nodes vs statement nodes are separated by purpose but are plain
    dataclasses so the rest of the toolchain can pattern-match on `node.type`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union, List
from symbols import SymbolType


class NodeType(Enum):
    INT_LITERAL = auto()
    BOOL_LITERAL = auto()
    IDENTIFIER = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    ARRAY_INDEX = auto()
    FUNC_CALL = auto()
    ASSIGNMENT = auto()
    EXPR_STMT = auto()
    WHILE_STMT = auto()
    IF_STMT = auto()
    BLOCK = auto()
    VAR_DECL = auto()
    FUNC_DECL = auto()
    RETURN_STMT = auto()
    PROGRAM = auto()

    def __str__(self) -> str:
        return self.name


# Base AST Node
@dataclass
class ASTNode:
    type: NodeType
    line: int = 0
    column: int = 0


# Expression Nodes
@dataclass
class LiteralNode(ASTNode):
    value: Union[int, bool] = 0


@dataclass
class IntLiteralNode(LiteralNode):
    type: NodeType = NodeType.INT_LITERAL


@dataclass
class BoolLiteralNode(LiteralNode):
    type: NodeType = NodeType.BOOL_LITERAL


@dataclass
class IdentifierNode(ASTNode):
    type: NodeType = NodeType.IDENTIFIER
    name: str = ""
    symbol_type: Optional[SymbolType] = None
    is_array: bool = False
    # For functions, flag and parameter types (populated by parser when available)
    is_function: bool = False
    parameters: List[SymbolType] = field(default_factory=list)


@dataclass
class BinaryOpNode(ASTNode):
    type: NodeType = NodeType.BINARY_OP
    left: ASTNode = field(default_factory=lambda: IntLiteralNode())
    operator: str = ""
    right: ASTNode = field(default_factory=lambda: IntLiteralNode())


@dataclass
class UnaryOpNode(ASTNode):
    type: NodeType = NodeType.UNARY_OP
    operator: str = ""
    right: ASTNode = field(default_factory=lambda: IntLiteralNode())


@dataclass
class ArrayIndexNode(ASTNode):
    type: NodeType = NodeType.ARRAY_INDEX
    array: IdentifierNode = field(default_factory=lambda: IdentifierNode())
    index: ASTNode = field(default_factory=lambda: IntLiteralNode())


@dataclass
class FunctionCallNode(ASTNode):
    type: NodeType = NodeType.FUNC_CALL
    function: ASTNode = field(default_factory=lambda: IdentifierNode())
    arguments: List[ASTNode] = field(default_factory=list)


@dataclass
class AssignmentNode(ASTNode):
    type: NodeType = NodeType.ASSIGNMENT
    left: ASTNode = field(default_factory=lambda: IdentifierNode())
    right: ASTNode = field(default_factory=lambda: IntLiteralNode())


# Statement Nodes
@dataclass
class ExpressionStatementNode(ASTNode):
    type: NodeType = NodeType.EXPR_STMT
    expression: ASTNode = field(default_factory=lambda: IntLiteralNode())


@dataclass
class BlockNode(ASTNode):
    type: NodeType = NodeType.BLOCK
    statements: List[ASTNode] = field(default_factory=list)


@dataclass
class WhileStatementNode(ASTNode):
    type: NodeType = NodeType.WHILE_STMT
    condition: ASTNode = field(default_factory=lambda: BoolLiteralNode())
    body: ASTNode = field(default_factory=lambda: BlockNode())


@dataclass
class IfStatementNode(ASTNode):
    type: NodeType = NodeType.IF_STMT
    condition: ASTNode = field(default_factory=lambda: BoolLiteralNode())
    then_block: ASTNode = field(default_factory=lambda: BlockNode())
    else_block: Optional[ASTNode] = None


# Declaration Nodes
@dataclass
class VariableDeclarationNode(ASTNode):
    type: NodeType = NodeType.VAR_DECL
    var_name: str = ""
    var_type: SymbolType = SymbolType.INT
    init_value: Optional[ASTNode] = None


@dataclass
class FunctionDeclarationNode(ASTNode):
    type: NodeType = NodeType.FUNC_DECL
    func_name: str = ""
    return_type: SymbolType = SymbolType.INT
    arg_types: List[SymbolType] = field(default_factory=list)
    arg_names: List[str] = field(default_factory=list)
    body: Optional[BlockNode] = None


@dataclass
class ReturnStatementNode(ASTNode):
    type: NodeType = NodeType.RETURN_STMT
    expression: Optional[ASTNode] = None


# Program Node
@dataclass
class ProgramNode(ASTNode):
    type: NodeType = NodeType.PROGRAM
    statements: List[ASTNode] = field(default_factory=list)
