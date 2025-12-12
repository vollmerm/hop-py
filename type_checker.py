"""Type checking utilities for the simple C-like language.

This module provides a `TypeChecker` class with static methods to verify
the types of expressions and statements in the AST produced by the parser.

Responsibilities:
- Determine and validate expression types (literals, binary/unary ops,
  array indexing, assignments, function calls).
- Validate statements including variable declarations, control flow, blocks,
  and `return` statements (return type matches enclosing function signature).

The type checker raises `SyntaxError` on type mismatches.
"""

from __future__ import annotations
from typing import Optional
from ast_nodes import *
from symbols import SymbolType


class TypeChecker:
    @staticmethod
    def check_expression(
        node: ASTNode, expected_type: Optional[SymbolType] = None
    ) -> SymbolType:
        """Check type of expression and return its type."""
        match node.type:
            case NodeType.INT_LITERAL:
                return SymbolType.INT

            case NodeType.BOOL_LITERAL:
                return SymbolType.BOOL

            case NodeType.IDENTIFIER:
                if not isinstance(node, IdentifierNode):
                    raise TypeError(f"Expected IdentifierNode, got {type(node)}")
                if node.symbol_type is None:
                    raise SyntaxError(f"Type not resolved for identifier '{node.name}'")
                return node.symbol_type

            case NodeType.BINARY_OP:
                if not isinstance(node, BinaryOpNode):
                    raise TypeError(f"Expected BinaryOpNode, got {type(node)}")

                left_type = TypeChecker.check_expression(node.left)
                right_type = TypeChecker.check_expression(node.right)

                # Arithmetic operators
                if node.operator in ("+", "-", "*", "/", "%"):
                    if left_type == SymbolType.INT and right_type == SymbolType.INT:
                        return SymbolType.INT
                    raise SyntaxError(
                        f"Cannot apply '{node.operator}' to types '{left_type}' and '{right_type}'"
                    )

                # Comparison operators
                elif node.operator in ("==", "!=", "<", ">", "<=", ">="):
                    if left_type == right_type:
                        return SymbolType.BOOL
                    raise SyntaxError(
                        f"Cannot compare types '{left_type}' and '{right_type}'"
                    )

                # Logical operators
                elif node.operator in ("&&", "||"):
                    if left_type == SymbolType.BOOL and right_type == SymbolType.BOOL:
                        return SymbolType.BOOL
                    raise SyntaxError(
                        f"Cannot apply logical '{node.operator}' to non-boolean types"
                    )

                else:
                    raise SyntaxError(f"Unknown operator: {node.operator}")

            case NodeType.UNARY_OP:
                if not isinstance(node, UnaryOpNode):
                    raise TypeError(f"Expected UnaryOpNode, got {type(node)}")

                expr_type = TypeChecker.check_expression(node.right)

                if node.operator == "-" and expr_type == SymbolType.INT:
                    return SymbolType.INT
                elif node.operator == "!" and expr_type == SymbolType.BOOL:
                    return SymbolType.BOOL

                raise SyntaxError(
                    f"Cannot apply unary '{node.operator}' to type '{expr_type}'"
                )

            case NodeType.ARRAY_INDEX:
                if not isinstance(node, ArrayIndexNode):
                    raise TypeError(f"Expected ArrayIndexNode, got {type(node)}")

                array_type = TypeChecker.check_expression(node.array)
                index_type = TypeChecker.check_expression(node.index)

                if array_type != SymbolType.INT_ARRAY:
                    raise SyntaxError(f"Cannot index non-array type '{array_type}'")

                if index_type != SymbolType.INT:
                    raise SyntaxError(
                        f"Array index must be integer, got '{index_type}'"
                    )

                return SymbolType.INT

            case NodeType.ASSIGNMENT:
                if not isinstance(node, AssignmentNode):
                    raise TypeError(f"Expected AssignmentNode, got {type(node)}")

                left_type = TypeChecker.check_expression(node.left)
                right_type = TypeChecker.check_expression(node.right)

                if left_type != right_type:
                    raise SyntaxError(
                        f"Cannot assign type '{right_type}' to variable of type '{left_type}'"
                    )

                return left_type

            case NodeType.FUNC_CALL:
                if not isinstance(node, FunctionCallNode):
                    raise TypeError(f"Expected FunctionCallNode, got {type(node)}")

                # Resolve function signature from the function expression if available
                func = node.function
                if isinstance(func, IdentifierNode):
                    if not func.is_function:
                        raise SyntaxError(f"'{func.name}' is not a function")

                    param_types = func.parameters or []
                    # Check argument count
                    if len(param_types) != len(node.arguments):
                        raise SyntaxError(
                            f"Function '{func.name}' expects {len(param_types)} args, got {len(node.arguments)}"
                        )

                    # Check each argument type
                    for expected, arg in zip(param_types, node.arguments):
                        arg_type = TypeChecker.check_expression(arg)
                        if arg_type != expected:
                            raise SyntaxError(
                                f"Function '{func.name}' argument type mismatch: expected {expected}, got {arg_type}"
                            )

                    # Return function's return type
                    if func.symbol_type is None:
                        raise SyntaxError(
                            f"Function '{func.name}' has unknown return type"
                        )
                    return func.symbol_type

                else:
                    raise SyntaxError("Can only call functions by identifier")

            case _:
                raise SyntaxError(f"Cannot type check node type: {node.type}")

    @staticmethod
    def check_statement(
        node: ASTNode, expected_return: Optional[SymbolType] = None
    ) -> None:
        """Check type correctness of a statement.

        expected_return: the expected return type if inside a function body, otherwise None.
        """
        match node.type:
            case NodeType.VAR_DECL:
                if not isinstance(node, VariableDeclarationNode):
                    raise TypeError(
                        f"Expected VariableDeclarationNode, got {type(node)}"
                    )

                if node.init_value:
                    init_type = TypeChecker.check_expression(node.init_value)
                    if init_type != node.var_type and not (
                        node.var_type == SymbolType.INT_ARRAY
                        and init_type == SymbolType.INT_ARRAY
                    ):
                        raise SyntaxError(
                            f"Type mismatch in initialization of '{node.var_name}': "
                            f"expected {node.var_type}, got {init_type}"
                        )

            case NodeType.IF_STMT:
                if not isinstance(node, IfStatementNode):
                    raise TypeError(f"Expected IfStatementNode, got {type(node)}")

                cond_type = TypeChecker.check_expression(node.condition)
                if cond_type != SymbolType.BOOL:
                    raise SyntaxError(f"If condition must be boolean, got {cond_type}")

                TypeChecker.check_statement(node.then_block, expected_return)
                if node.else_block:
                    TypeChecker.check_statement(node.else_block, expected_return)

            case NodeType.WHILE_STMT:
                if not isinstance(node, WhileStatementNode):
                    raise TypeError(f"Expected WhileStatementNode, got {type(node)}")

                cond_type = TypeChecker.check_expression(node.condition)
                if cond_type != SymbolType.BOOL:
                    raise SyntaxError(
                        f"While condition must be boolean, got {cond_type}"
                    )

                TypeChecker.check_statement(node.body, expected_return)

            case NodeType.BLOCK:
                if not isinstance(node, BlockNode):
                    raise TypeError(f"Expected BlockNode, got {type(node)}")

                for stmt in node.statements:
                    TypeChecker.check_statement(stmt, expected_return)

            case NodeType.EXPR_STMT:
                if not isinstance(node, ExpressionStatementNode):
                    raise TypeError(
                        f"Expected ExpressionStatementNode, got {type(node)}"
                    )

                TypeChecker.check_expression(node.expression)

            case NodeType.RETURN_STMT:
                if not isinstance(node, ReturnStatementNode):
                    raise TypeError(f"Expected ReturnStatementNode, got {type(node)}")

                if expected_return is None:
                    raise SyntaxError("Return statement outside of function")

                # No expression in return: only allowed for void
                if node.expression is None:
                    if expected_return != SymbolType.VOID:
                        raise SyntaxError(
                            f"Return without value in function returning {expected_return}"
                        )
                else:
                    expr_type = TypeChecker.check_expression(node.expression)
                    if expected_return == SymbolType.VOID:
                        raise SyntaxError("Void function cannot return a value")
                    if expr_type != expected_return:
                        raise SyntaxError(
                            f"Return type mismatch: expected {expected_return}, got {expr_type}"
                        )

            case NodeType.FUNC_DECL:
                if not isinstance(node, FunctionDeclarationNode):
                    raise TypeError(
                        f"Expected FunctionDeclarationNode, got {type(node)}"
                    )

                # If function has a body, type-check its statements with the function's return type
                if node.body:
                    TypeChecker.check_statement(node.body, node.return_type)

            case _:
                # For other statement types, just check their expressions
                pass
