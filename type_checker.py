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
        match node:
            case IntLiteralNode():
                return SymbolType.INT
            case BoolLiteralNode():
                return SymbolType.BOOL
            case IdentifierNode(symbol_type=st):
                if st is None:
                    raise SyntaxError(f"Type not resolved for identifier '{node.name}'")
                return st
            case BinaryOpNode(left=left, operator=op, right=right):
                left_type = TypeChecker.check_expression(left)
                right_type = TypeChecker.check_expression(right)

                # Arithmetic operators
                if op in ("+", "-", "*", "/", "%"):
                    if left_type == SymbolType.INT and right_type == SymbolType.INT:
                        return SymbolType.INT
                    raise SyntaxError(
                        f"Cannot apply '{op}' to types '{left_type}' and '{right_type}'"
                    )

                # Comparison operators
                if op in ("==", "!=", "<", ">", "<=", ">="):
                    if left_type == right_type:
                        return SymbolType.BOOL
                    raise SyntaxError(
                        f"Cannot compare types '{left_type}' and '{right_type}'"
                    )

                # Logical operators
                if op in ("&&", "||"):
                    if left_type == SymbolType.BOOL and right_type == SymbolType.BOOL:
                        return SymbolType.BOOL
                    raise SyntaxError(
                        f"Cannot apply logical '{op}' to non-boolean types"
                    )

                raise SyntaxError(f"Unknown operator: {op}")

            case UnaryOpNode(operator=op, right=right):
                expr_type = TypeChecker.check_expression(right)

                if op == "-" and expr_type == SymbolType.INT:
                    return SymbolType.INT
                if op == "!" and expr_type == SymbolType.BOOL:
                    return SymbolType.BOOL

                raise SyntaxError(
                    f"Cannot apply unary '{op}' to type '{expr_type}'"
                )

            case ArrayIndexNode(array=arr, index=idx):
                array_type = TypeChecker.check_expression(arr)
                index_type = TypeChecker.check_expression(idx)

                if array_type != SymbolType.INT_ARRAY:
                    raise SyntaxError(f"Cannot index non-array type '{array_type}'")

                if index_type != SymbolType.INT:
                    raise SyntaxError(
                        f"Array index must be integer, got '{index_type}'"
                    )

                return SymbolType.INT

            case AssignmentNode(left=left, right=right):
                left_type = TypeChecker.check_expression(left)
                right_type = TypeChecker.check_expression(right)

                if left_type != right_type:
                    raise SyntaxError(
                        f"Cannot assign type '{right_type}' to variable of type '{left_type}'"
                    )

                return left_type

            case FunctionCallNode(function=IdentifierNode(is_function=is_func, parameters=param_types, symbol_type=ret_type) as func, arguments=args):
                if not is_func:
                    raise SyntaxError(f"'{func.name}' is not a function")

                param_types = param_types or []
                if len(param_types) != len(args):
                    raise SyntaxError(
                        f"Function '{func.name}' expects {len(param_types)} args, got {len(args)}"
                    )

                for expected, arg in zip(param_types, args):
                    arg_type = TypeChecker.check_expression(arg)
                    if arg_type != expected:
                        raise SyntaxError(
                            f"Function '{func.name}' argument type mismatch: expected {expected}, got {arg_type}"
                        )

                if ret_type is None:
                    raise SyntaxError(f"Function '{func.name}' has unknown return type")
                return ret_type

            case _:
                raise SyntaxError(f"Cannot type check node type: {type(node)}")

    @staticmethod
    def check_statement(
        node: ASTNode, expected_return: Optional[SymbolType] = None
    ) -> None:
        """Check type correctness of a statement.

        expected_return: the expected return type if inside a function body, otherwise None.
        """
        match node:
            case VariableDeclarationNode(var_name=name, var_type=vtype, init_value=init):
                if init:
                    init_type = TypeChecker.check_expression(init)
                    if init_type != vtype and not (
                        vtype == SymbolType.INT_ARRAY and init_type == SymbolType.INT_ARRAY
                    ):
                        raise SyntaxError(
                            f"Type mismatch in initialization of '{name}': expected {vtype}, got {init_type}"
                        )

            case IfStatementNode(condition=cond, then_block=then_block, else_block=else_block):
                cond_type = TypeChecker.check_expression(cond)
                if cond_type != SymbolType.BOOL:
                    raise SyntaxError(f"If condition must be boolean, got {cond_type}")

                TypeChecker.check_statement(then_block, expected_return)
                if else_block:
                    TypeChecker.check_statement(else_block, expected_return)

            case WhileStatementNode(condition=cond, body=body):
                cond_type = TypeChecker.check_expression(cond)
                if cond_type != SymbolType.BOOL:
                    raise SyntaxError(f"While condition must be boolean, got {cond_type}")

                TypeChecker.check_statement(body, expected_return)

            case BlockNode(statements=stmts):
                for stmt in stmts:
                    TypeChecker.check_statement(stmt, expected_return)

            case ExpressionStatementNode(expression=expr):
                TypeChecker.check_expression(expr)

            case ReturnStatementNode(expression=None):
                if expected_return is None:
                    raise SyntaxError("Return statement outside of function")
                if expected_return != SymbolType.VOID:
                    raise SyntaxError(
                        f"Return without value in function returning {expected_return}"
                    )

            case ReturnStatementNode(expression=expr):
                if expected_return is None:
                    raise SyntaxError("Return statement outside of function")
                expr_type = TypeChecker.check_expression(expr)
                if expected_return == SymbolType.VOID:
                    raise SyntaxError("Void function cannot return a value")
                if expr_type != expected_return:
                    raise SyntaxError(
                        f"Return type mismatch: expected {expected_return}, got {expr_type}"
                    )

            case FunctionDeclarationNode(body=body, return_type=ret_type) if body is not None:
                TypeChecker.check_statement(body, ret_type)

            case _:
                # For other statement types, just check their expressions or ignore
                pass
