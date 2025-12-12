"""Pretty-printer for the AST.

Provides `PrettyPrinter.print_ast(node, indent, prefix)` which renders an
AST into a readable multi-line string. The printer is intentionally simple
and intended for debugging, tests and development rather than for producing
final source code.

Examples:
    PrettyPrinter.print_ast(program_node)
"""

from __future__ import annotations
from typing import Optional
from ast_nodes import *


class PrettyPrinter:
    @staticmethod
    def print_ast(node: ASTNode, indent: int = 0, prefix: str = "") -> str:
        """Pretty print AST and return as string."""
        lines = []
        indent_str = " " * indent

        if not isinstance(node, ASTNode):
            lines.append(f"{indent_str}{prefix}{node}")
            return "\n".join(lines)

        node_type = node.type

        match node_type:
            case NodeType.INT_LITERAL:
                if isinstance(node, IntLiteralNode):
                    lines.append(f"{indent_str}{prefix}IntLiteral({node.value})")

            case NodeType.BOOL_LITERAL:
                if isinstance(node, BoolLiteralNode):
                    lines.append(f"{indent_str}{prefix}BoolLiteral({node.value})")

            case NodeType.IDENTIFIER:
                if isinstance(node, IdentifierNode):
                    type_info = f", type={node.symbol_type}" if node.symbol_type else ""
                    if node.is_array:
                        type_info += " (array)"
                    lines.append(
                        f"{indent_str}{prefix}Identifier({node.name}{type_info})"
                    )

            case NodeType.BINARY_OP:
                if isinstance(node, BinaryOpNode):
                    lines.append(f"{indent_str}{prefix}BinaryOp({node.operator})")
                    lines.append(
                        PrettyPrinter.print_ast(node.left, indent + 2, "left: ")
                    )
                    lines.append(
                        PrettyPrinter.print_ast(node.right, indent + 2, "right: ")
                    )

            case NodeType.UNARY_OP:
                if isinstance(node, UnaryOpNode):
                    lines.append(f"{indent_str}{prefix}UnaryOp({node.operator})")
                    lines.append(PrettyPrinter.print_ast(node.right, indent + 2))

            case NodeType.ARRAY_INDEX:
                if isinstance(node, ArrayIndexNode):
                    lines.append(f"{indent_str}{prefix}ArrayIndex")
                    lines.append(
                        PrettyPrinter.print_ast(node.array, indent + 2, "array: ")
                    )
                    lines.append(
                        PrettyPrinter.print_ast(node.index, indent + 2, "index: ")
                    )

            case NodeType.FUNC_CALL:
                if isinstance(node, FunctionCallNode):
                    func_name = (
                        node.function.name
                        if isinstance(node.function, IdentifierNode)
                        else "anonymous"
                    )
                    lines.append(f"{indent_str}{prefix}FunctionCall({func_name})")
                    for i, arg in enumerate(node.arguments):
                        lines.append(
                            PrettyPrinter.print_ast(arg, indent + 4, f"arg[{i}]: ")
                        )

            case NodeType.FUNC_DECL:
                if isinstance(node, FunctionDeclarationNode):
                    args = ", ".join(
                        f"{n}: {t}" for n, t in zip(node.arg_names, node.arg_types)
                    )
                    lines.append(
                        f"{indent_str}{prefix}FunctionDecl({node.func_name} -> {node.return_type}, params=[{args}])"
                    )
                    if node.body:
                        lines.append(
                            PrettyPrinter.print_ast(node.body, indent + 4, "body: ")
                        )

            case NodeType.RETURN_STMT:
                if isinstance(node, ReturnStatementNode):
                    lines.append(f"{indent_str}{prefix}Return")
                    if node.expression:
                        lines.append(
                            PrettyPrinter.print_ast(
                                node.expression, indent + 2, "expr: "
                            )
                        )

            case NodeType.ASSIGNMENT:
                if isinstance(node, AssignmentNode):
                    lines.append(f"{indent_str}{prefix}Assignment")
                    lines.append(
                        PrettyPrinter.print_ast(node.left, indent + 2, "left: ")
                    )
                    lines.append(
                        PrettyPrinter.print_ast(node.right, indent + 2, "right: ")
                    )

            case NodeType.EXPR_STMT:
                if isinstance(node, ExpressionStatementNode):
                    lines.append(f"{indent_str}{prefix}ExpressionStatement")
                    lines.append(PrettyPrinter.print_ast(node.expression, indent + 2))

            case NodeType.WHILE_STMT:
                if isinstance(node, WhileStatementNode):
                    lines.append(f"{indent_str}{prefix}WhileStatement")
                    lines.append(
                        PrettyPrinter.print_ast(
                            node.condition, indent + 4, "condition: "
                        )
                    )
                    lines.append(
                        PrettyPrinter.print_ast(node.body, indent + 4, "body: ")
                    )

            case NodeType.IF_STMT:
                if isinstance(node, IfStatementNode):
                    lines.append(f"{indent_str}{prefix}IfStatement")
                    lines.append(
                        PrettyPrinter.print_ast(
                            node.condition, indent + 4, "condition: "
                        )
                    )
                    lines.append(
                        PrettyPrinter.print_ast(node.then_block, indent + 4, "then: ")
                    )
                    if node.else_block:
                        lines.append(
                            PrettyPrinter.print_ast(
                                node.else_block, indent + 4, "else: "
                            )
                        )

            case NodeType.BLOCK:
                if isinstance(node, BlockNode):
                    lines.append(f"{indent_str}{prefix}Block")
                    for i, stmt in enumerate(node.statements):
                        lines.append(
                            PrettyPrinter.print_ast(stmt, indent + 4, f"stmt[{i}]: ")
                        )

            case NodeType.VAR_DECL:
                if isinstance(node, VariableDeclarationNode):
                    init_str = f" = ..." if node.init_value else ""
                    lines.append(
                        f"{indent_str}{prefix}VarDecl({node.var_name}: {node.var_type}{init_str})"
                    )
                    if node.init_value:
                        lines.append(
                            PrettyPrinter.print_ast(
                                node.init_value, indent + 2, "init: "
                            )
                        )

            case NodeType.PROGRAM:
                if isinstance(node, ProgramNode):
                    lines.append(f"{indent_str}{prefix}Program")
                    for i, stmt in enumerate(node.statements):
                        lines.append(
                            PrettyPrinter.print_ast(stmt, indent + 4, f"stmt[{i}]: ")
                        )

            case _:
                lines.append(f"{indent_str}{prefix}Unknown node type: {node_type}")

        return "\n".join(line for line in lines if line)  # Remove empty lines
