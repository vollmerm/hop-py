"""Convert AST nodes into JSON-serializable structures.

This module provides `ast_to_json(node)` which returns a nested structure
of dicts/lists/primitives describing the AST node. It's intentionally
simple and conservative — it encodes the node type and key fields.
"""

from typing import Any, Dict, List, Optional
from ast_nodes import *


def ast_to_json(node: Optional[ASTNode]) -> Any:
    if node is None:
        return None

    t = node.type
    # literals
    if t == NodeType.INT_LITERAL and isinstance(node, IntLiteralNode):
        return {"node_type": "IntLiteral", "value": node.value}
    if t == NodeType.BOOL_LITERAL and isinstance(node, BoolLiteralNode):
        return {"node_type": "BoolLiteral", "value": node.value}
    if t == NodeType.IDENTIFIER and isinstance(node, IdentifierNode):
        return {
            "node_type": "Identifier",
            "name": node.name,
            "symbol_type": (
                str(node.symbol_type) if node.symbol_type is not None else None
            ),
            "is_array": getattr(node, "is_array", False),
        }
    # expressions
    if t == NodeType.BINARY_OP and isinstance(node, BinaryOpNode):
        return {
            "node_type": "BinaryOp",
            "operator": node.operator,
            "left": ast_to_json(node.left),
            "right": ast_to_json(node.right),
        }
    if t == NodeType.UNARY_OP and isinstance(node, UnaryOpNode):
        return {
            "node_type": "UnaryOp",
            "operator": node.operator,
            "right": ast_to_json(node.right),
        }
    if t == NodeType.ARRAY_INDEX and isinstance(node, ArrayIndexNode):
        return {
            "node_type": "ArrayIndex",
            "array": ast_to_json(node.array),
            "index": ast_to_json(node.index),
        }
    if t == NodeType.FUNC_CALL and isinstance(node, FunctionCallNode):
        return {
            "node_type": "FunctionCall",
            "function": ast_to_json(node.function),
            "arguments": [ast_to_json(a) for a in node.arguments],
        }
    if t == NodeType.ASSIGNMENT and isinstance(node, AssignmentNode):
        return {
            "node_type": "Assignment",
            "left": ast_to_json(node.left),
            "right": ast_to_json(node.right),
        }
    # statements and higher-level nodes
    if t == NodeType.EXPR_STMT and isinstance(node, ExpressionStatementNode):
        return {"node_type": "ExprStmt", "expression": ast_to_json(node.expression)}
    if t == NodeType.VAR_DECL and isinstance(node, VariableDeclarationNode):
        return {
            "node_type": "VarDecl",
            "var_name": node.var_name,
            "var_type": node.var_type,
            "init_value": (
                ast_to_json(node.init_value) if node.init_value is not None else None
            ),
        }
    if t == NodeType.RETURN_STMT and isinstance(node, ReturnStatementNode):
        return {"node_type": "Return", "expression": ast_to_json(node.expression)}
    if t == NodeType.IF_STMT and isinstance(node, IfStatementNode):
        return {
            "node_type": "If",
            "condition": ast_to_json(node.condition),
            "then": ast_to_json(node.then_block),
            "else": (
                ast_to_json(node.else_block) if node.else_block is not None else None
            ),
        }
    if t == NodeType.WHILE_STMT and isinstance(node, WhileStatementNode):
        return {
            "node_type": "While",
            "condition": ast_to_json(node.condition),
            "body": ast_to_json(node.body),
        }
    if t == NodeType.BLOCK and isinstance(node, BlockNode):
        return {
            "node_type": "Block",
            "statements": [ast_to_json(s) for s in node.statements],
        }
    if t == NodeType.FUNC_DECL and isinstance(node, FunctionDeclarationNode):
        return {
            "node_type": "FunctionDecl",
            "func_name": node.func_name,
            "return_type": node.return_type,
            "arg_types": list(node.arg_types) if node.arg_types is not None else [],
            "arg_names": list(node.arg_names) if node.arg_names is not None else [],
            "body": ast_to_json(node.body),
        }
    if t == NodeType.PROGRAM and isinstance(node, ProgramNode):
        return {
            "node_type": "Program",
            "statements": [ast_to_json(s) for s in node.statements],
        }

    # Fallback: try to serialize accessible fields
    data: Dict[str, Any] = {"node_type": getattr(t, "name", str(t))}
    for k, v in getattr(node, "__dict__", {}).items():
        if isinstance(v, ASTNode):
            data[k] = ast_to_json(v)
        elif isinstance(v, list):
            data[k] = [ast_to_json(x) if isinstance(x, ASTNode) else x for x in v]
        else:
            # primitives
            try:
                import json as _json

                _json.dumps(v)
                data[k] = v
            except Exception:
                data[k] = str(v)
    return data
