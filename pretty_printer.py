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
    def print_instr_cfg(cfg: dict) -> str:
        """Pretty print a CFG after instruction selection (lists of instructions per statement)."""
        lines = []
        blocks = cfg.get("blocks", [])
        entry = cfg.get("entry")
        exit_ = cfg.get("exit")
        lines.append(f"InstrCFG: entry={entry} exit={exit_}")
        for block in blocks:
            lbl = block["label"]
            lines.append(f"  Block {lbl}:")
            stmts = block.get("statements", [])
            for i, instrs in enumerate(stmts):
                if not instrs:
                    continue
                lines.append(f"    Statement {i}:")
                for instr in instrs:
                    # Format Register objects and dicts nicely
                    def fmt_val(val):
                        if isinstance(val, list):
                            return [fmt_val(v) for v in val]
                        if hasattr(val, "name") and hasattr(val, "is_virtual"):
                            # Register
                            return f"{val.name}{' (v)' if getattr(val, 'is_virtual', False) else ''}"
                        return repr(val)

                    # Special-case some ops for clearer output
                    op = instr.get("op")
                    if op == "FUNC_LABEL":
                        name = instr.get("name")
                        lines.append(f"      FUNC {name}:")
                        continue
                    if op == "CALL":
                        fname = instr.get("func")
                        lines.append(f"      CALL {fname}")
                        continue
                    if op == "RET":
                        lines.append(f"      RET")
                        continue

                    parts = []
                    for k, v in instr.items():
                        if k == "op":
                            parts.append(str(v))
                        else:
                            parts.append(f"{k}={fmt_val(v)}")
                    instr_str = ", ".join(parts)
                    lines.append(f"      {instr_str}")
            out_edges = block.get("out_edges", [])
            lines.append(f"    Out edges: {out_edges}")
        return "\n".join(lines)

    @staticmethod
    def print_cfg(
        cfg: dict, liveness: Optional[dict] = None, collapse_liveness: bool = False
    ) -> str:
        """Pretty print a CFG (as produced by build_cfg) and return as string.

        If `liveness` is provided (mapping produced by `analyze_liveness`), the
        printer will annotate each statement with its `live_in` and `live_out`
        sets.
        """
        lines = []
        blocks = cfg.get("blocks", [])
        entry = cfg.get("entry")
        exit_ = cfg.get("exit")
        lines.append(f"CFG: entry={entry} exit={exit_}")
        for block in blocks:
            lbl = block["label"]
            lines.append(f"  Block {lbl}:")
            stmts = block.get("statements", [])
            i = 0
            instr_l = []
            if liveness and lbl in liveness:
                instr_l = liveness[lbl].get("instr_liveness", [])

            def _fmt(s: set) -> str:
                if not s:
                    return "{}"
                return "{" + ", ".join(sorted(s)) + "}"

            while i < len(stmts):
                stmt = stmts[i]
                ast_str = PrettyPrinter.print_ast(stmt, indent=6)
                for l in ast_str.splitlines():
                    lines.append(f"    {l}")

                # If we have liveness info and collapsing is requested, find a run of
                # contiguous instructions that share identical liveness, and print
                # the live sets once for the entire group.
                if instr_l and i < len(instr_l):
                    live_in, live_out = instr_l[i]
                    if collapse_liveness:
                        j = i + 1
                        while j < len(instr_l) and instr_l[j] == instr_l[i]:
                            j += 1
                        # print live sets once for the group [i, j)
                        lines.append(f"      live_in: {_fmt(live_in)}")
                        lines.append(f"      live_out: {_fmt(live_out)}")
                        if j - i > 1:
                            lines.append(f"      (applies to {j-i} instructions above)")
                        i = j
                        continue
                    else:
                        lines.append(f"      live_in: {_fmt(live_in)}")
                        lines.append(f"      live_out: {_fmt(live_out)}")

                i += 1
            out_edges = block.get("out_edges", [])
            lines.append(f"    Out edges: {out_edges}")
        return "\n".join(lines)

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

    @staticmethod
    def print_surface(node: ASTNode) -> str:
        """Return a compact, surface-syntax-like one-line representation of an AST node.

        This is intended for use in visualizations (CFG cells) where a concise
        expression/statement syntax is desirable (e.g. `tmp0 = tmp1 + tmp2`).
        """
        if node is None:
            return ""

        # Helpers
        def _p(n: ASTNode) -> str:
            return PrettyPrinter.print_surface(n) if isinstance(n, ASTNode) else str(n)

        nt = node.type if isinstance(node, ASTNode) else None

        match nt:
            case NodeType.INT_LITERAL:
                return str(node.value)
            case NodeType.BOOL_LITERAL:
                return "true" if node.value else "false"
            case NodeType.IDENTIFIER:
                return node.name
            case NodeType.BINARY_OP:
                # left op right
                return f"{_p(node.left)} {node.operator} {_p(node.right)}"
            case NodeType.UNARY_OP:
                return f"{node.operator}{_p(node.right)}"
            case NodeType.ARRAY_INDEX:
                return f"{_p(node.array)}[{_p(node.index)}]"
            case NodeType.FUNC_CALL:
                fname = (
                    node.function.name
                    if isinstance(node.function, IdentifierNode)
                    else _p(node.function)
                )
                args = ", ".join(_p(a) for a in node.arguments)
                return f"{fname}({args})"
            case NodeType.ASSIGNMENT:
                return f"{_p(node.left)} = {_p(node.right)}"
            case NodeType.EXPR_STMT:
                return _p(node.expression)
            case NodeType.RETURN_STMT:
                if node.expression:
                    return f"return {_p(node.expression)}"
                return "return"
            case NodeType.VAR_DECL:
                tname = (
                    getattr(node.var_type, "name", str(node.var_type)).lower()
                    if node.var_type is not None
                    else "var"
                )
                if node.init_value:
                    return f"{tname} {node.var_name} = {_p(node.init_value)}"
                return f"{tname} {node.var_name}"
            case NodeType.WHILE_STMT:
                return f"while ({_p(node.condition)})"
            case NodeType.IF_STMT:
                return f"if ({_p(node.condition)})"
            case NodeType.FUNC_DECL:
                args = ", ".join(node.arg_names) if node.arg_names else ""
                return f"func {node.func_name}({args})"
            case NodeType.BLOCK:
                return "{...}"
            case NodeType.PROGRAM:
                return "<program>"
            case _:
                # Fallback to the verbose AST printer but collapse to single line
                s = PrettyPrinter.print_ast(node)
                return " ".join(line.strip() for line in s.splitlines())
