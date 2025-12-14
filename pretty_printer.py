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

        match node:
            case IntLiteralNode(value=v):
                lines.append(f"{indent_str}{prefix}IntLiteral({v})")

            case BoolLiteralNode(value=v):
                lines.append(f"{indent_str}{prefix}BoolLiteral({v})")

            case IdentifierNode(name=n, symbol_type=st, is_array=is_arr):
                type_info = f", type={st}" if st else ""
                if is_arr:
                    type_info += " (array)"
                lines.append(f"{indent_str}{prefix}Identifier({n}{type_info})")

            case BinaryOpNode(left=left, operator=op, right=right):
                lines.append(f"{indent_str}{prefix}BinaryOp({op})")
                lines.append(PrettyPrinter.print_ast(left, indent + 2, "left: "))
                lines.append(PrettyPrinter.print_ast(right, indent + 2, "right: "))

            case UnaryOpNode(operator=op, right=right):
                lines.append(f"{indent_str}{prefix}UnaryOp({op})")
                lines.append(PrettyPrinter.print_ast(right, indent + 2))

            case ArrayIndexNode(array=arr, index=idx):
                lines.append(f"{indent_str}{prefix}ArrayIndex")
                lines.append(PrettyPrinter.print_ast(arr, indent + 2, "array: "))
                lines.append(PrettyPrinter.print_ast(idx, indent + 2, "index: "))

            case FunctionCallNode(function=func, arguments=args):
                func_name = (
                    func.name if isinstance(func, IdentifierNode) else "anonymous"
                )
                lines.append(f"{indent_str}{prefix}FunctionCall({func_name})")
                for i, arg in enumerate(args):
                    lines.append(PrettyPrinter.print_ast(arg, indent + 4, f"arg[{i}]: "))

            case FunctionDeclarationNode(func_name=name, arg_names=anames, arg_types=atypes, body=body):
                args = ", ".join(f"{n}: {t}" for n, t in zip(anames, atypes))
                lines.append(f"{indent_str}{prefix}FunctionDecl({name} -> {node.return_type}, params=[{args}])")
                if body:
                    lines.append(PrettyPrinter.print_ast(body, indent + 4, "body: "))

            case ReturnStatementNode(expression=expr):
                lines.append(f"{indent_str}{prefix}Return")
                if expr:
                    lines.append(PrettyPrinter.print_ast(expr, indent + 2, "expr: "))

            case AssignmentNode(left=left, right=right):
                lines.append(f"{indent_str}{prefix}Assignment")
                lines.append(PrettyPrinter.print_ast(left, indent + 2, "left: "))
                lines.append(PrettyPrinter.print_ast(right, indent + 2, "right: "))

            case ExpressionStatementNode(expression=expr):
                lines.append(f"{indent_str}{prefix}ExpressionStatement")
                lines.append(PrettyPrinter.print_ast(expr, indent + 2))

            case WhileStatementNode(condition=cond, body=body):
                lines.append(f"{indent_str}{prefix}WhileStatement")
                lines.append(PrettyPrinter.print_ast(cond, indent + 4, "condition: "))
                lines.append(PrettyPrinter.print_ast(body, indent + 4, "body: "))

            case IfStatementNode(condition=cond, then_block=then_b, else_block=else_b):
                lines.append(f"{indent_str}{prefix}IfStatement")
                lines.append(PrettyPrinter.print_ast(cond, indent + 4, "condition: "))
                lines.append(PrettyPrinter.print_ast(then_b, indent + 4, "then: "))
                if else_b:
                    lines.append(PrettyPrinter.print_ast(else_b, indent + 4, "else: "))

            case BlockNode(statements=stmts):
                lines.append(f"{indent_str}{prefix}Block")
                for i, stmt in enumerate(stmts):
                    lines.append(PrettyPrinter.print_ast(stmt, indent + 4, f"stmt[{i}]: "))

            case VariableDeclarationNode(var_name=vname, var_type=vtype, init_value=init):
                init_str = f" = ..." if init else ""
                lines.append(f"{indent_str}{prefix}VarDecl({vname}: {vtype}{init_str})")
                if init:
                    lines.append(PrettyPrinter.print_ast(init, indent + 2, "init: "))

            case ProgramNode(statements=stmts):
                lines.append(f"{indent_str}{prefix}Program")
                for i, stmt in enumerate(stmts):
                    lines.append(PrettyPrinter.print_ast(stmt, indent + 4, f"stmt[{i}]: "))

            case _:
                lines.append(f"{indent_str}{prefix}Unknown node type: {type(node)}")

        return "\n".join(line for line in lines if line)

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

        match node:
            case IntLiteralNode(value=v):
                return str(v)
            case BoolLiteralNode(value=v):
                return "true" if v else "false"
            case IdentifierNode(name=n):
                return n
            case BinaryOpNode(left=l, operator=op, right=r):
                return f"{_p(l)} {op} {_p(r)}"
            case UnaryOpNode(operator=op, right=right):
                return f"{op}{_p(right)}"
            case ArrayIndexNode(array=arr, index=idx):
                return f"{_p(arr)}[{_p(idx)}]"
            case FunctionCallNode(function=func, arguments=args):
                fname = func.name if isinstance(func, IdentifierNode) else _p(func)
                args_s = ", ".join(_p(a) for a in args)
                return f"{fname}({args_s})"
            case AssignmentNode(left=left, right=right):
                return f"{_p(left)} = {_p(right)}"
            case ExpressionStatementNode(expression=expr):
                return _p(expr)
            case ReturnStatementNode(expression=expr):
                if expr:
                    return f"return {_p(expr)}"
                return "return"
            case VariableDeclarationNode(var_type=vt, var_name=vn, init_value=init):
                tname = (
                    getattr(vt, "name", str(vt)).lower() if vt is not None else "var"
                )
                if init:
                    return f"{tname} {vn} = {_p(init)}"
                return f"{tname} {vn}"
            case WhileStatementNode(condition=cond):
                return f"while ({_p(cond)})"
            case IfStatementNode(condition=cond):
                return f"if ({_p(cond)})"
            case FunctionDeclarationNode(func_name=fn, arg_names=an):
                args = ", ".join(an) if an else ""
                return f"func {fn}({args})"
            case BlockNode():
                return "{...}"
            case ProgramNode():
                return "<program>"
            case _:
                s = PrettyPrinter.print_ast(node)
                return " ".join(line.strip() for line in s.splitlines())
