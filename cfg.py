"""CFG construction from a flattened AST.

Defines a simple, imperative-style control flow graph (CFG) representation
suitable for code generation and analysis. The CFG consists of basic blocks,
each with a list of statements, and explicit edges for control flow.

This module provides a function to convert a flattened AST (BlockNode or
FunctionDeclarationNode) into a CFG. The CFG is represented as a list of
basic blocks, each with a unique label, and a mapping of edges between blocks.

Example usage:
    cfg = build_cfg(flat_block)
    for block in cfg['blocks']:
        print(block['label'], block['statements'], '->', block['out_edges'])
"""

from ast_nodes import *
from typing import List, Dict, Optional, Any, Union

# A basic block is a dict with:
#   'label': str
#   'statements': List[ASTNode]
#   'out_edges': List[str] (labels of successor blocks)
#
# The CFG is a dict with:
#   'blocks': List[block]
#   'entry': str (label of entry block)
#   'exit': Optional[str] (label of exit block, if any)


def fresh_label(counter: List[int]) -> str:
    counter[0] += 1
    return f"block_{counter[0]}"


def build_cfg(
    ast: Union[BlockNode, FunctionDeclarationNode, ProgramNode],
) -> Dict[str, Any]:
    """Builds a CFG from a flattened BlockNode, FunctionDeclarationNode, or ProgramNode.

    When given a `ProgramNode`, this function will create a top-level CFG
    containing the global code blocks and separate blocks for each function
    declaration found in the program. Function bodies are converted into
    their own basic blocks (so visualizations and analyses can treat them
    independently).
    """
    label_counter = [0]
    blocks: List[Dict[str, Any]] = []

    # Helper to create a fresh empty block and append
    def new_block() -> Dict[str, Any]:
        lbl = fresh_label(label_counter)
        b = {"label": lbl, "statements": [], "out_edges": []}
        blocks.append(b)
        return b

    # If program: create global block(s) and extract functions into separate blocks
    if isinstance(ast, ProgramNode):
        entry_block = new_block()
        cur_block = entry_block
        # accumulate global statements until we hit a function declaration
        global_stmts: List[ASTNode] = []

        for stmt in ast.statements:
            if isinstance(stmt, FunctionDeclarationNode):
                # First flush any accumulated global statements into CFG blocks
                if global_stmts:
                    exit_lbl = _build_cfg_block(
                        BlockNode(statements=global_stmts),
                        cur_block,
                        blocks,
                        label_counter,
                    )
                    # Find the block with this label to continue
                    cur_block = next(
                        (b for b in blocks if b["label"] == exit_lbl), new_block()
                    )
                    global_stmts = []

                # Start a fresh block for the function body
                # Create a block for the function and mark subsequent blocks
                # created for this function. We capture the start index so we can
                # tag all blocks added by _build_cfg_block with the function name.
                start_idx = len(blocks)
                func_entry = new_block()
                # Add the original FunctionDeclarationNode as header in the function's entry
                func_entry["statements"].append(stmt)
                # Build the CFG for the function body into func_entry and following blocks
                _build_cfg_block(
                    stmt.body if stmt.body is not None else BlockNode(statements=[]),
                    func_entry,
                    blocks,
                    label_counter,
                )
                end_idx = len(blocks)
                for b in blocks[start_idx:end_idx]:
                    b["function"] = stmt.func_name
                # After adding function blocks, create a new global block to continue
                cur_block = new_block()
            else:
                global_stmts.append(stmt)

        # flush remaining global statements
        if global_stmts:
            _build_cfg_block(
                BlockNode(statements=global_stmts), cur_block, blocks, label_counter
            )

        cfg = {"blocks": blocks, "entry": entry_block["label"], "exit": None}
        return cfg

    # For single function or block, build normally
    entry_label = fresh_label(label_counter)
    block = {"label": entry_label, "statements": [], "out_edges": []}
    blocks.append(block)
    exit_label = _build_cfg_block(
        ast.body if isinstance(ast, FunctionDeclarationNode) else ast,
        block,
        blocks,
        label_counter,
    )
    cfg = {"blocks": blocks, "entry": entry_label, "exit": exit_label}
    return cfg


def _build_cfg_block(
    block_node: BlockNode, cur_block: Dict, blocks: List[Dict], label_counter: List[int]
) -> Optional[str]:
    stmts = block_node.statements
    i = 0
    while i < len(stmts):
        stmt = stmts[i]
        t = stmt.type
        match t:
            case NodeType.IF_STMT:
                # End current block at the if
                then_label = fresh_label(label_counter)
                else_label = fresh_label(label_counter) if stmt.else_block else None
                after_label = fresh_label(label_counter)
                # Add the if statement as a terminator
                cur_block["statements"].append(stmt)
                cur_block["out_edges"].append(then_label)
                if else_label:
                    cur_block["out_edges"].append(else_label)
                else:
                    cur_block["out_edges"].append(after_label)
                # Then block
                then_block = {"label": then_label, "statements": [], "out_edges": []}
                blocks.append(then_block)
                _build_cfg_block(stmt.then_block, then_block, blocks, label_counter)
                then_block["out_edges"].append(after_label)
                # Else block
                if stmt.else_block:
                    else_block = {
                        "label": else_label,
                        "statements": [],
                        "out_edges": [],
                    }
                    blocks.append(else_block)
                    _build_cfg_block(stmt.else_block, else_block, blocks, label_counter)
                    else_block["out_edges"].append(after_label)
                # Continue after if
                next_block = {"label": after_label, "statements": [], "out_edges": []}
                blocks.append(next_block)
                cur_block = next_block
                i += 1

            case NodeType.WHILE_STMT:
                # End current block at the while
                cond_label = fresh_label(label_counter)
                body_label = fresh_label(label_counter)
                after_label = fresh_label(label_counter)
                # Jump to condition
                cur_block["out_edges"].append(cond_label)
                # Condition block
                cond_block = {
                    "label": cond_label,
                    "statements": [stmt],
                    "out_edges": [],
                }
                blocks.append(cond_block)
                cond_block["out_edges"].append(body_label)
                cond_block["out_edges"].append(after_label)
                # Body block
                body_block = {"label": body_label, "statements": [], "out_edges": []}
                blocks.append(body_block)
                # Build the body and get the label of the block where the body falls through
                body_exit_label = _build_cfg_block(
                    stmt.body, body_block, blocks, label_counter
                )
                # Attach loop-back edge from the body's exit block back to the condition
                if body_exit_label is not None:
                    for blk in blocks:
                        if blk["label"] == body_exit_label:
                            blk["out_edges"].append(cond_label)
                            break
                # Continue after while
                next_block = {"label": after_label, "statements": [], "out_edges": []}
                blocks.append(next_block)
                cur_block = next_block
                i += 1

            case NodeType.RETURN_STMT:
                cur_block["statements"].append(stmt)
                # Return is a terminator; no out_edges
                return cur_block["label"]

            case _:
                cur_block["statements"].append(stmt)
                i += 1
    # If we reach here, this block falls through
    return cur_block["label"]
