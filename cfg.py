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


def build_cfg(ast: Union[BlockNode, FunctionDeclarationNode]) -> Dict[str, Any]:
    """Builds a CFG from a flattened BlockNode or FunctionDeclarationNode."""
    label_counter = [0]
    blocks = []
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
                    else_block = {"label": else_label, "statements": [], "out_edges": []}
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
                cond_block = {"label": cond_label, "statements": [stmt], "out_edges": []}
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
