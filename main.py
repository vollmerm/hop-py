from __future__ import annotations
from typing import List, Optional
from lexer import Lexer
from tokens import Token
from ast_nodes import ASTNode, NodeType
from parser import Parser
from type_checker import TypeChecker

from pretty_printer import PrettyPrinter
from ast_flatten import flatten_program
from cfg import build_cfg
from liveness import analyze_liveness
import json
from typing import Optional
from ast_json import ast_to_json
from cfg_viz import write_and_render
from cfg_instrsel import select_instructions
from regalloc import build_interference, allocate_registers
from cfg_viz import render_cfg_dot
from cfg_opt import copy_propagation_cfg, dead_code_elim


def lex(text: str) -> List[Token]:
    """Tokenize input string."""
    lexer = Lexer(text)
    return lexer.tokenize()


def parse_tokens(tokens: List[Token]) -> ASTNode:
    """Parse tokens into AST."""
    parser = Parser(tokens)
    return parser.parse()


def process_program(
    text: str,
    *,
    print_tokens: bool = False,
    print_ast: bool = True,
    print_flat: bool = True,
    print_cfg_flag: bool = True,
    show_liveness: bool = True,
    collapse_liveness: bool = False,
    dump_cfg_path: Optional[str] = None,
    viz_path: Optional[str] = None,
    viz_format: str = "svg",
    viz_surface: bool = False,
    viz_include_liveness: bool = False,
    viz_alloc: Optional[str] = None,
    run_optimizations: bool = False,
    dce_level: str = "conservative",
) -> None:
    """Process a single program: lex, parse, typecheck, flatten, build CFG and optionally print stages.

    Flags control which parts are printed; printing can be toggled separately.
    """
    try:
        tokens = lex(text)
        if print_tokens:
            print(f"Tokens ({len(tokens)}):")
            for i, token in enumerate(tokens[:50]):
                print(f"  {i:3}: {token}")
            if len(tokens) > 50:
                print(f"  ... and {len(tokens) - 50} more")

        ast = parse_tokens(tokens)
        if print_ast:
            print("\nAST:")
            print(PrettyPrinter.print_ast(ast))

        # Type check
        try:
            if ast.type == NodeType.PROGRAM:
                for stmt in ast.statements:
                    TypeChecker.check_statement(stmt)
            else:
                TypeChecker.check_statement(ast)
            if print_ast:
                print("\n✓ Type check passed")
        except SyntaxError as e:
            print(f"\n✗ Type error: {e}")
            return

        flat_ast = flatten_program(ast)
        if print_flat:
            print("\nFlattened AST:")
            print(PrettyPrinter.print_ast(flat_ast))

        if print_cfg_flag:
            cfg = build_cfg(flat_ast)
            # Optionally run CFG-level optimizations (copy-propagation, DCE)
            if run_optimizations:
                try:
                    cfg = copy_propagation_cfg(cfg)
                except Exception:
                    pass
                try:
                    cfg = dead_code_elim(cfg, level=dce_level)
                except Exception:
                    pass
            liv = None
            if show_liveness:
                liv = analyze_liveness(cfg)
                print("\nCFG (with liveness):")
                print(
                    PrettyPrinter.print_cfg(
                        cfg, liveness=liv, collapse_liveness=collapse_liveness
                    )
                )
            else:
                print("\nCFG:")
                print(PrettyPrinter.print_cfg(cfg))

            # Optionally dump CFG + liveness to JSON. Produce a serializable
            # representation: statements are converted to pretty-printed strings
            # and live sets are lists of variable names.
            if dump_cfg_path:
                export = {
                    "entry": cfg.get("entry"),
                    "exit": cfg.get("exit"),
                    "blocks": [],
                }
                for b in cfg.get("blocks", []):
                    lbl = b["label"]
                    block_entry = {
                        "label": lbl,
                        "out_edges": list(b.get("out_edges", [])),
                        "statements": [],
                        "live_in": [],
                        "live_out": [],
                        "instr_liveness": [],
                    }
                    # statements as pretty-printed strings
                    for stmt in b.get("statements", []):
                        block_entry["statements"].append(ast_to_json(stmt))

                    if liv and lbl in liv:
                        block_entry["live_in"] = sorted(
                            list(liv[lbl].get("live_in", []))
                        )
                        block_entry["live_out"] = sorted(
                            list(liv[lbl].get("live_out", []))
                        )
                        for li, lo in liv[lbl].get("instr_liveness", []):
                            block_entry["instr_liveness"].append(
                                {
                                    "live_in": sorted(list(li)),
                                    "live_out": sorted(list(lo)),
                                }
                            )

                    export["blocks"].append(block_entry)

                try:
                    with open(dump_cfg_path, "w", encoding="utf-8") as fh:
                        json.dump(export, fh, indent=2)
                    print(f"Wrote CFG+liveness JSON to {dump_cfg_path}")
                except Exception as e:
                    print(f"Failed to write CFG JSON to {dump_cfg_path}: {e}")

            # Optionally render visualization via Graphviz
            if viz_path:
                try:
                    write_and_render(
                        cfg,
                        viz_path,
                        liveness=liv,
                        collapse_liveness=collapse_liveness,
                        fmt=viz_format,
                        use_surface=viz_surface,
                        include_liveness=viz_include_liveness,
                    )
                    print(f"Wrote CFG visualization to {viz_path}.{viz_format}")
                except Exception as e:
                    print(f"Failed to render CFG visualization to {viz_path}: {e}")

            # Optionally run instruction selection + register allocation and
            # render before/after allocation visualizations.
            if viz_alloc:
                try:
                    instr_cfg = select_instructions(cfg)

                    def _stringify_liveness(liv):
                        if not liv:
                            return None
                        out = {}
                        for lbl, info in liv.items():
                            li = sorted([str(r) for r in info.get("live_in", set())])
                            lo = sorted([str(r) for r in info.get("live_out", set())])
                            instr_l = []
                            for pair in info.get("instr_liveness", []):
                                a, b = pair
                                instr_l.append((sorted([str(x) for x in a]), sorted([str(x) for x in b])))
                            out[lbl] = {"live_in": li, "live_out": lo, "instr_liveness": instr_l}
                        return out

                    ig_before, liv_before, moves = build_interference(instr_cfg)
                    s_liv_before = _stringify_liveness(liv_before)
                    # write dot and try render
                    dot = render_cfg_dot(instr_cfg, liveness=s_liv_before, include_liveness=True, use_surface=viz_surface)
                    try:
                        write_and_render(instr_cfg, f"{viz_alloc}_before", liveness=s_liv_before, fmt=viz_format, use_surface=viz_surface, include_liveness=True)
                        print(f"Wrote allocation-before visualization to {viz_alloc}_before.{viz_format}")
                    except Exception:
                        # fallback: write dot source
                        with open(f"{viz_alloc}_before.dot", "w", encoding="utf-8") as fh:
                            fh.write(dot.source)
                        print(f"Wrote DOT to {viz_alloc}_before.dot (PNG render failed)")

                    assign, rewritten_cfg, spilled = allocate_registers(instr_cfg)

                    ig_after, liv_after, _ = build_interference(rewritten_cfg)
                    s_liv_after = _stringify_liveness(liv_after)
                    dot2 = render_cfg_dot(rewritten_cfg, liveness=s_liv_after, include_liveness=True, use_surface=viz_surface)
                    try:
                        write_and_render(rewritten_cfg, f"{viz_alloc}_after", liveness=s_liv_after, fmt=viz_format, use_surface=viz_surface, include_liveness=True)
                        print(f"Wrote allocation-after visualization to {viz_alloc}_after.{viz_format}")
                    except Exception:
                        with open(f"{viz_alloc}_after.dot", "w", encoding="utf-8") as fh:
                            fh.write(dot2.source)
                        print(f"Wrote DOT to {viz_alloc}_after.dot (PNG render failed)")
                except Exception as e:
                    print(f"Failed to produce allocation visualization: {e}")

    except SyntaxError as e:
        print(f"Syntax Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()


def interactive_mode(
    print_tokens: bool = False,
    print_ast: bool = True,
    print_flat: bool = True,
    print_cfg_flag: bool = True,
    show_liveness: bool = True,
    collapse_liveness: bool = False,
) -> None:
    """Run interactive compiler REPL reading programs from stdin."""
    print("\nInteractive Compiler Mode (type 'quit' to exit)")
    print("=" * 80)

    while True:
        try:
            text = input("\nEnter program or expression: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not text:
                continue

            process_program(
                text,
                print_tokens=print_tokens,
                print_ast=print_ast,
                print_flat=print_flat,
                print_cfg_flag=print_cfg_flag,
                show_liveness=show_liveness,
                collapse_liveness=collapse_liveness,
            )

        except SyntaxError as e:
            print(f"Syntax error: {e}")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run compiler pipeline on a file or interactively from stdin"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--file", "-f", dest="file", help="Path to source file to process"
    )
    group.add_argument(
        "--interactive",
        "-i",
        dest="interactive",
        action="store_true",
        help="Start interactive REPL mode",
    )
    # printing/verbosity options
    parser.add_argument(
        "--print-tokens", dest="print_tokens", action="store_true", help="Print tokens"
    )
    parser.add_argument(
        "--no-ast", dest="print_ast", action="store_false", help="Do not print AST"
    )
    parser.add_argument(
        "--no-flat",
        dest="print_flat",
        action="store_false",
        help="Do not print flattened AST",
    )
    parser.add_argument(
        "--no-cfg", dest="print_cfg_flag", action="store_false", help="Do not print CFG"
    )
    # liveness options
    parser.add_argument(
        "--no-liveness",
        dest="show_liveness",
        action="store_false",
        help="Do not compute or print liveness information",
    )
    parser.add_argument(
        "--collapse-liveness",
        dest="collapse_liveness",
        action="store_true",
        help="Collapse identical live sets across contiguous instructions",
    )

    # default behavior: print everything
    parser.set_defaults(
        print_tokens=False,
        print_ast=True,
        print_flat=True,
        print_cfg_flag=True,
        show_liveness=True,
        collapse_liveness=False,
    )
    parser.add_argument(
        "--dump-cfg", dest="dump_cfg", help="Path to write CFG+liveness JSON"
    )
    parser.add_argument(
        "--viz-cfg",
        dest="viz_cfg",
        help="Path (without extension) to write Graphviz visualization of CFG",
    )
    parser.add_argument(
        "--viz-format",
        dest="viz_format",
        default="svg",
        help="Format for Graphviz output (svg, png, pdf, etc)",
    )
    parser.add_argument(
        "--viz-surface",
        dest="viz_surface",
        action="store_true",
        help="Use surface-style expression printing in CFG visualization",
    )
    parser.add_argument(
        "--viz-liveness",
        dest="viz_liveness",
        action="store_true",
        help="Include liveness information in the CFG visualization image",
    )
    parser.add_argument(
        "--viz-alloc",
        dest="viz_alloc",
        help="Path (without extension) to write allocation visualization (before/after)",
    )
    parser.add_argument(
        "--optimize",
        dest="optimize",
        action="store_true",
        help="Run CFG-level optimizations (copy-propagation + DCE) before lowering",
    )
    parser.add_argument(
        "--dce-level",
        dest="dce_level",
        choices=["conservative", "aggressive"],
        default="conservative",
        help="Aggressiveness of dead-code-elim (conservative or aggressive)",
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(
            print_tokens=args.print_tokens,
            print_ast=args.print_ast,
            print_flat=args.print_flat,
            print_cfg_flag=args.print_cfg_flag,
            show_liveness=args.show_liveness,
            collapse_liveness=args.collapse_liveness,
        )
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as fh:
                text = fh.read()
        except Exception as e:
            print(f"Failed to read file {args.file}: {e}")
            sys.exit(1)

        process_program(
            text,
            print_tokens=args.print_tokens,
            print_ast=args.print_ast,
            print_flat=args.print_flat,
            print_cfg_flag=args.print_cfg_flag,
            show_liveness=args.show_liveness,
            collapse_liveness=args.collapse_liveness,
            dump_cfg_path=args.dump_cfg,
            viz_path=args.viz_cfg,
            viz_format=args.viz_format,
            viz_surface=args.viz_surface,
            viz_include_liveness=(args.viz_liveness),
            viz_alloc=args.viz_alloc,
            run_optimizations=args.optimize,
            dce_level=args.dce_level,
        )
    else:
        parser.print_help()
