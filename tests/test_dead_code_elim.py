"""Unit tests for dead code elimination (CFG-level).

These tests ensure that assignments/var-decls whose values are unused and
have no side-effects are removed, but that side-effectful initializers
(e.g. function calls) are preserved.
"""

"""Unit tests for dead code elimination (CFG-level).

These tests ensure that assignments/var-decls whose values are unused and
have no side-effects are removed, but that side-effectful initializers
(e.g. function calls) are preserved.
"""

from ast_nodes import *
from ast_flatten import flatten_program

"""Unit tests for dead code elimination (CFG-level).

These tests ensure that assignments/var-decls whose values are unused and
have no side-effects are removed, but that side-effectful initializers
(e.g. function calls) are preserved.
"""

from ast_nodes import *
from ast_flatten import flatten_program
from cfg import build_cfg
from cfg_opt import dead_code_elim
from lexer import Lexer
from parser import Parser


def run_pipeline_from_text(text):
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    flat = flatten_program(ast)
    cfg = build_cfg(flat)
    return cfg


def test_dce_removes_unused_assignment():
    # x = 1; return 0;  => assignment to x is dead and should be removed
    prog_text = """
    int main() {
      int x = 1;
      return 0;
    }
    """
    cfg = run_pipeline_from_text(prog_text)
    cfg2 = dead_code_elim(cfg)
    # ensure no AssignmentNode or VarDecl for x remains in any block
    for b in cfg2.get("blocks", []):
        for s in b.get("statements", []):
            assert not (
                isinstance(s, AssignmentNode)
                and isinstance(s.left, IdentifierNode)
                and s.left.name == "x"
            )
            if isinstance(s, VariableDeclarationNode):
                assert s.var_name != "x"


def test_dce_keeps_call_with_side_effects():
    # x = foo(); return 0; => should keep the var-decl/assignment because call may have side-effects
    prog_text = """
    int foo(int a) { return a; }
    int main() {
      int x = foo(5);
      return 0;
    }
    """
    cfg = run_pipeline_from_text(prog_text)
    cfg2 = dead_code_elim(cfg)
    found = False
    for b in cfg2.get("blocks", []):
        for s in b.get("statements", []):
            if isinstance(s, VariableDeclarationNode) and s.var_name == "x":
                found = True
    assert found
