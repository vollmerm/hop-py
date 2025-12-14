"""Unit tests for dead code elimination (CFG-level)."""

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


def test_dce_respects_occurs_in_for_while_condition_and_body():
        # A temporary vardecl followed by a while that uses the temp in its
        # condition/body should not be removed by conservative DCE. 
        prog_text = '''
        int main() {
            int _tmp1 = 0;
            while (_tmp1 < 3) {
                _tmp1 = _tmp1 + 1;
            }
            return 0;
        }
        '''
        cfg = run_pipeline_from_text(prog_text)
        cfg2 = dead_code_elim(cfg, level="conservative")
        # Ensure the temporary vardecl `_tmp1` still exists in the CFG
        found_tmp = False
        for b in cfg2.get("blocks", []):
                for s in b.get("statements", []):
                        if isinstance(s, VariableDeclarationNode) and s.var_name == "_tmp1":
                                found_tmp = True
        assert found_tmp, "conservative DCE removed a temporary used in a while loop"
