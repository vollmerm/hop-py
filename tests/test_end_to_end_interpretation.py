from lexer import Lexer
from parser import Parser
from ast_flatten import flatten_program
from cfg import build_cfg
from ast_interpreter import interpret_program
from cfg_interpreter import interpret_cfg
from cfg_opt import copy_propagation_cfg, dead_code_elim


def run_pipeline_and_get_return(text, *, run_opt=False, dce_level="conservative"):
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    # AST interpreter runs `main` automatically
    ast_env = interpret_program(ast)
    ast_ret = ast_env.get("__return__")

    # Flatten + CFG, then interpret via CFG interpreter
    flat = flatten_program(ast)
    cfg = build_cfg(flat)
    cfg_env = interpret_cfg(cfg)
    cfg_ret = cfg_env.get("__return__")

    if not run_opt:
        return ast_ret, cfg_ret

    # Apply copy-propagation and conservative DCE (or chosen level)
    cfg2 = copy_propagation_cfg(cfg)
    cfg2 = dead_code_elim(cfg2, level=dce_level)
    cfg2_env = interpret_cfg(cfg2)
    cfg2_ret = cfg2_env.get("__return__")

    return ast_ret, cfg_ret, cfg2_ret


PROGRAM = """
int fact(int n) {
  if (n <= 1) return 1;
  int t = n - 1;
  int r = fact(t);
  return n * r;
}

int main() {
  int x = fact(5);
  return x;
}
"""


def test_end_to_end_no_opt():
    ast_ret, cfg_ret = run_pipeline_and_get_return(PROGRAM, run_opt=False)
    assert ast_ret == cfg_ret


def test_end_to_end_with_opt_conservative():
    ast_ret, cfg_ret, cfg2_ret = run_pipeline_and_get_return(PROGRAM, run_opt=True, dce_level="conservative")
    assert ast_ret == cfg_ret == cfg2_ret


def test_end_to_end_with_opt_aggressive():
    ast_ret, cfg_ret, cfg2_ret = run_pipeline_and_get_return(PROGRAM, run_opt=True, dce_level="aggressive")
    assert ast_ret == cfg_ret == cfg2_ret
