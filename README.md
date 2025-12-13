hop-py — small C-like language toolchain
=====================================

A tiny compiler pipeline.

Requirements
- Python 3.8+ (use a virtual environment)
- `graphviz` system `dot` executable for rendering images (optional)

Running tests
- From the repository root:

```bash
python -m pytest -q
```

Command-line usage
- Run the pipeline on a source file and optionally render a CFG image.

Basic:

```bash
python main.py --file examples/complex.hop
# or as a module
python -m main --file examples/complex.hop
```

Printing / verbosity options
- `--print-tokens` : Print the token stream.
- `--no-ast`       : Do not print the parsed AST.
- `--no-flat`      : Do not print the flattened AST.
- `--no-cfg`       : Do not print the textual CFG.

Liveness options
- `--no-liveness`       : Do not compute or print liveness information.
- `--collapse-liveness` : Collapse identical live sets across contiguous instructions when printing.

CFG JSON dump
- `--dump-cfg PATH` : Write a JSON dump of the CFG + liveness to `PATH`.

Graphviz visualization
- `--viz-cfg PATH`    : Write a Graphviz visualization (PATH is without extension).
- `--viz-format FMT`  : Output format for visualization (e.g. `svg`, `png`). Default: `svg`.
- `--viz-surface`     : Use a compact, surface-style expression printer in the visualization (e.g. `a = b + c`).
- `--viz-no-liveness` : Omit liveness annotations from the visualization image.

Examples
- Generate a PNG with surface-style printing and liveness included (default):

```bash
python -m main --file examples/complex.hop --viz-cfg out/complex --viz-format png --viz-surface
```

- Generate a visualization without any liveness shown:

```bash
python -m main --file examples/complex.hop --viz-cfg out/complex_no_liv --viz-format png --viz-surface --viz-no-liveness
```

Notes
- The `graphviz` Python package is used to construct DOT graphs, but rendering images requires the Graphviz `dot` binary to be installed and available on `PATH`.
- For development and testing, run inside a virtual environment (example):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```
