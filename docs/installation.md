# Installation

## From PyPI

```bash
pip install polar-high-opt
```

## From source

```bash
git clone https://github.com/nodal-tools/polar-high-opt.git
cd polar-high-opt
pip install -e .
```

## Optional extras

| Extra | Installs | Use |
|---|---|---|
| `test` | `pytest` | Running the test suite |
| `docs` | `mkdocs-material`, `mkdocstrings[python]`, `mike`, `mkdocs-include-markdown-plugin` | Building this site locally |

```bash
pip install -e ".[test]"          # for tests
pip install -e ".[docs]"          # for docs
pip install -e ".[test,docs]"     # both
```

## Requirements

- Python 3.11+
- [polars](https://pola.rs/) (≥ 1.0)
- [highspy](https://pypi.org/project/highspy/) — HiGHS bundled, no
  separate install
- [numpy](https://numpy.org/)

No system-level solver install is required: HiGHS comes from
`highspy`. If you want to swap in a different HiGHS build, drop in
a different `highspy` wheel.
