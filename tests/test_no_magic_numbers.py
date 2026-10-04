"""No hardcoded 'magic' modelling numbers in physics packages (spec 20.2).

Numbers must come from the property DB (``hydra.thermo.propdb``) or ``hydra.constants``. Structural
numbers (small integers, 0.5, exponents...) are allowed; anything else needs a ``# magic:``
comment with a reason, or lives in a function default / class-level config default.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "hydra"
PHYSICS = ["core", "electrochem", "thermo", "transport", "fuelcell", "system", "scaleup"]  # strict
ALGORITHMIC = ["inference", "design", "optimize", "control", "surrogate", "hardware", "eln"]  # only physical constants are forbidden
PHYSICAL_VALUES = {273.15, 298.15, 293.15, 323.15, 373.15, 101325.0, 1.0e5, 22.414, 22.4, 96485.0, 96485.33212, 8.314, 8.314462618,
                   26.9815385, 26.98, 18.015, 39.997, 2.016, 2700.0, 9.80665, 9.81, 1.380649e-23, 6.02214076e23, 5.670374419e-08}
EXEMPT_FILES = {"constants.py", "propdb.py"}
ALLOWED_INT = set(range(0, 11)) | {20, 30, 50, 60, 100, 1000, 3600}
ALLOWED_FLOAT = {0.0, 0.5, 1.5, 2.5, 0.25, 0.1, 0.01, 0.05, 0.75, 1e-3, 1e3, 1e6, 1e-6, 1e-9, 1e9, 1e-12,
                 1e-10, 1e-30, 1e-8, 1e-15, 1e-4, 1e-5, 1e-7}


def violations() -> list[str]:
    out = []
    for pkg in PHYSICS + ALGORITHMIC:
        for f in (SRC / pkg).glob("*.py"):
            if f.name in EXEMPT_FILES or f.name == "__init__.py":
                continue
            text = f.read_text()
            lines = text.splitlines()
            tree = ast.parse(text)
            skip: set[int] = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for d in node.args.defaults + [d for d in node.args.kw_defaults if d]:
                        skip |= {n.lineno for n in ast.walk(d) if hasattr(n, "lineno")}
                if isinstance(node, ast.ClassDef):
                    for b in node.body:
                        if isinstance(b, ast.AnnAssign) and b.value is not None:
                            skip |= {n.lineno for n in ast.walk(b.value) if hasattr(n, "lineno")}
                if (isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(getattr(node.body[0], "value", None), ast.Constant)):
                    d = node.body[0]
                    skip |= set(range(d.lineno, d.end_lineno + 1))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                        and not isinstance(node.value, bool):
                    v = node.value
                    if pkg in ALGORITHMIC:
                        if float(v) in PHYSICAL_VALUES and "# magic:" not in lines[node.lineno - 1] and node.lineno not in skip:
                            out.append(f"{f.relative_to(SRC)}:{node.lineno}: physical constant {v}")
                        continue
                    if node.lineno in skip or "# magic:" in lines[node.lineno - 1]:
                        continue
                    if isinstance(v, int) and v in ALLOWED_INT:
                        continue
                    if isinstance(v, float) and (v in ALLOWED_FLOAT or float(v).is_integer() and int(v) in ALLOWED_INT):
                        continue
                    out.append(f"{f.relative_to(SRC)}:{node.lineno}: {v}")
    return out


def test_no_magic_numbers():
    v = violations()
    assert not v, "magic numbers (move to property DB or add '# magic:'):\n" + "\n".join(v[:60])
