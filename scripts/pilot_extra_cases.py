"""Preselected original-baseline failures and one success; public-issue DoD only."""

FAILURE_IDS = ("sympy__sympy-21171", "sympy__sympy-13177", "sphinx-doc__sphinx-8595")
SUCCESS_IDS = ("pytest-dev__pytest-11143",)
IMPORTS = {
    FAILURE_IDS[0]: ("sympy",),
    FAILURE_IDS[1]: ("sympy",),
    FAILURE_IDS[2]: ("sphinx",),
    SUCCESS_IDS[0]: ("pytest", "_pytest"),
}
PUBLIC_CHECKS = {
    FAILURE_IDS[0]: r"""from sympy import symbols, latex, SingularityFunction
from sympy.physics.continuum_mechanics import Beam
E, L, F, B, H = symbols("E L F B H")
I = B*H**3/12
beam = Beam(L, E, I)
beam.apply_load(-F, L/2, -1)
beam.apply_support(0, "fixed")
R0, M0 = symbols("R_0 M_0")
beam.solve_for_reaction_loads(R0, M0)
x = symbols("x")
for expression in (beam.shear_force(), SingularityFunction(x, 0, -1)**2):
    assert isinstance(latex(expression), str)
print("public regression passed")
""",
    FAILURE_IDS[1]: r"""from sympy import Symbol, Mod, Rational
x = Symbol("x")
expression = Mod(x**2, x)
assert expression != 0
assert expression.subs(x, Rational(3,2)) == Rational(3,4)
i = Symbol("i", integer=True, nonzero=True)
assert Mod(i**2, i) == 0
print("public regression passed")
""",
    FAILURE_IDS[2]: r"""import io, tempfile
from pathlib import Path
from sphinx.application import Sphinx
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    source = root / "source"
    source.mkdir()
    (source / "example.py").write_text('__all__ = []\n\ndef foo():\n    "foo docstring"\n\ndef bar():\n    "bar docstring"\n\ndef baz():\n    "baz docstring"\n')
    (source / "conf.py").write_text("import sys\nsys.path.insert(0, " + repr(str(source)) + ")\nextensions = ['sphinx.ext.autodoc']\nmaster_doc = 'index'\nproject = 'public-probe'\n")
    (source / "index.rst").write_text("Public issue\n============\n\n.. automodule:: example\n   :members:\n")
    output = root / "output"
    app = Sphinx(str(source), str(source), str(output), str(root / "doctrees"), "text", status=io.StringIO(), warning=io.StringIO(), freshenv=True)
    app.build(force_all=True)
    assert app.statuscode == 0
    rendered = (output / "index.txt").read_text()
    assert all(name not in rendered for name in ("foo", "bar", "baz")), rendered
print("public regression passed")
""",
    SUCCESS_IDS[0]: r"""import ast, os
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
from _pytest.config import Config
from _pytest.assertion.rewrite import rewrite_asserts
config = Config.fromdictargs({}, [])
try:
    for source in (b"123\nassert 1 == 1\n", b"'a docstring'\nassert 1 == 1\n"):
        tree = ast.parse(source)
        rewrite_asserts(tree, source, "public_reproduction.py", config)
        exec(compile(tree, "public_reproduction.py", "exec"), {})
finally:
    config._ensure_unconfigure()
print("public regression passed")
""",
}
