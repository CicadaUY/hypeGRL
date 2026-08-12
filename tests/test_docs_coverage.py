# -*- coding: utf-8 -*-
"""
Guard that every public module reaches the documentation.

Autodoc is opt-in per module: a new module is invisible on the rendered site
until some ``.rst`` names it. Nothing fails when that step is skipped — the
build succeeds and the page simply lacks the class, which is how the whole
``representations`` layer stayed unpublished while being the library's central
abstraction. This test is the missing failure.

A module is exempt when it is private (``_name``), or a stub with nothing
public to document.
"""

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "hypegrl"
DOCS = REPO / "docs"

# Directives that hand a module (or something inside it) to autodoc.
AUTODOC = re.compile(r"^\s*\.\.\s+auto(module|class|function|data)::\s+(\S+)", re.M)


def _documented_modules() -> set[str]:
    """Every module named by an autodoc directive anywhere under ``docs/``."""
    documented = set()
    for rst in DOCS.rglob("*.rst"):
        if "_build" in rst.parts:
            continue
        for kind, target in AUTODOC.findall(rst.read_text()):
            # autoclass/autofunction/autodata name a member: drop the last part.
            documented.add(target if kind == "module" else target.rsplit(".", 1)[0])
    return documented


def _public_modules() -> list[str]:
    """
    Dotted names of the modules that owe the docs an entry: importable, public,
    and defining at least one public class or function.
    """
    public = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if any(
            part.startswith((".", "_")) and part != "__init__.py"
            for part in path.parts
        ):
            continue

        tree = ast.parse(path.read_text())
        has_public_member = any(
            isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and not node.name.startswith("_")
            for node in tree.body
        )
        if not has_public_member:
            continue  # stub, or a module of private helpers only

        dotted = ".".join(path.relative_to(REPO).with_suffix("").parts)
        public.append(dotted.removesuffix(".__init__"))
    return public


def _methods_page_classes() -> list[str]:
    """Embedder classes given a dedicated page under ``docs/methods/``."""
    classes = []
    for rst in sorted((DOCS / "methods").glob("*.rst")):
        found = AUTODOC.findall(rst.read_text())
        classes += [target for kind, target in found if kind == "class"]
    return classes


@pytest.mark.parametrize("module", _public_modules())
def test_public_module_is_documented(module):
    assert module in _documented_modules(), (
        f"{module} defines public API but no docs/**.rst names it, so it is "
        f"absent from the rendered documentation. Add an automodule directive "
        f"(or an autoclass for its main class) to the matching page under "
        f"docs/api/."
    )


@pytest.mark.parametrize("cls", _methods_page_classes())
def test_embedder_is_listed_in_the_api_summary(cls):
    """
    The API reference's summary table is the index of the methods: it is written
    by hand, so a new embedder reaches it only if someone adds the row. Without
    this, a method can be fully documented on its own page and still be
    unreachable from the API reference.
    """
    summary = (DOCS / "api" / "embedders.rst").read_text()
    assert cls in summary, (
        f"{cls} has a page under docs/methods/ but no row in the autosummary "
        f"table in docs/api/embedders.rst, so the API reference does not list "
        f"it. Add the dotted class name to that table."
    )
