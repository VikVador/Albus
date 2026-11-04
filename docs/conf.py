# Configuration file for the Sphinx documentation builder

import glob
import importlib
import importlib.metadata
import inspect
import pathlib
import re
import subprocess

from sphinx.application import Sphinx

import albus

## Project

package = "albus"
project = "Albus"
version = importlib.metadata.version(package)
copyright = "2026"
repository = "https://github.com/VikVador/albus"
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
root = pathlib.Path(albus.__file__).parents[1]

## Extensions

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__call__",
}
autodoc_inherit_docstrings = False
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"

autosummary_ignore_module_all = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}


def linkcode_resolve(domain: str, info: dict) -> str:
    module = info.get("module", "")
    fullname = info.get("fullname", "")

    if not module or not fullname:
        return None

    objct = importlib.import_module(module)
    for name in fullname.split("."):
        objct = getattr(objct, name)

    while hasattr(objct, "__wrapped__"):
        objct = objct.__wrapped__

    try:
        file = inspect.getsourcefile(objct)
        file = pathlib.Path(file).relative_to(root)

        lines, start = inspect.getsourcelines(objct)
        end = start + len(lines) - 1
    except Exception:
        return None
    else:
        return f"{repository}/blob/{commit}/{file}#L{start}-L{end}"


# Albus names its returned values only when there are several (e.g. "skill: Skill, ..."),
# render those like arguments and keep single prose descriptions as they are
def render_returns(
    app: Sphinx, what: str, name: str, obj: object, options: dict, lines: list
) -> None:
    r"""Turns a "Returns:" section of several named values into an argument-like field list."""

    try:
        start = lines.index("Returns:")
    except ValueError:
        return

    end = start + 1
    while end < len(lines) and (lines[end].startswith(" ") or not lines[end].strip()):
        end += 1

    items = [line for line in lines[start + 1 : end] if re.match(r"^    \w+\s+:", line)]
    if len(items) > 1:
        lines[start] = "Returned values:"


napoleon_custom_sections = [("Returned values", "params_style")]

## Settings

add_function_parentheses = False
default_role = "literal"
exclude_patterns = ["templates"]
html_copy_source = False
html_css_files = [
    "custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
]
html_show_sourcelink = False
html_static_path = ["static"]
html_theme = "furo"
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": repository,
            "html": '<i class="fa-brands fa-github fa-lg"></i>',
            "class": "",
        },
    ],
    "light_css_variables": {
        "color-api-keyword": "#007020",
        "color-api-name": "#0e84b5",
        "color-api-pre-name": "#0e84b5",
    },
    "dark_css_variables": {
        "color-api-keyword": "#66d9ef",
        "color-api-name": "#a6e22e",
        "color-api-pre-name": "#a6e22e",
    },
    "sidebar_hide_name": True,
}
html_title = project
pygments_style = "sphinx"
pygments_dark_style = "monokai"
rst_prolog = """
.. role:: py(code)
    :class: highlight
    :language: python
"""
templates_path = ["templates"]


## Edit HTML
def edit_html(app: Sphinx, exception: Exception) -> None:
    if exception:
        raise exception

    for file in glob.glob(f"{app.outdir}/**/*.html", recursive=True):
        with open(file) as f:
            text = f.read()

        # fmt: off
        text = text.replace('<a class="muted-link" href="https://pradyunsg.me">@pradyunsg</a>\'s', "")
        text = re.sub(r'(<a class="reference external".*</a>)(<a class="headerlink".*</a>)', r"\2\1", text)
        text = text.replace('>Returned values<span class="colon">', '>Returns<span class="colon">')
        # fmt: on

        with open(file, "w") as f:
            f.write(text)


## Formulas
def render_formulas(
    app: Sphinx, what: str, name: str, obj: object, options: dict, lines: list
) -> None:
    r"""Turns the "Formula:" section of a docstring into a literal block, before napoleon parses it."""

    for i, line in enumerate(lines):
        if line.strip() == "Formula:":
            lines[i : i + 1] = [".. rubric:: Formula", "", ".. code-block:: text", ""]
            break


def setup(app: Sphinx) -> None:
    app.connect("autodoc-process-docstring", render_formulas, priority=400)
    app.connect("autodoc-process-docstring", render_returns, priority=400)
    app.connect("build-finished", edit_html)
