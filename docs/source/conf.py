# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add source directory so Sphinx can find your package
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
project = "pyFTLE"
copyright = "2025, Renato Fuzaro Miotto, Lucas Feitosa de Souza, William Roberto Wolf"
author = "Renato Fuzaro Miotto, Lucas Feitosa de Souza, William Roberto Wolf"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # for Google/NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",  # allows Markdown (for README.md)
]

autosummary_generate = True  # Auto-generate .rst for modules

templates_path = ["_templates"]
exclude_patterns = []

# Enable MyST anchor generation for headings
myst_enable_extensions = [
    "colon_fence",
    "attrs_block",
    "attrs_inline",
]
# Automatically add HTML anchors to all headings up to depth 3
myst_heading_anchors = 3

# -- Mock C++ extension modules that can't be imported ------------------------
# This prevents Sphinx from trying to import compiled pybind11 extensions,
# which would otherwise raise "PyInit_*" import errors during autodoc.
autodoc_mock_imports = ["pyftle.ginterp"]


# -- Autodoc / Napoleon configuration ----------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "__init__, __weakref__, __dict__",
    "no-index": True,  # prevents duplicate indexing
}

# Dataclass / type hint handling to avoid duplicate object descriptions
autodoc_typehints = "description"  # move hints into description
autodoc_typehints_description_target = "documented"  # only for documented attributes
autodoc_class_signature = "separated"
autodoc_inherit_docstrings = False
autodoc_preserve_defaults = True

# Napoleon config: support for Google and NumPy docstring styles
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

# -- Intersphinx configuration -----------------------------------------------
# Helps link to Python/NumPy docs automatically
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]

# -- Markdown / README inclusion ---------------------------------------------
# Allow including README.md as part of the docs
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


# -- Custom hook to avoid duplicate dataclass field documentation ------------
def skip_dataclass_field_duplicates(app, what, name, obj, skip, options):
    """
    Skip auto-documenting dataclass fields that are already documented
    in the class docstring.

    This prevents warnings like:
        duplicate object description of module.Class.field
    """
    # Skip if this is a dataclass field (detected via __dataclass_fields__)
    if hasattr(obj, "__dataclass_fields__"):
        return True
    return skip


def setup(app):
    # Connect the autodoc-skip-member event to our custom hook
    app.connect("autodoc-skip-member", skip_dataclass_field_duplicates)
