# Copyright 2020 DeepMind Technologies Limited.
# Licensed under the Apache License, Version 2.0

"""Sphinx configuration for Jraph documentation."""

import os
import sys
import inspect

extensions = [
    "myst_nb",            # supports MyST markdown + notebook cells
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

# Allow .md files as source
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST / myst-nb settings
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_admonition",
    "html_image",
]

# Don't execute cells during the docs build (safe default for heavy ML libs).
nb_execution_mode = "off"   # avoid executing heavy ML code during the build

# -- BibTeX config ----------------------------------------------------------
bibtex_bibfiles = []  # empty list if no .bib files yet

# -- HTML static path -------------------------------------------------------
os.makedirs(os.path.join(os.path.dirname(__file__), "_static"), exist_ok=True)
html_static_path = ['_static']


# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath('../'))
sys.path.append(os.path.abspath('ext'))

# -- Project information -----------------------------------------------------

project = 'Jraph'
copyright = '2021, Jraph Authors'
author = 'Jraph Authors'

# Safe version info (do not import heavy packages)
version = "0.0.0"
release = "0.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'sphinx_autodoc_typehints',
    'myst_parser'
]

pygments_style = 'sphinx'
templates_path = ['_templates']
exclude_patterns = ['_build']

# -- Autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': True,
}

# Mock heavy dependencies to prevent import errors
autodoc_mock_imports = [
    "jax", "jaxlib", "dm_haiku", "flax", "optax", "jraph"
]

# -- HTML output -------------------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']

# -- Linkcode support --------------------------------------------------------

def linkcode_resolve(domain, info):
    """Resolve GitHub URL corresponding to a Python object."""
    if domain != 'py':
        return None

    try:
        mod = sys.modules.get(info['module'])
        if mod is None:
            return None

        obj = mod
        for attr in info['fullname'].split('.'):
            obj = getattr(obj, attr)
        obj = inspect.unwrap(obj)

        filename = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    try:
        import jraphzzz
        base_path = os.path.dirname(jraphzzz.__file__)
    except ImportError:
        base_path = ""

    return 'https://github.com/syedzayyan/jraphzzz/blob/master/jraph/%s#L%d#L%d' % (
        os.path.relpath(filename, start=base_path),
        lineno,
        lineno + len(source) - 1
    )
