# Configuration file for the Sphinx documentation builder.

# -- sys path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

# -- Project information

project = 'Lucent'
copyright = '2021, Lim Swee Kiat'
author = 'Lim Swee Kiat'

release = '0.1.8'
version = '0.1.8'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
