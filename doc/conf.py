import sys
import sphinx_rtd_theme

from ase import __version__

sys.path.append('.')
assert sys.version_info >= (2, 7)

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx']
source_suffix = '.rst'
master_doc = 'index'
project = 'Ballistico'
copyright = '2018, Giuseppe Barbalinardo'
templates_path = ['templates']
exclude_patterns = ['build']
default_role = 'math'
pygments_style = 'sphinx'
autoclass_content = 'both'
modindex_common_prefix = ['bal.']

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['static']
html_last_updated_fmt = '%a, %d %b %Y %H:%M:%S'



# Avoid GUI windows during doctest:
doctest_global_setup = """
import ase.visualize as visualize
from ase import Atoms
visualize.view = lambda atoms: None
Atoms.edit = lambda self: None
"""
