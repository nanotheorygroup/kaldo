import sys
# import sphinx_rtd_theme


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
