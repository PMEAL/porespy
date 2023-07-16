#------------------------------------------------------------------------#
# Path setup                                                             #
#------------------------------------------------------------------------#
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from datetime import datetime
import mock

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))

MOCK_MODULES = ['imagej']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

#------------------------------------------------------------------------#
# Project info                                                           #
#------------------------------------------------------------------------#

project = 'PoreSpy'
copyright = f'{datetime.now().year}, PMEAL'
author = 'PoreSpy Dev Team'

# Copy examples folder from PoreSpy root to docs folder
import shutil
shutil.copytree('../examples', 'examples', dirs_exist_ok=True)

#------------------------------------------------------------------------#
# General config                                                         #
#------------------------------------------------------------------------#

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_nb',
    'ablog',
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# So that 'sphinx-copybutton' only copies the actual code, not the prompt
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

add_module_names = False  # porespy.generators --> generators
autosummary_generate = True
globaltoc_maxdepth = 2

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# The master toctree document.
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'posts/*/.ipynb_checkpoints/*']
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False
# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['porespy']

#------------------------------------------------------------------------#
# Options for HTML output                                                #
#------------------------------------------------------------------------#

html_theme = 'pydata_sphinx_theme'
html_logo = '_static/images/porespy_logo.png'
html_js_files = ['js/custom.js']
html_css_files = ['css/custom.css']
html_static_path = ['_static']
# If false, no module index is generated.
html_domain_indices = True
# If false, no index is generated.
html_use_index = True
# If true, the index is split into individual pages for each letter.
html_split_index = False
# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/PMEAL/porespy",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/porespy",
            "icon": "fab fa-twitter-square",
        },
    ],
    "external_links": [
        {
            "name": "Issue Tracker", "url": "https://github.com/PMEAL/porespy/issues"
        },
        {
            "name": "Get Help", "url": "https://github.com/PMEAL/porespy/discussions"
        },
    ],
    "navigation_with_keys": False,
    "show_prev_next": False,
    "icon_links_label": "Quick Links",
    "use_edit_page_button": False,
    "search_bar_position": "sidebar",
    "navbar_align": "left",
}

html_sidebars = {
   "blog": ['ablog/recentposts.html', 'ablog/archives.html', ],
   "blog/**": ["ablog/postcard.html", 'ablog/archives.html', ],
}


#------------------------------------------------------------------------#
# Options for HTMLHelp output                                            #
#------------------------------------------------------------------------#

# Output file base name for HTML help builder.
htmlhelp_basename = 'PoreSpydoc'

#------------------------------------------------------------------------#
# Options for ABlog                                                      #
#------------------------------------------------------------------------#
blog_baseurl = "https://porespy.org"
blog_title = "PoreSpy"
blog_path = "blog"
blog_post_pattern = "blog/*/*"
blog_feed_fulltext = True
blog_feed_subtitle = "News, Updates, and Demos"
fontawesome_included = True
post_redirect_refresh = 1
post_auto_image = 1
post_auto_excerpt = 2
