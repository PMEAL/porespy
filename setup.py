import os
import sys
from distutils.util import convert_path
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

sys.path.append(os.getcwd())

about = {}
ver_path = convert_path('porespy/__version__.py')
with open(ver_path) as f:
    for line in f:
        if line.startswith('__version__'):
            exec(line, about)

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='porespy',
    description='A set of tools for analyzing 3D images of porous materials',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=about['__version__'],
    zip_safe=False,
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Physics'],
    packages=['porespy',
              'porespy.tools',
              'porespy.generators',
              'porespy.metrics',
              'porespy.filters',
              'porespy.networks',
              'porespy.dns',
              'porespy.visualization',
              'porespy.io'],
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'scikit-image',
                      'pandas',
                      'imageio',
                      'tqdm',
                      'array_split',
                      'pytrax',
                      'pyevtk',
                      'numba',
                      'openpnm',
                      'dask[complete]',
                      'edt'],
    author='PoreSpy Team',
    author_email='jgostick@gmail.com',
    download_url='https://github.com/PMEAL/porespy/',
    url='http://porespy.org',
    project_urls={
        'Documentation': 'https://porespy.readthedocs.io/en/dev/',
        'Source': 'https://github.com/PMEAL/porespy/',
        'Tracker': 'https://github.com/PMEAL/porespy/issues',
    },
)
