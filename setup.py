import os
import sys
import codecs
import os.path
from distutils.util import convert_path
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

sys.path.append(os.getcwd())
ver_path = convert_path('porespy/__version__.py')


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            ver = line.split(delim)[1].split(".")
            if "dev0" in ver:
                ver.remove("dev0")
            return ".".join(ver)
    else:
        raise RuntimeError("Unable to find version string.")


# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='porespy',
    description='A set of tools for analyzing 3D images of porous materials',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=get_version(ver_path),
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    packages=[
        'porespy',
        'porespy.tools',
        'porespy.generators',
        'porespy.metrics',
        'porespy.filters',
        'porespy.filters.imagej',
        'porespy.networks',
        'porespy.dns',
        'porespy.simulations',
        'porespy.visualization',
        'porespy.io'
    ],
    install_requires=[
        'dask',
        'deprecated',
        'edt',
        'imageio',
        'loguru',
        'matplotlib',
        'numba',
        'numpy',
        'numpy-stl',
        'openpnm',
        'pandas',
        'psutil',
        'pyevtk',
        'pyfastnoisesimd',
        'scikit-fmm',
        'scikit-image',
        'scipy',
        'tqdm',
        'trimesh',
    ],
    author='PoreSpy Team',
    author_email='jgostick@gmail.com',
    download_url='https://github.com/PMEAL/porespy/',
    url='http://porespy.org',
    project_urls={
        'Documentation': 'https://porespy.org/',
        'Source': 'https://github.com/PMEAL/porespy/',
        'Tracker': 'https://github.com/PMEAL/porespy/issues',
    },
)
