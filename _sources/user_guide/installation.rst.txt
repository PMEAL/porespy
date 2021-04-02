.. _installation:

############
Installation
############

PoreSpy depends heavily on the SciPy Stack. The best way to get a fully
functional environment is the `Anaconda
distribution <https://www.anaconda.com/products/individual#Downloads>`__. Be sure to get the
**Python 3.7+ version**.

Once you've installed *Anaconda* you can then install ``porespy``. It is
available on `conda-forge <https://anaconda.org/conda-forge/porespy>`__
and can be installed by typing the following at the *conda* prompt::

   conda install -c conda-forge porespy

It's possible to use ``pip install porespy``, but this will not result
in a full installation and some features won't work (i.e. outputing to
paraview and calling imagej functions).

Windows
-------

On Windows you should have a shortcut to the "Anaconda prompt" in the
Anaconda program group in the start menu. This will open a Windows
command console with access to the Python features added by *conda*,
such as installing things via ``conda``.

Mac and Linux
-------------

On Mac or Linux, you need to open a normal terminal window, then type
``source activate env`` where you replace ``env`` with the name of
the environment you want to install PoreSpy. If you don't know what this
means, then use ``source activate base``, which will install PoreSpy in
the base environment which is the default.
