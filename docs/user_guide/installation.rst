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
 
   $ conda install -c conda-forge porespy

It's possible to use ``pip install porespy``, but this will not result
in a full installation and some features won't work (i.e. outputing to
paraview and calling imagej functions).

Installing the dev version
##########################
We frequently publish new releases every couple of months, but if you
still want to use the latest features available on the `dev` branch,
(but not yet officially released), you need to do the followings:

Open up the terminal/cmd and ``cd`` to the directory you want to clone ``porespy``.

Clone the repo somewhere in your disk using::

   $ git clone https://github.com/PMEAL/porespy

``cd`` to the root folder of ``porespy``::

   $ cd porespy

Install ``porespy`` dependencies::

   $ conda install --file=requirements/conda_requirements.txt
   $ pip install -r requirements/pip_requirements.txt

Install ``porespy`` in "editable" mode::

   $ pip install --no-deps -e .

Voila! You can now use the latest features available on the ``dev`` branch. To
keep your "local" ``porespy`` installation up to date, every now and then, ``cd``
to the root folder of ``porespy`` and pull the latest changes::

   $ git pull

.. warning::
   For the development version of ``porespy`` to work, you need to first remove
   the ``porespy`` that you've previously installed using ``pip`` or ``conda``.

Where's my ``conda`` prompt?
###################################
All the commands in this page need to be typed in the ``conda`` prompt.

.. tabbed:: Windows

   On Windows you should have a shortcut to the "Anaconda prompt" in the
   Anaconda program group in the start menu. This will open a Windows
   command console with access to the Python features added by *conda*,
   such as installing things via ``conda``.
   
.. tabbed:: Mac and Linux

   On Mac or Linux, you need to open a normal terminal window, then type
   ``source activate env`` where you replace ``env`` with the name of
   the environment you want to install PoreSpy. If you don't know what this
   means, then use ``source activate base``, which will install PoreSpy in
   the base environment which is the default.
