.. _installation:

############
Installation
############

PoreSpy depends heavily on SciPy and its dependencies. The best way to get a fully
functional environment is the `Anaconda
distribution <https://www.anaconda.com/products/individual#Downloads>`__. Be sure to get the
**Python 3.10+ version**.

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
(but not yet officially released), you have two options:

The easy way
------------
If you're looking for an easy way to install the development version of
``porespy`` and use the latest features, you can install it using::

   $ pip install git+https://github.com/PMEAL/porespy.git@dev

.. warning::
   This approach is not recommended if you are a porespy contributor or
   want to frequently get new updates as they roll in. If you insist on
   using this approach, to get the latest version at any point, you
   need to first uninstall your porespy and then rerun the command above.

The hard (but correct) way
--------------------------
If you are a porespy contributor or want to easily get the new updates as
they roll in, you need to properly clone our repo and install it locally.
It's not as difficult as it sounds, just follow these steps:

Open up the terminal/cmd and ``cd`` to the directory you want to clone ``porespy``.

Clone the repo somewhere in your disk using::

   $ git clone https://github.com/PMEAL/porespy

``cd`` to the root folder of ``porespy``::

   $ cd porespy

Install ``porespy`` dependencies::

   $ conda install --file=requirements/conda.txt
   $ pip install -r requirements.txt

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


.. tab-set::

   .. tab-item:: Windows

      On Windows you should have a shortcut to the "Anaconda prompt" in the
      Anaconda program group in the start menu. This will open a Windows
      command console with access to the Python features added by *conda*,
      such as installing things via ``conda``.

   .. tab-item:: Mac and Linux

      On Mac or Linux, you need to open a normal terminal window, then type
      ``source activate env`` where you replace ``env`` with the name of
      the environment you want to install PoreSpy. If you don't know what this
      means, then use ``source activate base``, which will install PoreSpy in
      the base environment which is the default.
