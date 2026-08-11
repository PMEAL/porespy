PoreSpy settings and parallel execution
########################################

PoreSpy exposes its process-wide configuration as ``porespy.settings``.  The
``settings.ncores`` value controls the number of threads passed to the
``edt.edt`` distance-transform backend and is also used as the default worker
count by selected PoreSpy routines.  It can be changed at any time::

    import porespy as ps

    ps.settings.ncores = 4

The setting persists until it is explicitly changed.  Changes apply to
subsequent EDT calls even when a module obtained its EDT callable before the
change.  A direct call can still override it with ``parallel=...``.

By default, PoreSpy uses the number of logical CPUs available to the current
process.  CPU affinity is preferred when the operating system supports it;
the logical CPU count is used as a fallback.  On Linux, a cgroup CPU quota can
further reduce the default.  The result is always at least one.  Assigning
``None`` to ``settings.ncores`` selects this default again, while assigning a
positive integer provides an explicit value.

PoreSpy prefers ``pyedt`` when that optional backend is installed.  Its
``edt`` function does not accept the ``parallel`` keyword, so ``ncores`` is not
passed to it.  The separate parallel network-extraction path that requires
``pyedt`` is configured with its ``threads`` option.

``settings.ncores`` does not configure every parallel facility in PoreSpy.
Chunk decomposition (``divs`` and ``overlap``), Dask scheduling, and Numba
threading have their own controls.  Their documentation identifies the
appropriate settings for each algorithm.
