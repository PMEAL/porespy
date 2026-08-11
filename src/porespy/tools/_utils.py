import importlib
import inspect
import logging
import math
import sys
import time
import warnings
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np
import psutil

__all__ = [
    "sanitize_filename",
    "get_tqdm",
    "show_docstring",
    "Results",
    "tic",
    "toc",
    "get_edt",
    "get_skel",
    "parse_shape",
    "get_fixtures_path",
    "Settings",
]


logger = logging.getLogger("porespy")


def parse_shape(im_or_shape):
    r"""
    Given a list of dimensions or an image finds shape in a clean format

    Parameters
    ----------
    im_or_shape : scalar, list or ndarray
        Given a list of dimensions removes any `0`, `inf` or `None`
        values. Given an image removes any singleton dimensions and returns
        shape. If a scalar then assumes a 3D shape is requested.

    Returns
    -------
    shape : list
        List of [X, Y] or [X, Y, Z] dimensions
    """
    s = np.array(im_or_shape)
    if len(s) == 1:
        s = np.array([s] * 3).flatten()
    elif s.ndim > 1:  # if arg is an image
        s = s.squeeze()
        s = np.shape(s)
    shape = np.array([i for i in s if i not in [0, np.inf, None]], dtype=int)
    return shape


def get_skel():
    package = importlib.import_module("skimage.morphology")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            func = package.skeletonize_3d
        except (FutureWarning, AttributeError):
            func = package.skeletonize
    return func


def get_edt():
    r"""
    Return PoreSpy's preferred Euclidean distance transform callable.

    ``pyedt.edt`` is preferred when it is installed.  Otherwise, the callable
    wraps ``edt.edt`` and reads :attr:`porespy.settings.ncores` each time it is
    called.  Consequently, changing ``settings.ncores`` affects existing
    callables returned by this function.  An explicit ``parallel`` keyword at
    the call site takes precedence over the setting.

    Notes
    -----
    The ``pyedt`` API does not accept the ``parallel`` keyword, so PoreSpy does
    not pass ``settings.ncores`` to that backend.  Parallel network extraction
    using ``pyedt`` has its own ``threads`` option.
    """
    try:
        package = importlib.import_module("pyedt")
        return package.edt
    except ModuleNotFoundError:
        package = importlib.import_module("edt")
        backend = package.edt

        @wraps(backend)
        def edt(*args, **kwargs):
            kwargs.setdefault("parallel", Settings().ncores)
            return backend(*args, **kwargs)

        return edt


def _get_cgroup_cpu_limit():
    """Return the Linux cgroup CPU quota as a thread count, if available."""
    paths = [
        (Path("/sys/fs/cgroup/cpu.max"),),
        (
            Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        ),
    ]
    for group in paths:
        try:
            if len(group) == 1:
                quota_text, period_text = group[0].read_text().split()[:2]
                if quota_text == "max":
                    continue
                quota, period = int(quota_text), int(period_text)
            else:
                quota = int(group[0].read_text())
                period = int(group[1].read_text())
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    return None


def _get_available_cpu_count():
    """Return the logical CPUs available to this process, always at least one."""
    try:
        count = len(psutil.Process().cpu_affinity())
    except (AttributeError, NotImplementedError, TypeError, psutil.Error):
        count = None
    if not count:
        count = psutil.cpu_count(logical=True)
    count = max(1, count or 1)
    quota = _get_cgroup_cpu_limit()
    if quota is not None:
        count = min(count, quota)
    return max(1, int(count))


def _format_time(timespan, precision=3):
    """Formats the timespan in a human readable form"""

    if timespan >= 60.0:
        # we have more than a minute, format that in a human readable form
        # Idea from http://snipplr.com/view/5713/
        parts = [("d", 60 * 60 * 24), ("h", 60 * 60), ("min", 60), ("s", 1)]
        time = []
        leftover = timespan
        for suffix, length in parts:
            value = int(leftover / length)
            if value > 0:
                leftover = leftover % length
                time.append("%s%s" % (str(value), suffix))
            if leftover < 1:
                break
        return " ".join(time)

    # Unfortunately the unicode 'micro' symbol can cause problems in
    # certain terminals.
    # See bug: https://bugs.launchpad.net/ipython/+bug/348466
    # Try to prevent crashes by being more secure than it needs to
    # E.g. eclipse is able to print a µ, but has no sys.stdout.encoding set.
    units = ["s", "ms", "us", "ns"]  # the save value
    if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
        try:
            "\xb5".encode(sys.stdout.encoding)
            units = ["s", "ms", "\xb5s", "ns"]
        except UnicodeEncodeError:
            pass
    scaling = [1, 1e3, 1e6, 1e9]

    if timespan > 0.0:
        order = min(-int(np.floor(np.log10(timespan)) // 3), 3)
    else:
        order = 3
    return "%.*g %s" % (precision, timespan * scaling[order], units[order])


def tic():
    r"""
    Homemade version of matlab tic and toc function, tic starts or resets
    the clock, toc reports the time since the last call of tic.

    See Also
    --------
    toc

    """
    global _startTime_for_tictoc
    _startTime_for_tictoc = time.time()


def toc(quiet=False):
    r"""
    Homemade version of matlab tic and toc function, tic starts or resets
    the clock, toc reports the time since the last call of tic.

    Parameters
    ----------
    quiet : bool, default is False
        If False then a message is output to the console. If
        True the message is not displayed and the elapsed time is returned.

    See Also
    --------
    tic

    """
    if "_startTime_for_tictoc" not in globals():
        raise Exception("Start time not set, call tic first")
    t = time.time() - _startTime_for_tictoc
    if quiet is False:
        print(f"Elapsed time: {_format_time(t)}")
    return t


def _is_ipython_notebook():  # pragma: no cover
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        if shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


@dataclass
class Settings:  # pragma: no cover
    r"""
    A dataclass for use at the module level to store settings.  This class
    is defined as a Singleton so now matter how or where it gets
    instantiated the same object is returned, containing all existing
    settings.

    Parameters
    ----------
    notebook : boolean
        Is automatically determined upon initialization of PoreSpy, and is
        ``True`` if running within a Jupyter notebook and ``False``
        otherwise. This is used by the ``porespy.tools.get_tqdm`` function
        to determine whether a standard or a notebook version of the
        progress bar should be used.
    tqdm : dict
        This dictionary is passed directly to the the ``tqdm`` function
        throughout PoreSpy (``for i in tqdm(range(N), **settings.tqdm)``).
        To see a list of available options visit the tqdm website.
        Probably the most important is ``'disable'`` which when set to
        ``True`` will silence the progress bars.  It's also possible to
        adjust the formatting such as ``'colour'`` and ``'ncols'``, which
        controls width.
    loglevel : str, or int
        Determines what messages to get printed in console. Options are:
        ``'TRACE'`` (5), ``'DEBUG'`` (10), ``'INFO'`` (20), ``'SUCCESS'`` (25),
        ``'WARNING'`` (30), ``'ERROR'`` (40), ``'CRITICAL'`` (50)
    ncores : int
        Number of threads used by the fallback ``edt`` distance-transform
        backend and the default worker count for selected PoreSpy routines.
        The initial value is the number of logical CPUs available to this
        process, constrained by CPU affinity and Linux cgroup quota where
        available.  Assign a positive integer to override it.  Changes affect
        subsequent EDT calls, including calls through previously obtained
        :func:`get_edt` callables.  This setting is distinct from chunk layout
        (``divs`` and ``overlap``), Dask scheduling, and Numba thread settings.

    """

    __instance__ = None
    # Might need to add 'file': sys.stdout to tqdm dict
    tqdm = {
        "disable": True,
        "colour": None,
        "ncols": None,
        "leave": False,
        "file": sys.stdout,
    }
    _loglevel = 40
    # add parallel settings
    divs = 2  # choose 2 as default
    overlap = None

    def __init__(self, *args, **kwargs):
        if getattr(self, "_initialized", False):
            return
        super().__init__(*args, **kwargs)
        self._notebook = None
        self._ncores = _get_available_cpu_count()
        self._initialized = True

    @property
    def loglevel(self):
        return self._loglevel

    @loglevel.setter
    def loglevel(self, value):
        if isinstance(value, str):
            options = {
                "TRACE": 5,
                "DEBUG": 10,
                "INFO": 20,
                "SUCESS": 25,
                "WARNING": 30,
                "ERROR": 40,
                "CRITICAL": 50,
            }
            value = options[value]
        self._loglevel = value
        logger.setLevel(value)

    def __new__(cls):
        if Settings.__instance__ is None:
            Settings.__instance__ = super().__new__(cls)
        return Settings.__instance__

    def __repr__(self):
        indent = 0
        for item in self.__dir__():
            if not item.startswith("_"):
                indent = max(indent, len(item) + 1)
        s = ""
        for item in self.__dir__():
            if not item.startswith("_"):
                s += "".join((item, ":", " " * (indent - len(item))))
                attr = getattr(self, item)
                temp = "".join((attr.__repr__(), "\n"))
                if isinstance(attr, dict):
                    temp = temp.replace(",", "\n" + " " * (indent + 1))
                s += temp
        return s

    def _get_ncores(self):
        if self._ncores is None:
            self._ncores = _get_available_cpu_count()
        return self._ncores

    def _set_ncores(self, val):
        if val is None:
            val = _get_available_cpu_count()
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise TypeError("ncores must be a positive integer or None")
        if val < 1:
            raise ValueError("ncores must be at least 1")
        self._ncores = int(val)

    ncores = property(fget=_get_ncores, fset=_set_ncores)

    def _get_notebook(self):
        if self._notebook is None:
            self._notebook = _is_ipython_notebook()
        return self._notebook

    def _set_notebook(self, val):
        logger.error("This value is determined automatically at runtime")

    notebook = property(fget=_get_notebook, fset=_set_notebook)


def get_tqdm():  # pragma: no cover
    r"""
    Fetches a version of ``tqdm`` function that depends on the environment.

    Either text-based for the IPython console or gui-based for Jupyter
    notebooks.

    Returns
    -------
    tqdm : function handle
        The function to use when wrapping an iterator (i.e. tqdm(range(n)))

    """
    if Settings().notebook is True:
        tqdm = importlib.import_module("tqdm.notebook")
    else:
        tqdm = importlib.import_module("tqdm")
    return tqdm.tqdm


def show_docstring(func, fold=True, method='pandoc'):  # pragma: no cover
    r"""
    Fetches the docstring for a function and returns it in markdown format.

    Useful for printing in a Jupyter notebook.

    Parameters
    ----------
    func : object
        Handle to function whose docstring is desired

    Returns
    -------
    md : str
        A string with the markdown syntax included, suitable for printing
        in a Jupyter notebook using the ``IPython.display.Markdown``
        function.

    """
    # Note: The following could work too:
    # import pandoc
    # Markdown(pandoc.write(pandoc.read(func, format='rst'), format='markdown'))
    # Although the markdown conversion is not numpydoc specific so is less pretty
    if method == 'npdoc_to_md':
        from npdoc_to_md import render_obj_docstring
        name = func.__module__.rsplit(".", 1)[0] + "." + func.__name__
        txt = render_obj_docstring(name)
    elif method == 'pandoc':
        import pandoc
        txt = pandoc.write(pandoc.read(func.__doc__, format='rst'), format='html')
    elif method in ['none', None]:
        txt = func.__doc__
    # The following creates an accordian around text
    if fold:
        txt = fr"<details><summary><b>Click to see docs</b></summary>{txt}</details>"
    return txt


def sanitize_filename(filename, ext, exclude_ext=False):
    r"""
    Returns a sanitized string in the form of name.extension

    Parameters
    ----------
    filename : str
        Unsanitized filename, could be 'test.vtk' or just 'test'

    ext : str
        Extension of the file, could be 'vtk'

    exclude_ext : bool
        If True, the returned string doesn't have the extension

    Returns
    -------
    sanitized : str
        Sanitized filename in form of name.extension

    """
    ext.strip(".")
    if filename.endswith(f".{ext}"):
        name = ".".join(filename.split(".")[:-1])
    else:
        name = filename
    filename_formatted = f"{name}" if exclude_ext else f"{name}.{ext}"
    return filename_formatted


class Results:
    r"""
    A minimal class for use when returning multiple values from a function

    This class supports dict-like assignment and retrieval
    (``obj['im'] = im``), namedtuple-like attribute look-ups (``obj.im``),
    and generic class-like object assignment (``obj.im = im``)

    """

    # Resist the urge to add method to this class...the point is to keep
    # the namespace clean!!

    def __init__(self, **kwargs):
        self._func = inspect.getouterframes(inspect.currentframe())[1].function
        self._time = time.asctime()

    def __iter__(self):
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                yield v

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __str__(self):
        header = "―" * 78
        lines = [
            header,
            f"Results of {self._func} generated at {self._time}",
            header,
        ]
        for item in list(self.__dict__.keys()):
            if item.startswith("_"):
                continue
            if isinstance(self[item], np.ndarray):
                s = np.shape(self[item])
                lines.append("{0:<25s} Array of size {1}".format(item, s))
            elif hasattr(self[item], "keys"):
                N = len(self[item].keys())
                lines.append("{0:<25s} Dictionary with {1} items".format(item, N))
            else:
                lines.append("{0:<25s} {1}".format(item, self[item]))
        lines.append(header)
        return "\n".join(lines)


def get_fixtures_path():
    r"""
    Get the path to the test fixtures directory.

    Returns
    -------
    Path
        Path object pointing to the test/fixtures directory relative to the
        package root.
    """
    # Get the package root directory (where porespy package is)
    # This file is at src/porespy/tools/_utils.py, so we go up 3 levels
    pkg_root = Path(__file__).parent.parent.parent.parent
    fixtures_path = pkg_root / "test" / "fixtures"
    return fixtures_path
