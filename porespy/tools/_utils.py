import os
import sys
import numpy as np
import importlib
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm
import psutil


__all__ = [
    'sanitize_filename',
    'get_tqdm',
    'show_docstring',
    'Results',
]


def _is_ipython_notebook():  # pragma: no cover
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True     # Jupyter notebook or qtconsole
        if shell == 'TerminalInteractiveShell':
            return False    # Terminal running IPython
        return False        # Other type (?)
    except NameError:
        return False        # Probably standard Python interpreter


def config_logger(fmt, loglevel):  # pragma: no cover
    r"""
    Configures loguru logger with the given format and log level.

    Parameters
    ----------
    fmt : str
        loguru-compatible format used to format logger messages.
    loglevel : str
        Determines what messages to get printed in console. Options are:
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"

    Returns
    -------
    None.

    """
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""),
               level=loglevel,
               format=fmt,
               colorize=True)


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
    logger_fmt : str
        luguru-compatible format used to format the logger messages.
    loglevel : str, or int
        Determines what messages to get printed in console. Options are:
        "TRACE" (5), "DEBUG" (10), "INFO" (20), "SUCCESS" (25), "WARNING" (30),
        "ERROR" (40), "CRITICAL" (50)

    """
    __instance__ = None
    # Might need to add 'file': sys.stdout to tqdm dict
    tqdm = {'disable': False,
            'colour': None,
            'ncols': None,
            'leave': False,
            'file': sys.stdout}
    _logger_fmt = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | ' \
          '<level>{level: <8}</level> | ' \
          '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>' \
          '\n--> <level>{message}</level>'
    _loglevel = "ERROR" if _is_ipython_notebook() else "WARNING"
    config_logger(_logger_fmt, _loglevel)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._notebook = None
        self._ncores = psutil.cpu_count(logical=False)

    @property
    def logger_fmt(self):
        return self._logger_fmt

    @property
    def loglevel(self):
        return self._loglevel

    @logger_fmt.setter
    def logger_fmt(self, value):
        self._logger_fmt = value
        config_logger(fmt=value, loglevel=self.loglevel)

    @loglevel.setter
    def loglevel(self, value):
        if isinstance(value, int):
            options = {5: "TRACE",
                       10: "DEBUG",
                       20: "INFO",
                       25: "SUCESS",
                       30: "WARNING",
                       40: "ERROR",
                       50: "CRITICAL"}
            value = options[value]
        self._loglevel = value
        os.environ["LOGURU_LEVEL"] = value
        config_logger(fmt=self.logger_fmt, loglevel=value)

    def __new__(cls):
        if Settings.__instance__ is None:
            Settings.__instance__ = super().__new__(cls)
        return Settings.__instance__

    def __repr__(self):
        indent = 0
        for item in self.__dir__():
            if not item.startswith('_'):
                indent = max(indent, len(item) + 1)
        s = ''
        for item in self.__dir__():
            if not item.startswith('_'):
                s += ''.join((item, ':', ' '*(indent-len(item))))
                attr = getattr(self, item)
                temp = ''.join((attr.__repr__(), '\n'))
                if isinstance(attr, dict):
                    temp = temp.replace(',', '\n' + ' '*(indent + 1))
                s += temp
        return s

    def _get_ncores(self):
        if self._ncores is None:
            self._ncores = psutil.cpu_count(logical=False)
        return self._ncores

    def _set_ncores(self, val):
        cpu_count = psutil.cpu_count(logical=False)
        if val is None:
            val = cpu_count
        elif val > cpu_count:
            logger.error('Value is more than the available number of cores')
            val = cpu_count
        self._ncores = val

    ncores = property(fget=_get_ncores, fset=_set_ncores)

    def _get_notebook(self):
        if self._notebook is None:
            self._notebook = _is_ipython_notebook()
        return self._notebook

    def _set_notebook(self, val):
        logger.error('This value is determined automatically at runtime')

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
        tqdm = importlib.import_module('tqdm.notebook')
    else:
        tqdm = importlib.import_module('tqdm')
    return tqdm.tqdm


def show_docstring(func):  # pragma: no cover
    r"""
    Fetches the docstring for a function and returns it in markdown format.

    Useful for printing in a Jupyter notebook.

    Parameters
    ----------
    func : object
        Function handle to function whose docstring is desired

    Returns
    -------
    md : str
        A string with the markdown syntax included, suitable for printing
        in a Jupyter notebook using the ``IPython.display.Markdown``
        function.

    """
    title = f'---\n ## Documentation for ``{func.__name__}``\n ---\n'
    try:
        from npdoc_to_md import render_md_from_obj_docstring
        txt = render_md_from_obj_docstring(obj=func, obj_namespace=func.__name__)
    except ModuleNotFoundError:
        txt = func.__doc__
    return title + txt + '\n---'


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
    _value = "Description"
    _key = "Item"

    def __iter__(self):
        for item in self.__dict__.values():
            yield item

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __str__(self):
        header = "―" * 78
        lines = [header, "{0:<25s} {1}".format(self._key, self._value), header]
        for item in list(self.__dict__.keys()):
            if item.startswith('_'):
                continue
            if (isinstance(self[item], np.ndarray)):
                s = np.shape(self[item])
                if (self[item].ndim > 1):
                    lines.append("{0:<25s} Image of size {1}".format(item, s))
                else:
                    lines.append("{0:<25s} Array of size {1}".format(item, s))
            else:
                lines.append("{0:<25s} {1}".format(item, self[item]))
        lines.append(header)
        return "\n".join(lines)
