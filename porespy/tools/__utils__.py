import os
import sys
import importlib
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm


def _is_ipython_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


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
    loglevel : str
        Determines what messages to get printed in console. Options are:
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"

    """
    __instance__ = None
    notebook = False
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
    _loglevel = "ERROR" if _is_ipython_notebook() else "INFO"
    config_logger(_logger_fmt, _loglevel)

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
