from dataclasses import dataclass
import importlib


@dataclass
class Settings:
    r"""
    A dataclass for use at the module level to store settings.  This class
    is defined as a Singleton so now matter how or where it gets instantiated
    the same object is returned, containing all existing settings.

    Parameters
    ----------
    notebook : boolean
        Is automatically determined upon initialization of PoreSpy, and is
        ``True`` if running within a Jupyter notebook and ``False`` otherwise.
        This is used by the ``porespy.tools.get_tqdm`` function to determine
        whether a standard or a notebook version of the progress bar should
        be used.
    tqdm : dict
        This dictionary is passed directly to the the ``tqdm`` function
        throughout PoreSpy (i.e. ``for i in tqdm(range(N), **settings.tqdm)``).
        To see a list of available options visit the tqdm website.  Probably
        the most important is ``'disable'`` which when set to ``True`` will
        silence the progress bars.  It's also possible to adjust the formatting
        such as ``'colour'`` and ``'ncols'``, which controls width.

    """
    __instance__ = None
    notebook = False
    tqdm = {'disable': False,
            'colour': None,
            'ncols': None}

    def __new__(cls):
        if Settings.__instance__ is None:
            Settings.__instance__ = super().__new__(cls)
        return Settings.__instance__

    def __repr__(self):
        indent = 0
        for item in self.__dir__():
            if not item.startswith('_'):
                indent = max(indent, len(item) + 1)
        print(indent)
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


def get_tqdm():
    r"""
    Fetches a version of the ``tqdm`` function that depends on the environment

    Either text-based for the IPython console or gui-based for Jupyter
    notebooks.

    Returns
    -------
    tqdm : function handle
        The function to use when wrapping an iterator (i.e. tqdm(range(n)))
    """
    s = Settings()
    if s.notebook is True:
        tqdm = importlib.import_module('tqdm.notebook')
    else:
        tqdm = importlib.import_module('tqdm')
    return tqdm.tqdm


def show_docstring(func):
    r"""
    Fetches the docstring for a function and returns it in markdown format

    Useful for printing in a Jupyter notebook

    Parameters
    ----------
    func : object
        Function handle to function whose docstring is desired

    Returns
    -------
    md : str
        A text string with the markdown syntax included, suitable for printing in
        a Jupyter notebook using the ``IPython.display.Markdown`` function.
    """
    title = f'---\n ## Documentation for {func.__name__}\n ---\n'
    try:
        from npdoc_to_md import render_md_from_obj_docstring
        txt = render_md_from_obj_docstring(obj=func, obj_namespace=func.__name__)
    except ModuleNotFoundError:
        txt = func.__doc__
    return title + txt + '\n---'
