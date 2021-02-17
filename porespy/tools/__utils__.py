from dataclasses import dataclass
import importlib


@dataclass
class Settings:
    r"""
    A dataclass for use at the module level to store settings.  This class
    is defined as a Singleton so now matter how or where it gets instantiated
    the same object is returned, containing all existing settings.
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
        s = ''
        for item in self.__dir__():
            if not item.startswith('_'):
                s += ''.join((item, ':\t'))
                s += ''.join((getattr(self, item).__repr__(), '\n'))
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
    Fetches docstring for function and returns in markdown format

    Useful for printing in a Jupyternote book

    Parameters
    ----------
    func : object
        Function handle to function whose docstring is desired

    Returns
    -------
    md : str
        A text string with the markdown syntax included suitable for printing in
        a Jupyter notebook with the ``IPython.display.Markdown`` function.
    """
    from npdoc_to_md import render_md_from_obj_docstring
    title = f'---\n ## Documentation for {func.__name__}\n ---\n'
    txt = render_md_from_obj_docstring(obj=func, obj_namespace=func.__name__)
    return title + txt + '\n---'
