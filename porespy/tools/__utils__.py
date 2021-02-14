

class SettingsDict(dict):
    r"""
    A settings dictionary for use at the moduel level.  This class is a
    singleton so can be instantiated anywhere within the package to access
    the common settings.
    """

    __instance__ = None

    def __new__(cls):
        if SettingsDict.__instance__ is None:
            SettingsDict.__instance__ = dict.__new__(cls)
        return SettingsDict.__instance__

    def __init__(self):
        super().__init__()
        self.__dict__ = self
        # Apply package settings
        self['show_progress'] = True
