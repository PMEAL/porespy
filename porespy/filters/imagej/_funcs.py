import numpy as np
from loguru import logger


def imagej_wrapper(im, plugin_name, path):  # pragma: no cover
    r"""
    Apply ImageJ filters on 3D images.

    Parameters
    ----------
    im : ndarray
        The 3D image of the porous material.

    plugin_name : str
        Name of the applied ImageJ plugin which could be one of these options:
            - Distance Transform Watershed 3D
            - Dilation 3D
            - Erosion 3D
            - Opening 3D
            - Closing 3D
            - Median 3D
            - Gaussian Blur 3D
            - Minimum 3D
            - Maximum 3D
            - Mean 3D

    path: str
        path to the Fiji application in the local directory

    Returns
    -------
    ndarray
        Outputs a ndarray after applying the desired filter on the image.

    """
    try:
        import imagej
        from scyjava import jimport
    except ModuleNotFoundError:
        msg = ("The pyimagej python bindings must be installed using conda"
               " install -c conda-forge paraview, however this may require"
               " using a virtualenv since conflicts with other packages are"
               " common. This is why it is not explicitly included as a"
               " dependency in porespy.")
        logger.critical(msg)

    i = False
    ij = imagej.init(path, headless=False)
    img = 255 * np.array(im.astype("uint8"))
    WindowManager = jimport("ij.WindowManager")
    ij.ui().show("Image", ij.py.to_java(img))
    plugin = "Duplicate..."
    args = {"duplicate range": (1, im.shape[0])}
    ij.py.run_plugin(plugin, args)

    if "watershed" in plugin_name.lower():
        ij.py.run_macro("""run("Distance Transform Watershed 3D","options=True");""")
        ij.py.run_macro("""run("3-3-2 RGB");""")
    elif "erosion" in plugin_name.lower():
        plugin = "Morphological Filters (3D)"
        args = {"operation": "Erosion"}
        ij.py.run_plugin(plugin, args)
    elif "dilation" in plugin_name.lower():
        plugin = "Morphological Filters (3D)"
        args = {"operation": "Dilation"}
        ij.py.run_plugin(plugin, args)
    elif "opening" in plugin_name.lower():
        plugin = "Morphological Filters (3D)"
        args = {"operation": "Opening"}
        ij.py.run_plugin(plugin, args)
    elif "closing" in plugin_name.lower():
        plugin = "Morphological Filters (3D)"
        args = {"operation": "Closing"}
        ij.py.run_plugin(plugin, args)
    elif "median" in plugin_name.lower():
        plugin = "Median 3D..."
        args = {"options": "True"}
        ij.py.run_plugin(plugin, args)
        i = True
    elif "mean" in plugin_name.lower():
        plugin = "Mean 3D..."
        args = {"options": "True"}
        ij.py.run_plugin(plugin, args)
        i = True
    elif "gaussian" in plugin_name.lower():
        plugin = "Gaussian Blur 3D..."
        args = {"options": "True"}
        ij.py.run_plugin(plugin, args)
        i = True
    elif "maximum" in plugin_name.lower():
        plugin = "Maximum 3D..."
        args = {"options": "True"}
        ij.py.run_plugin(plugin, args)
        i = True
    elif "minimum" in plugin_name.lower():
        plugin = "Minimum 3D..."
        args = {"options": "True"}
        ij.py.run_plugin(plugin, args)
        i = True
    results = np.array(ij.py.from_java(WindowManager.getCurrentImage()))
    if i:
        results = np.moveaxis(results, -1, 0)
    ij.getContext().dispose()
    WindowManager.closeAllWindows()
    return results


def imagej_plugin(im, path, plugin_name, args=None):  # pragma: no cover
    r"""
    Apply ImageJ filters on 3D images.

    In This function the plugin_name should have a same format as the
    plugin_name in the ImageJ. For example, to apply a Gaussian blur on a 3D
    image, the plugin_name should be 'Gaussian Blur 3D...'

    Parameters
    ----------
    im : ndarray
        The 3D image of the porous material

    path: str
        Path to the Fiji application in the local directory

    plugin_name : str
        Name of the applied ImageJ plugin

    args : dict
         A dictionary that containes the required arguments of the
         applied plugin. For example, it could be {'options': 'True'}

    Returns
    -------
    ndarray
        Outputs a ndarray after applying the desired filter on the image.

    """
    try:
        import imagej
        from scyjava import jimport
    except ModuleNotFoundError:
        msg = ("The pyimagej python bindings must be installed using conda"
               " install -c conda-forge paraview, however this may require"
               " using a virtualenv since conflicts with other packages are"
               " common. This is why it is not explicitly included as a"
               " dependency in porespy.")
        logger.critical(msg)
    ij = imagej.init(path, headless=False)
    img = 255 * np.array(im.astype("uint8"))
    WindowManager = jimport('ij.WindowManager')
    ij.ui().show('Image', ij.py.to_java(img))
    plugin = 'Duplicate...'
    arg = {'duplicate range': (1, im.shape[0])}
    ij.py.run_plugin(plugin, arg)
    ij.py.run_plugin(plugin_name, args)
    results = np.array(ij.py.from_java(WindowManager.getCurrentImage()))
    ij.getContext().dispose()
    WindowManager.closeAllWindows()
    return (results)
