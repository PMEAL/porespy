import imagej
import numpy as np
from scyjava import jimport

def imagej_wrapper(im, plugin_name, path):
    r"""
    Apply ImageJ filters on 3D images
    
    Parameters
    ----------
    im : ndarray
        The 3D image of the porous material
        
    Plugin_name : str
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
            
    Path: str
        path to the Fiji application in the local directory
        
    Returns
    -----
    Outputs a ndarray after applying the desired filter on the image
    """
    ij = imagej.init(path, headless=False)
    i = False
    img = 255 * np.array(im.astype("uint8"))
    WindowManager = jimport('ij.WindowManager')
    #ij.ui().showUI()
    ij.ui().show('Image', ij.py.to_java(img))
    plugin = 'Duplicate...'
    args = {
        'duplicate range': (1,im.shape[0])
        }
    ij.py.run_plugin(plugin, args)
    if 'watershed' in plugin_name.lower():
         ij.py.run_macro("""run("Distance Transform Watershed 3D",
                         "options=True");""")
         ij.py.run_macro("""run("3-3-2 RGB");""")
    elif 'erosion' in plugin_name.lower(): 
        plugin = 'Morphological Filters (3D)'
        args = {
            'operation': 'Erosion'
            }
        ij.py.run_plugin(plugin, args)
    elif 'dilation' in plugin_name.lower():
        plugin = 'Morphological Filters (3D)'
        args = {
            'operation': 'Dilation'
            }
        ij.py.run_plugin(plugin, args)
    elif 'opening' in plugin_name.lower():
        plugin = 'Morphological Filters (3D)'
        args = {
            'operation': 'Opening'
            }
        ij.py.run_plugin(plugin, args)
    elif 'closing' in plugin_name.lower():
        plugin = 'Morphological Filters (3D)'
        args = {
            'operation': 'Closing'
            }
        ij.py.run_plugin(plugin, args)
    elif 'median' in plugin_name.lower():
        plugin = 'Median 3D...'
        args = {
            "options" : "True"
            }
        ij.py.run_plugin(plugin, args)
        i= True
    elif 'mean' in plugin_name.lower():
        plugin = 'Mean 3D...'
        args = {
            "options" : "True"
            }
        ij.py.run_plugin(plugin, args)
        i= True
    elif 'gaussian' in plugin_name.lower():
        plugin = 'Gaussian Blur 3D...'
        args = {
            "options" : "True"
            }
        ij.py.run_plugin(plugin, args)
        i= True
    elif 'maximum' in plugin_name.lower():
        plugin = 'Maximum 3D...'
        args = {
            "options" : "True"
            }
        ij.py.run_plugin(plugin, args)
        i= True  
    elif 'minimum' in plugin_name.lower():
        plugin = 'Minimum 3D...'
        args = {
            "options" : "True"
            }
        ij.py.run_plugin(plugin, args)
        i= True  
    results = np.array(ij.py.from_java(WindowManager.getCurrentImage()))
    if i:
        results = np.moveaxis(results, -1, 0)
    ij.getContext().dispose()
    WindowManager.closeAllWindows()
    return (results) 