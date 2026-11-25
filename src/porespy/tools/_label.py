from typing import Literal

import cc3d
import numpy as np
import numpy.typing as npt
import scipy.ndimage as spim

from ._morphology import get_strel


def _label_components(
    im: npt.NDArray,
    conn: Literal["max", "min"] = "min",
):
    """Label a 2D or 3D binary image using the matching PoreSpy connectivity."""
    connectivity = {
        2: {"min": 4, "max": 8},
        3: {"min": 6, "max": 26},
    }
    if im.ndim in connectivity:
        return cc3d.connected_components(
            im,
            connectivity=connectivity[im.ndim][conn],
            binary_image=True,
            return_N=True,
        )
    structure = get_strel()[im.ndim][conn].copy()
    return spim.label(input=im, structure=structure)


def _isin_labels(labels, hits, N, invert=False, out=None):
    """Select labels using a compact lookup table instead of a full-volume isin."""
    lookup = np.zeros(N + 1, dtype=bool)
    lookup[np.asarray(hits, dtype=np.intp)] = True
    lookup[0] = False
    if out is None:
        out = np.empty(labels.shape, dtype=bool)
    np.take(lookup, labels, out=out)
    if invert:
        np.logical_not(out, out=out)
    return out
