import logging

import numba
import numpy as np
import scipy.ndimage as spim
from numba import njit, prange
from numba.core import types
from numba.typed import Dict, List

from porespy.tools import (
    center_of_mass,
    create_mc_template_list,
    get_tqdm,
    jit_extend_slice,
    jit_marching_cubes_area_and_volume,
    jit_marching_squares_perimeter_and_area,
    make_contiguous,
    pad,
)

try:
    from pyedt import edt, jit_edt_cpu
except ModuleNotFoundError:
    from edt import edt


IDLE = np.uint32(0)
ASSIGNED = np.uint32(1)
DONE = np.uint32(2)
FINISHED = np.uint32(3)

FLOAT_TYPE = numba.types.float64[:]
INT_TYPE = numba.types.int64[:]


__all__ = [
    "regions_to_network_parallel",
    "_jit_regions_to_network",
]


tqdm = get_tqdm()
logger = logging.getLogger(__name__)


@njit
def wait():
    np.random.binomial(1, 0.5, 1000)


def calculate_throat_perimeter(vx, sub_dt, voxel_size):
    # Directions used to evaluate geom properties
    # dirs = [
    #     (1, 0, 0),
    #     (0, 1, 0),
    #     (0, 0, 1),
    #     (-1, 0, 0),
    #     (0, -1, 0),
    #     (0, 0, -1),
    # ]
    # voxel_size = np.array(voxel_size)
    # voxel_areas = np.array([
    #     voxel_size[1] * voxel_size[2],
    #     voxel_size[2] * voxel_size[0],
    #     voxel_size[0] * voxel_size[1],
    # ])

    dx = voxel_size[0]*(max(vx[0]) - min(vx[0]))
    dy = voxel_size[1]*(max(vx[1]) - min(vx[1]))
    dz = voxel_size[2]*(max(vx[2]) - min(vx[2]))
    normal = np.array([dx*dy, dy*dz, dz*dx], dtype=np.float64)
    norm = np.linalg.norm(normal)

    if norm != 0:
        normal /= norm
        t_perimeter_loc = np.sum(sub_dt[vx] < 2*max(voxel_size)) \
            * np.linalg.norm(normal * voxel_size)
        if t_perimeter_loc < 4. * np.linalg.norm(normal * voxel_size):
            t_perimeter_loc = 4. * np.linalg.norm(normal * voxel_size)
    else:  # cases where the throat cross section is aligned in a line of voxels
        t_perimeter_loc = np.sum(sub_dt[vx] < 2*voxel_size[0]) * voxel_size[0]
        if t_perimeter_loc < 4. * voxel_size[0]:
            t_perimeter_loc = 4. * voxel_size[0]

    return t_perimeter_loc


def regions_to_network_parallel(
    regions,
    phases=None,
    voxel_size=(1, 1, 1),
    accuracy='standard',
    porosity_map=None,
    threads=None,
):
    r"""
    Analyzes an image that has been partitioned into pore regions and extracts
    the pore and throat geometry as well as network connectivity.

    Parameters
    ----------
    regions : ndarray
        An image of the material partitioned into individual regions.
        Zeros in this image are ignored.
    phases : ndarray, optional
        An image indicating to which phase each voxel belongs. The returned
        network contains a 'pore.phase' array with the corresponding value.
        If not given a value of 1 is assigned to every pore.
    voxel_size : tuple (default = (1, 1, 1))
        The resolution of the image, expressed as the length of the sides of a
        voxel, so the volume of a voxel would be the product of **voxel_size**
        coords.
    accuracy : string
        Controls how accurately certain properties are calculated. Options are:

        'standard' (default)
            Computes the surface areas and perimeters by simply counting
            voxels.  This is *much* faster but does not properly account
            for the rough, voxelated nature of the surfaces.
        'high'
            Computes surface areas using the marching cube method, and
            perimeters using the fast marching method.  These are substantially
            slower but better account for the voxelated nature of the images.

    Returns
    -------
    net : dict
        A dictionary containing all the pore and throat size data, as well as
        the network topological information.  The dictionary names use the
        OpenPNM convention (i.e. 'pore.coords', 'throat.conns').

    Notes
    -----
    The meaning of each of the values returned in ``net`` are outlined below:

    'pore.region_label'
        The region labels corresponding to the watershed extraction. The
        pore indices and regions labels will be offset by 1, so pore 0
        will be region 1.
    'throat.conns'
        An *Nt-by-2* array indicating which pores are connected to each other
    'pore.region_label'
        Mapping of regions in the watershed segmentation to pores in the
        network
    'pore.local_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the pore region in isolation
    'pore.global_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the full image
    'pore.geometric_centroid'
        The center of mass of the pore region as calculated by
        ``skimage.measure.center_of_mass``
    'throat.global_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the full image
    'pore.region_volume'
        The volume of the pore region computed by summing the voxels
    'pore.volume'
        The volume of the pore found by as volume of a mesh obtained from the
        marching cubes algorithm
    'pore.surface_area'
        The surface area of the pore region as calculated by either counting
        voxels or using the fast marching method to generate a tetramesh (if
        ``accuracy`` is set to ``'high'``.)
    'throat.cross_sectional_area'
        The cross-sectional area of the throat found by either counting
        voxels or using the fast marching method to generate a tetramesh (if
        ``accuracy`` is set to ``'high'``.)
    'throat.perimeter'
        The perimeter of the throat found by counting voxels on the edge of
        the region defined by the intersection of two regions.
    'pore.inscribed_diameter'
        The diameter of the largest sphere inscribed in the pore region. This
        is found as the maximum of the distance transform on the region in
        isolation.
    'pore.extended_diameter'
        The diamter of the largest sphere inscribed in overal image, which
        can extend outside the pore region. This is found as the local maximum
        of the distance transform on the full image.
    'throat.inscribed_diameter'
        The diameter of the largest sphere inscribed in the throat.  This
        is found as the local maximum of the distance transform in the area
        where to regions meet.
    'throat.total_length'
        The length between pore centered via the throat center.
    'throat.direct_length'
        The length between two pore centers on a straight line between them
        that does not pass through the throat centroid.
    'pore.phase'
        Highest phase label in the pore.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/networks/reference/regions_to_network.html>`_
    to view online example.

    """
    logger.trace('Extracting pore/throat information')
    template_areas, template_volumes = create_mc_template_list(spacing=voxel_size)
    vertex_index_array = np.array([2**i for i in range(8)])
    vertex_index_array = vertex_index_array.reshape((2, 2, 2), order="F")

    if threads is None:
        threads = (numba.config.NUMBA_NUM_THREADS - 2) // 2

    im = make_contiguous(regions)
    # struc_elem = disk if im.ndim == 2 else ball
    voxel_size = tuple([float(i) for i in voxel_size])
    if phases is None:
        phases = (im > 0).astype(int)
    if im.size != phases.size:
        raise Exception('regions and phase are different sizes, probably ' +
                        'because boundary regions were not added to phases')
    dt = edt(phases >= 1, scale=voxel_size)

    # Get 'slices' into im for each pore region
    slices = spim.find_objects(im)
    flat_slices = []
    for sl in slices:
        flat_slices.extend(sl)

    net = _jit_regions_to_network(
        im=im,
        dt=dt,
        flat_slices=flat_slices,
        template_areas=template_areas,
        template_volumes=template_volumes,
        phases=phases,
        voxel_size=voxel_size,
        porosity_map=porosity_map,
        threads=threads,
    )

    net = dict(net)
    net_keys = [k for k in net.keys() if k[-6:] == "_int64"]
    for key in net_keys:
        net[key[:-6]] = net[key].view(np.int64)
        del net[key]

    multiple_keys = {}
    for key in net.keys():
        if key[-1].isdigit():
            root = key[:-2]
            if root not in multiple_keys.keys():
                multiple_keys[root] = 1
            else:
                multiple_keys[root] += 1

    for multi_root, n in multiple_keys.items():
        single_arrays = (net[f"{multi_root}_{i}"] for i in range(n))
        net[multi_root] = np.stack(tuple(single_arrays), axis=1)
        for single_key in (f"{multi_root}_{i}" for i in range(n)):
            del net[single_key]

    return net


@njit("i8(i8)")
def SimpleGenerator(x):
    print("gen")
    print("x ", x, type(x))
    i = 1
    print("i ", i, type(i))
    while i <= x:
        yield i
        i += 1
    while True:
        yield types.int32(-1)


@njit
def _is_throat(pore_im, x, y, z):
    w, h, d = pore_im.shape
    for dx, dy, dz in (
                    (-1, 0, 0),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ):
        x2 = x+dx
        y2 = y+dy
        z2 = z+dz
        if (x2 < 0 or x2 >= w or y2 < 0 or y2 >= h or z2 < 0 or z2 >= d):
            pass
        else:
            if pore_im[x2, y2, z2] == 1:
                return True
    return False


@njit
def lateral_columns_generator(axis):
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if axis == 0:
                    px = dy
                    py = dz
                elif axis == 1:
                    px = dx
                    py = dz
                elif axis == 2:
                    px = dx
                    py = dy
                yield dx, dy, dz, px, py


@njit
def _get_max_coords(im):
    w, h, d = im.shape
    coords = np.zeros(4, dtype=np.float64)
    coords[3] = np.float64(im[0, 0, 0])
    curr_val = 0.
    max_count = 1
    for x in range(w):
        for y in range(h):
            for z in range(d):
                new_val = im[x, y, z]
                if new_val > curr_val:
                    curr_val = new_val
                    coords[0] = np.float64(x)
                    coords[1] = np.float64(y)
                    coords[2] = np.float64(z)
                    coords[3] = np.float64(new_val)
                    max_count = 1
                elif new_val == curr_val:
                    max_count += 1
                    coords[0] = ((max_count-1)*coords[0])/max_count \
                        + np.float64(x)/max_count
                    coords[1] = ((max_count-1)*coords[1])/max_count \
                        + np.float64(y)/max_count
                    coords[2] = ((max_count-1)*coords[2])/max_count \
                        + np.float64(z)/max_count

    return coords


@njit
def _get_throats(
    pore_im,
    sub_im,
    sub_dt,
    voxel_size,
    throat_perimeter_mode="original",
):
    """
    pore_im : array_like
        A Boolean image of the current pore
    sub_im : array_like
        An image with integers indicating the pore labels
        (n+1 of actual pore table index)
    """
    w, h, d = pore_im.shape
    conns = set()
    inscribed_diameters = {-1: 0.}
    areas = {-1: 0.}
    perimeters = {-1: 0.}
    centers = {-1: (0., 0., 0.)}
    val_count = Dict()

    face_neighbours = (
                    (-1, 0, 0, 0),
                    (1, 0, 0, 0),
                    (0, -1, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, -1, 2),
                    (0, 0, 1, 2),
                )

    for x in range(1, w - 1):
        for y in range(1, h - 1):
            for z in range(1, d - 1):
                if pore_im[x, y, z] == 0:
                    continue
                for dx, dy, dz, ax in face_neighbours:
                    x2 = x + dx
                    y2 = y + dy
                    z2 = z + dz
                    if pore_im[x2, y2, z2] == 0:
                        val = sub_im[x2, y2, z2] - 1
                        if val == -1:
                            continue
                        conns.add(np.int64(val))

                        # Center and diameter calculations, for every throat voxel
                        dt = sub_dt[x2, y2, z2]
                        if val in list(inscribed_diameters.keys()):
                            last_diameter = inscribed_diameters[val]
                            if dt > last_diameter:
                                val_count[val] = 1
                                inscribed_diameters[val] = dt
                                centers[val] = (
                                    (x2) * voxel_size[0],
                                    (y2) * voxel_size[1],
                                    (z2) * voxel_size[2],
                                    )
                            elif dt == last_diameter:
                                val_count[val] += 1
                                n = val_count[val]
                                centers[val] = (
                                    ((n-1) * centers[val][0]) / n
                                    + ((x2) * voxel_size[0]) / n,
                                    ((n-1) * centers[val][1]) / n
                                    + ((y2) * voxel_size[1]) / n,
                                    ((n-1) * centers[val][2]) / n
                                    + ((z2) * voxel_size[2]) / n,
                                )
                        else:
                            inscribed_diameters[val] = dt
                            val_count[val] = 1
                            centers[val] = (
                                (x2) * voxel_size[0],
                                (y2) * voxel_size[1],
                                (z2) * voxel_size[2],
                                )

                        # get pseudo-projection
                        if throat_perimeter_mode == "original":
                            projection = np.ones((3, 3), dtype=np.uint8)
                        else:
                            projection = np.zeros((3, 3), dtype=np.uint8)
                            projection[1, 1] = 1

                        for dx2, dy2, dz2, px, py in lateral_columns_generator(ax):
                            x3 = x2 + dx2
                            y3 = y2 + dy2
                            z3 = z2 + dz2

                            if (x3 < 0 or x3 >= w) or \
                                    (y3 < 0 or y3 >= h) or \
                                    (z3 < 0 or z3 >= d):
                                continue

                            if throat_perimeter_mode == "original":
                                if sub_im[x3, y3, z3] == 0:
                                    projection[px, py] = 0
                            else:
                                if sub_im[x3, y3, z3] != (val + 1):
                                    pass
                                elif _is_throat(pore_im, x3, y3, z3):
                                    projection[px, py] = 1

                        if ax == 0:
                            projection_size = (voxel_size[1], voxel_size[2])
                        elif ax == 1:
                            projection_size = (voxel_size[0], voxel_size[2])
                        elif ax == 2:
                            projection_size = (voxel_size[0], voxel_size[1])
                        perimeter, area = jit_marching_squares_perimeter_and_area(
                            projection,
                            target_label=1,
                            spacing=projection_size,
                            overlap=True,
                            )

                        if val in list(perimeters.keys()):
                            perimeters[val] += perimeter
                        else:
                            perimeters[val] = perimeter

                        if val in list(areas.keys()):
                            areas[val] += area
                        else:
                            areas[val] = area

    return (
        list(conns),
        inscribed_diameters,
        areas,
        perimeters,
        centers,
        )


@njit(parallel=True, debug=False)
def _jit_regions_to_network_parallel(
    im,
    dt,
    flat_slices,
    template_areas,
    template_volumes,
    phases,
    voxel_size,
    porosity_map,
    threads,
):
    slices = []
    for i in range(0, len(flat_slices), 3):
        slices.append((
            flat_slices[i],
            flat_slices[i+1],
            flat_slices[i+2],
        ))
    mc_debug = False

    # Initialize arrays
    Ps = np.arange(1, np.amax(im)+1)
    Np = np.int64(Ps.size)
    p_coords_cm = np.zeros((Np, im.ndim), dtype=float)
    p_coords_dt = np.zeros((Np, im.ndim), dtype=float)
    p_coords_dt_global = np.zeros((Np, im.ndim), dtype=float)
    p_volume = np.zeros((Np, ), dtype=float)
    p_dia_local = np.zeros((Np, ), dtype=float)
    p_dia_global = np.zeros((Np, ), dtype=float)
    p_label = np.zeros((Np, ), dtype=np.int64)
    p_area_surf = np.zeros((Np, ), dtype=float)
    p_phase = np.zeros((Np, ), dtype=np.int64)
    p_porosity = np.ones((Np, ), dtype=float)
    # The number of throats is not known at the start, so lists are used
    # which can be dynamically resized more easily.
    t_conns_0 = []
    t_conns_1 = []
    t_dia_inscribed = []
    t_area = []
    t_perimeter = []
    t_coords_0 = []
    t_coords_1 = []
    t_coords_2 = []

    partial_t_conns_0 = []
    partial_t_conns_1 = []
    partial_t_dia_inscribed = []
    partial_t_area = []
    partial_t_perimeter = []
    partial_t_coords_0 = []
    partial_t_coords_1 = []
    partial_t_coords_2 = []

    for i in range(threads-1):
        partial_t_conns_0.append(List.empty_list(np.uint64))
        partial_t_conns_1.append(List.empty_list(np.uint64))
        partial_t_dia_inscribed.append(List.empty_list(np.float64))
        partial_t_area.append(List.empty_list(np.float64))
        partial_t_perimeter.append(List.empty_list(np.float64))
        partial_t_coords_0.append(List.empty_list(np.float64))
        partial_t_coords_1.append(List.empty_list(np.float64))
        partial_t_coords_2.append(List.empty_list(np.float64))

    worker_status = np.zeros((threads-1,), dtype=np.uint32)
    worker_target = np.zeros((threads-1,), dtype=np.uint32)

    for self_id in prange(threads):
        if (self_id == (threads - 1)):
            current_pore = 1
            while True:
                wait()
                for worker_id in range(len(worker_status)):
                    if worker_status[worker_id] == IDLE:
                        if current_pore <= Np:
                            worker_target[worker_id] = current_pore
                            current_pore += 1
                            worker_status[worker_id] = ASSIGNED
                        else:
                            worker_status[worker_id] = FINISHED
                    if worker_status[worker_id] == DONE:
                        for throat_i in range(len(partial_t_conns_0[worker_id])):
                            t_conns_0.append(
                                partial_t_conns_0[worker_id][throat_i])
                            t_conns_1.append(
                                partial_t_conns_1[worker_id][throat_i])
                            t_dia_inscribed.append(
                                partial_t_dia_inscribed[worker_id][throat_i])
                            t_perimeter.append(
                                partial_t_perimeter[worker_id][throat_i])
                            t_area.append(
                                partial_t_area[worker_id][throat_i])
                            t_coords_0.append(
                                partial_t_coords_0[worker_id][throat_i])
                            t_coords_1.append(
                                partial_t_coords_1[worker_id][throat_i])
                            t_coords_2.append(
                                partial_t_coords_2[worker_id][throat_i])
                        if current_pore <= Np:
                            worker_target[worker_id] = current_pore
                            current_pore += 1
                            worker_status[worker_id] = ASSIGNED
                        else:
                            worker_status[worker_id] = FINISHED
                if (np.equal(worker_status, FINISHED).all()):
                    break
        else:
            while True:
                wait()
                status = worker_status[self_id]
                if status == ASSIGNED:
                    partial_t_conns_0[self_id] = List.empty_list(np.uint64)
                    partial_t_conns_1[self_id] = List.empty_list(np.uint64)
                    partial_t_dia_inscribed[self_id] = List.empty_list(np.float64)
                    partial_t_area[self_id] = List.empty_list(np.float64)
                    partial_t_perimeter[self_id] = List.empty_list(np.float64)
                    partial_t_coords_0[self_id] = List.empty_list(np.float64)
                    partial_t_coords_1[self_id] = List.empty_list(np.float64)
                    partial_t_coords_2[self_id] = List.empty_list(np.float64)

                    pore_i = worker_target[self_id]

                    pore = pore_i - 1
                    s = jit_extend_slice(slices[pore], im.shape)
                    sub_im = im[s]
                    sub_dt = dt[s]
                    pore_im = sub_im == pore_i
                    padded_mask = pad(pore_im)
                    pore_dt = \
                        jit_edt_cpu(padded_mask, scale=voxel_size, sqrt_result=True)
                    s_offset = np.array([a.start for a in s], dtype=np.float64)
                    p_label[pore] = pore_i
                    p_coords_cm[pore, :] = \
                        (center_of_mass(pore_im) + s_offset) * np.array(voxel_size)
                    max_dt_coords_local = _get_max_coords(pore_dt)
                    max_pore_dt_local = max_dt_coords_local[-1]
                    max_dt_coords_local = max_dt_coords_local[:-1]
                    p_coords_dt[pore, :] = \
                        (max_dt_coords_local + s_offset) * np.array(voxel_size)
                    p_phase[pore] = (phases[s]*pore_im).max()
                    if porosity_map is not None:
                        p_porosity[pore] = \
                            ((porosity_map[s]*pore_im).sum() / pore_im.sum()) / 100
                    else:
                        p_porosity[pore] = 1.

                    p_area_surf[pore], p_volume[pore] = \
                        jit_marching_cubes_area_and_volume(
                            sub_im,
                            target_label=pore_i,
                            template_areas=template_areas,
                            template_volumes=template_volumes,
                            debug=mc_debug,  # debugging line, TODO: remove
                        )
                    max_dt_coords = _get_max_coords(sub_dt)
                    max_pore_dt = max_dt_coords[-1]
                    max_dt_coords = max_dt_coords[:-1]
                    p_coords_dt_global[pore, :] = \
                        (max_dt_coords + s_offset) * np.array(voxel_size)
                    p_dia_local[pore] = 2*max_pore_dt_local
                    p_dia_global[pore] = 2*max_pore_dt
                    Pn, inscribed_diameter, areas, perimeters, centers = \
                        _get_throats(pore_im, sub_im, sub_dt, voxel_size)
                    for j in Pn:
                        if j > pore:
                            partial_t_conns_0[self_id].append(pore)
                            partial_t_conns_1[self_id].append(j)
                            partial_t_dia_inscribed[self_id].append(
                                inscribed_diameter[j])
                            partial_t_perimeter[self_id].append(perimeters[j])
                            partial_t_area[self_id].append(areas[j])
                            partial_t_coords_0[self_id].append(centers[j][0] +
                                                               s_offset[0])
                            partial_t_coords_1[self_id].append(centers[j][1] +
                                                               s_offset[1])
                            partial_t_coords_2[self_id].append(centers[j][2] +
                                                               s_offset[2])

                    worker_status[self_id] = DONE

                elif status == FINISHED:
                    break

    # Clean up values
    # Nt = len(t_conns_0)  # Get number of throats

    if len(t_conns_0) == 0:
        return None

    net_float = Dict.empty(
        key_type=types.unicode_type,
        value_type=FLOAT_TYPE,
    )

    net_int = Dict.empty(
        key_type=types.unicode_type,
        value_type=INT_TYPE,
    )

    t_coords_0_arr = np.array(t_coords_0, dtype=np.float64)
    t_coords_1_arr = np.array(t_coords_1, dtype=np.float64)
    t_coords_2_arr = np.array(t_coords_2, dtype=np.float64)

    ND = im.ndim
    # Define all the fundamental stuff
    net_int['throat.conns_0'] = np.array(t_conns_0)
    net_int['throat.conns_1'] = np.array(t_conns_1)
    net_float['pore.coords_0'] = p_coords_cm[:, 0]
    net_float['pore.coords_1'] = p_coords_cm[:, 1]
    net_float['pore.coords_2'] = p_coords_cm[:, 2]
    net_int['pore.all'] = np.ones_like(net_float['pore.coords_0'][:], dtype=np.int64)
    net_int['throat.all'] = \
        np.ones_like(net_int['throat.conns_0'][:], dtype=np.int64)
    net_int['pore.region_label'] = p_label
    net_int['pore.phase'] = p_phase
    net_float['pore.subresolution_porosity'] = p_porosity
    net_int['throat.phases_0'] = net_int['pore.phase'][net_int['throat.conns_0']]
    net_int['throat.phases_1'] = net_int['pore.phase'][net_int['throat.conns_1']]
    V = np.copy(p_volume)
    net_float['pore.region_volume'] = V
    f = 3/4
    net_float['pore.equivalent_diameter'] = 2*(V/np.pi * f)**(1/ND)
    # Extract the geometric stuff
    net_float['pore.local_peak_0'] = np.copy(p_coords_dt[:, 0])
    net_float['pore.local_peak_1'] = np.copy(p_coords_dt[:, 1])
    net_float['pore.local_peak_2'] = np.copy(p_coords_dt[:, 2])
    net_float['pore.global_peak_0'] = np.copy(p_coords_dt_global[:, 0])
    net_float['pore.global_peak_1'] = np.copy(p_coords_dt_global[:, 1])
    net_float['pore.global_peak_2'] = np.copy(p_coords_dt_global[:, 2])
    net_float['pore.geometric_centroid_0'] = np.copy(p_coords_cm[:, 0])
    net_float['pore.geometric_centroid_1'] = np.copy(p_coords_cm[:, 1])
    net_float['pore.geometric_centroid_2'] = np.copy(p_coords_cm[:, 2])
    net_float['throat.global_peak_0'] = t_coords_0_arr
    net_float['throat.global_peak_1'] = t_coords_1_arr
    net_float['throat.global_peak_2'] = t_coords_2_arr
    net_float['pore.inscribed_diameter'] = np.copy(p_dia_local)
    net_float['pore.extended_diameter'] = np.copy(p_dia_global)
    net_float['throat.inscribed_diameter'] = \
        np.array(t_dia_inscribed, dtype=np.float64) * 2.0
    P1 = net_int['throat.conns_0']
    P2 = net_int['throat.conns_1']
    PT1 = np.sqrt((net_float['pore.coords_0'][P1]-t_coords_0_arr)**2
                  + (net_float['pore.coords_1'][P1]-t_coords_1_arr)**2
                  + (net_float['pore.coords_2'][P1]-t_coords_2_arr)**2
                  )
    PT2 = np.sqrt((net_float['pore.coords_0'][P2]-t_coords_0_arr)**2
                  + (net_float['pore.coords_1'][P2]-t_coords_1_arr)**2
                  + (net_float['pore.coords_2'][P2]-t_coords_2_arr)**2
                  )
    net_float['throat.total_length'] = PT1 + PT2
    dist = \
        np.sqrt((net_float['pore.coords_0'][P1]-net_float['pore.coords_0'][P2])**2
                + (net_float['pore.coords_1'][P1]-net_float['pore.coords_1'][P2])**2
                + (net_float['pore.coords_2'][P1]-net_float['pore.coords_2'][P2])**2
                )
    net_float['throat.direct_length'] = dist
    net_float['throat.perimeter'] = np.array(t_perimeter)
    net_float['pore.volume'] = p_volume
    net_float['pore.surface_area'] = p_area_surf
    A = np.array(t_area)
    net_float['throat.cross_sectional_area'] = A
    net_float['throat.equivalent_diameter'] = (4*A/np.pi)**(1/2)

    for key, val in net_int.items():
        net_float[f"{key}_int64"] = val.view(np.float64)

    return net_float


@njit
def _jit_regions_to_network(
    im,
    dt,
    flat_slices,
    template_areas,
    template_volumes,
    phases,
    voxel_size,
    porosity_map,
    threads,
):
    return _jit_regions_to_network_parallel(
        im=im,
        dt=dt,
        flat_slices=flat_slices,
        template_areas=template_areas,
        template_volumes=template_volumes,
        phases=phases,
        voxel_size=voxel_size,
        porosity_map=porosity_map,
        threads=threads,
    )
