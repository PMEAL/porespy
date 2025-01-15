import itertools
import math
import os
import numpy as np
from pathlib import Path
from numba import njit
from scipy import spatial
from skimage import measure


__all__ = [
    'face_orientation',
    'area_of_triangle',
    'mc_templates_generator',
    'create_mc_template_list',
    'calculate_area_and_volume',
    'marching_cubes_area_and_volume',
    'jit_marching_cubes_area_and_volume',
]


MC_TEMPLATES_FILENAME = 'marching_cubes_templates.dat'
THIS_FOLDER = Path(__file__).absolute().parent
MC_TEMPLATES_PATH = os.path.join(THIS_FOLDER, MC_TEMPLATES_FILENAME)


def face_orientation(v0, v1, v2):
    '''
    Return outward perpendicular vector distance of face along the z axis
    '''
    v0 = np.array(v0),
    v1 = np.array(v1)
    v2 = np.array(v2)
    vector = np.cross(v1 - v0, v2 - v0)
    z_comp = vector[0][2]
    if z_comp > 0.1:
        return -1
    elif z_comp < -0.1:
        return 1
    else:
        return 0


def area_of_triangle(p0, p1, p2):
    '''
    As per Herons formula
    '''
    lines = list(itertools.combinations((p0, p1, p2), 2))
    distances = [(spatial.distance.euclidean(i[0], i[1])) for i in lines]
    s = sum(distances)/2
    product_of_diferences = np.prod([(s-i) for i in distances])
    area = math.sqrt(s*product_of_diferences)
    return area


def mc_templates_generator(override=False):
    '''
    Generates a marching cubes template list file, if one is not available
    '''
    if os.path.isfile(MC_TEMPLATES_PATH) and not override:
        return
    summation_to_coordinate = {}
    for i in [(x, y, z) for x in range(2) for y in range(2) for z in range(2)]:
        summation_to_coordinate[2 ** (i[0] + 2*i[1] + 4*i[2])] = i

    templates_triangles = []
    for _ in range(256):
        templates_triangles.append([[], []])

    for i in range(1, 255):
        array = np.zeros((2, 2, 2))
        index = i
        for j in range(7, -1, -1):
            e = 2**j
            if index >= e:
                index -= e
                array[summation_to_coordinate[e]] = 1
        verts, faces = measure.marching_cubes(array)[0:2]
        templates_triangles[i][0] = verts
        templates_triangles[i][1] = faces

    with open(MC_TEMPLATES_PATH, mode='w') as file:
        for i in range(256):
            verts, faces = templates_triangles[i]
            file.write(f'{i};')
            for v in verts:
                file.write(f'[{v[0]},{v[1]},{v[2]}]')
            file.write(';')
            for f in faces:
                file.write(f'[{f[0]},{f[1]},{f[2]}]')
            file.write('\n')


def create_mc_template_list(spacing=(1, 1, 1)):
    '''
    Return area and volume lists for the marching cubes templates
    Reads the templates file
    Input:
        Tuple with three values for x, y, and z lengths of the voxel edges
    '''
    mc_templates_generator()

    areas = {}
    volumes = {}
    triangles = {}
    vertices_on_top = set((16, 32, 64, 128))
    with open(MC_TEMPLATES_PATH, mode='r') as file:
        for line in file:
            index, verts, faces = line.split(';')
            index = int(index)
            if len(verts) > 0:
                verts = verts.strip()[1:-1].split('][')
                verts = [v.split(',') for v in verts]
                verts = [[float(edge) for edge in v] for v in verts]
                faces = faces.strip()[1:-1].split('][')
                faces = [f.split(',') for f in faces]
                faces = [[int(edge) for edge in f] for f in faces]
            else:
                verts = []
                faces = []

            occupied_vertices = set()
            sub_index = index
            for i in range(7, -1, -1):
                e = 2 ** i
                if sub_index >= e:
                    occupied_vertices.add(e)
                    sub_index -= e
            total_vertices_on_top = len(occupied_vertices & vertices_on_top)
            if total_vertices_on_top == 0:
                basic_volume = 0
            elif total_vertices_on_top == 1:
                basic_volume = 1/8
            elif total_vertices_on_top == 2:
                if ((16 in occupied_vertices and 128 in occupied_vertices) or
                        (32 in occupied_vertices and 64 in occupied_vertices)):
                    basic_volume = 1/4
                else:
                    basic_volume = 1/2
            elif total_vertices_on_top == 3:
                basic_volume = 7/8
            elif total_vertices_on_top == 4:
                basic_volume = 1

            for f in faces:
                v0, v1, v2 = [verts[i] for i in f]
                v0_proj, v1_proj, v2_proj = [(i[0], i[1], 0) for i in (v0, v1, v2)]
                mean_z = sum([i[2] for i in (v0, v1, v2)])/3
                proj_area = area_of_triangle(v0_proj, v1_proj, v2_proj)
                direction = face_orientation(v0, v1, v2)
                basic_volume += mean_z * proj_area * direction

            for i in range(len(verts)):
                verts[i] = [j[0] * j[1] for j in zip(verts[i], spacing)]

            triangles[index] = (tuple(verts), tuple(faces), basic_volume)

    voxel_volume = np.prod(np.array(spacing))
    for i in triangles:
        area = 0
        verts, faces, relative_volume = triangles[i]
        for f in faces:
            triangle_area = area_of_triangle(verts[f[0]],
                                             verts[f[1]],
                                             verts[f[2]])
            area += triangle_area
        volume = voxel_volume * relative_volume
        areas[i] = area
        volumes[i] = volume

    areas = np.array(list(areas.values()), dtype=np.float32)
    volumes = np.array(list(volumes.values()), dtype=np.float32)

    return areas, volumes


@njit
def calculate_area_and_volume(
    img,
    vertex_index_array,
    target_label=1,
    spacing=(1, 1, 1),
    template_areas=None,
    template_volumes=None,
    debug=False,
):
    w, h, d = img.shape
    area = 0.0
    volume = 0.0

    if debug:
        print("Shape: ", w, h, d)

    for x in range(w - 1):
        for y in range(h - 1):
            for z in range(d - 1):
                if debug:
                    print(x, y, z)
                sub_array = img[x:x + 2, y:y + 2, z:z + 2]
                if target_label not in sub_array:
                    continue

                template_number = (
                    (sub_array == target_label) * vertex_index_array
                ).sum()

                area += template_areas[template_number]
                volume += template_volumes[template_number]

    return area, volume


def marching_cubes_area_and_volume(
    img,
    target_label=1,
    spacing=(1, 1, 1),
    template_areas=None,
    template_volumes=None,
):

    if (template_areas is None) or (template_volumes is None):
        template_areas, template_volumes = create_mc_template_list(spacing)

    vertex_index_array = np.array([2**i for i in range(8)])
    vertex_index_array = vertex_index_array.reshape((2, 2, 2), order="F")

    area, volume = calculate_area_and_volume(
        img,
        vertex_index_array,
        target_label=target_label,
        spacing=spacing,
        template_areas=template_areas,
        template_volumes=template_volumes,
    )

    return area, volume


@njit
def jit_marching_cubes_area_and_volume(
    img,
    target_label=1,
    spacing=(1, 1, 1),
    template_areas=None,
    template_volumes=None,
    debug=False,
):

    vertex_index_array = np.array(
        [[[1, 16],
          [4, 64]],
         [[2, 32],
          [8, 128]]],
        dtype=np.int32
    )

    area, volume = calculate_area_and_volume(
        img,
        vertex_index_array,
        target_label=target_label,
        spacing=spacing,
        template_areas=template_areas,
        template_volumes=template_volumes,
        debug=debug,
    )

    return area/2, volume
