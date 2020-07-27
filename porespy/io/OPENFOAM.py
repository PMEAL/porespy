import numpy as np
from scipy.ndimage import zoom
import sys

np.set_printoptions(threshold=sys.maxsize)


class OPENFOAM():
    """
    """

    def save(im, scale=1, zoom_factor=1, label=True):
        """
        Given a boolean numpy array where True is void and False is solid,
        creates a blockMesh file for use in OpenFOAM
        index: The boolean array
        scale: Scaling factor for vertex coords (ex. 0.001 scales to mm)
        """

        file = """
/*--------------------------------*- C++ -*----------------------------------*\
| \n =========                 |                                              |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v1912                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
|*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale ;

vertices
(

);

blocks
(

);

edges
(
);

boundary
(
    top
    {
        type patch;
        faces
        (

        );
    }
    back
    {
        type patch;
        faces
        (

        );
    }
    bottom
    {
        type patch;
        faces
        (

        );
    }
    front
    {
        type patch;
        faces
        (

        );
    }
    left
    {
        type patch;
        faces
        (

        );
    }
    right
    {
        type patch;
        faces
        (

        );
    }
    walls
    {
        type wall;
        faces
        (

        );
    }
);

mergePatchPairs
(
);

// ************************************************************************* //
"""
        # Applies image compression
        im = zoom(im.astype(int), zoom_factor)

        im = im.astype(bool)

        # Finds solids and unpacks coords
        x, y, z = np.where(im)

        # Setup for appending
        x = x.reshape(-1, 1) + 0.5
        y = y.reshape(-1, 1) + 0.5
        z = z.reshape(-1, 1) + 0.5

        # Combines coords to create centers matrix
        centers = np.append(x, y, axis=1)
        centers = np.append(centers, z, axis=1).reshape(-1, 3)

        # Finds boundary coords, used for labelling boundaries
        if label:
            minx = np.array([x == min(x)]).flatten()
            miny = np.array([y == min(y)]).flatten()
            minz = np.array([z == min(z)]).flatten()

            maxx = np.array([x == max(x)]).flatten()
            maxy = np.array([y == max(y)]).flatten()
            maxz = np.array([z == max(z)]).flatten()

            left_points = centers[minx] - 0.5
            bottom_points = centers[miny] - 0.5
            front_points = centers[minz] - 0.5

            right_points = centers[maxx] + 0.5
            top_points = centers[maxy] + 0.5
            back_points = centers[maxz] + 0.5

        # Matrix to convert the centers to vertices
        d = np.array([-0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5,
                      0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5,
                      -0.5, -0.5, 0.5, -0.5]).reshape(-1, 3)

        # Adjusting the scale
    #    d = d * scale / compression

        # Matrix to face coords that have an outward facing normal
        d_left = np.array([0, 0, 0, 0, 0, 1,
                           0, 1, 1, 0, 1, 0]).reshape(-1, 3)
        d_bottom = np.array([0, 0, 0, 1, 0, 0,
                             1, 0, 1, 0, 0, 1]).reshape(-1, 3)
        d_front = np.array([0, 0, 0, 0, 1, 0,
                            1, 1, 0, 1, 0, 0]).reshape(-1, 3)

        d_right = np.array([0, 0, 0, 0, -1, 0,
                            0, -1, -1, 0, 0, -1]).reshape(-1, 3)
        d_top = np.array([0, 0, 0, 0, 0, -1,
                          -1, 0, -1, -1, 0, 0]).reshape(-1, 3)
        d_back = np.array([0, 0, 0, -1, 0, 0,
                           -1, -1, 0, 0, -1, 0]).reshape(-1, 3)

        # Adjusting the scale
    #     d_left = d_left * scale / compression
    #     d_bottom = d_bottom * scale / compression
    #     d_front = d_front * scale / compression
    #
    #     d_right = d_right * scale / compression
    #     d_top = d_top * scale / compression
    #     d_back = d_back * scale / compression

        # Initialize vertices matrix
        vertices = np.zeros(len(centers) * len(d) * 3).reshape(-1, 8, 3)

        # Initialize faces matrices
        top = np.zeros((len(centers), 4), dtype=int)
        back = np.zeros((len(centers), 4), dtype=int)
        bottom = np.zeros((len(centers), 4), dtype=int)
        front = np.zeros((len(centers), 4), dtype=int)
        left = np.zeros((len(centers), 4), dtype=int)
        right = np.zeros((len(centers), 4), dtype=int)

        # Initialize blocks matrix
        blocks = np.zeros((len(centers), 8), dtype=int)

        # Combining centers and distance to create vertices, uses broadcasting
        for i in range(len(centers)):
            vertices[i] = centers[i] + d

        vertices = vertices.reshape(-1, 3)

        # Creating unique vertices matrix, as well as an index to map between
        # vertices and vertices_unique such that
        # vertices_unique[index] == vertices
        vertices_unique, index = np.unique(vertices, axis=0,
                                           return_inverse=True)

        if label:
            top_faces = boundary(top_points, d_top, vertices_unique)
            back_faces = boundary(back_points, d_back, vertices_unique)
            bottom_faces = boundary(bottom_points, d_bottom, vertices_unique)
            front_faces = boundary(front_points, d_front, vertices_unique)
            left_faces = boundary(left_points, d_left, vertices_unique)
            right_faces = boundary(right_points, d_right, vertices_unique)

            # Expanding faces (patches)
            for i in range(len(centers)):
                top[i] = np.array([2, 6, 7, 3]) + 8 * i
                back[i] = np.array([2, 3, 0, 1]) + 8 * i
                bottom[i] = np.array([4, 5, 1, 0]) + 8 * i
                front[i] = np.array([7, 6, 5, 4]) + 8 * i
                left[i] = np.array([0, 3, 7, 4]) + 8 * i
                right[i] = np.array([2, 1, 5, 6]) + 8 * i

            # Replaces duplicates vertices with their corresponding index
            # in the unique vertices list
            top = index[top]
            back = index[back]
            bottom = index[bottom]
            front = index[front]
            left = index[left]
            right = index[right]

            # Deleting all internal faces

            # Only opposing faces will overlap, makes one matrix where the face
            # definitions are sorted
            top_and_bottom = np.sort(np.vstack((top, bottom)))
            front_and_back = np.sort(np.vstack((front, back)))
            left_and_right = np.sort(np.vstack((left, right)))

            # Creates a matrix of unique face definitions as well as
            # their count
            t_and_b_unique, t_and_b_count = np.unique(top_and_bottom, axis=0,
                                                      return_counts=True)
            f_and_b_unique, f_and_b_count = np.unique(front_and_back, axis=0,
                                                      return_counts=True)
            l_and_r_unique, l_and_r_count = np.unique(left_and_right, axis=0,
                                                      return_counts=True)

            # Deletes the faces mentioned more than once
            top_and_bottom = t_and_b_unique[t_and_b_count == 1]
            front_and_back = f_and_b_unique[f_and_b_count == 1]
            left_and_right = l_and_r_unique[l_and_r_count == 1]

            # Putting ALL faces into one wall matrix
            walls = np.vstack((top_and_bottom, front_and_back, left_and_right,
                               left_faces, bottom_faces, front_faces,
                               right_faces, top_faces, back_faces))
            walls_sorted = np.sort(walls)

            # This sorts out the faces on a boundary and keeps the rest
            # Will be labelled walls
            walls_u, walls_index, walls_count = np.unique(walls_sorted, axis=0,
                                                          return_counts=True,
                                                          return_index=True)
            walls = walls[walls_index][walls_count == 1]

            string_top = stringify(top_faces)
            string_back = stringify(back_faces)
            string_bottom = stringify(bottom_faces)
            string_front = stringify(front_faces)
            string_left = stringify(left_faces)
            string_right = stringify(right_faces)
            string_walls = stringify(walls)

        # Expanding blocks matrix
        for i in range(len(centers)):
            blocks[i] = np.array([4, 5, 6, 7, 0, 1, 2, 3]) + 8 * i

        # Replaces duplicates vertices with their corresponding index in the
        # unique vertices list
        blocks = index[blocks]

        # OpenFOAM format
        vertices_unique = vertices_unique.reshape(-1, 3)

        # Creating the vertices string
        string_vert = str(list(vertices_unique)).replace("array([", "(")
        string_vert = string_vert.replace("])", ")\n\t\t")
        string_vert = "\t(" + string_vert.strip("[()]")

        # Creating all face strings - replaces [] with () for C++ format

        # Creating block string
        str_blocks = "[" + str(blocks).strip("[]") + "]"
        str_blocks = str_blocks.replace("[", "hex (")
        str_blocks = str_blocks.replace("]",
                                        ") (1 1 1) simpleGrading (1 1 1)\n\t")

        # Inserting vertices string
        str1 = "vertices\n(\n"
        index1 = file.find(str1) + len(str1)
        file = file[:index1] + "\t" + string_vert + file[index1:]

        # Inserting scale
        str2 = "scale "
        index2 = file.find(str2) + len(str2)
        file = file[:index2] + str(scale) + file[index2:]

        # Inserting faces
        if label:
            str3 = 'top\n    {\ntype patch;\nfaces\n(\n'
#            str3 = "top\n    {\n        type patch;\
#            faces\n        (\n"
            index3 = file.find(str3) + len(str3)
            file = file[:index3] + str(string_top) + file[index3:]

            str4 = 'back\n    {\ntype patch;\nfaces\n(\n'
            index4 = file.find(str4) + len(str4)
            file = file[:index4] + str(string_back) + file[index4:]

            str5 = 'bottom\n    {\ntype patch;\nfaces\n(\n'
            index5 = file.find(str5) + len(str5)
            file = file[:index5] + str(string_bottom) + file[index5:]

            str6 = 'front\n    {\ntype patch;\nfaces\n(\n'
            index6 = file.find(str6) + len(str6)
            file = file[:index6] + str(string_front) + file[index6:]

            str7 = 'left\n    {\ntype patch;\nfaces\n(\n'
            index7 = file.find(str7) + len(str7)
            file = file[:index7] + str(string_left) + file[index7:]

            str8 = 'right\n    {\ntype patch;\nfaces\n(\n'
            index8 = file.find(str8) + len(str8)
            file = file[:index8] + str(string_right) + file[index8:]

            str9 = 'walls\n    {\ntype patch;\nfaces\n(\n'
            if string_walls != "()":
                index8 = file.find(str9) + len(str9)
                file = file[:index8] + str(string_walls) + file[index8:]

        # Deletes all empty labels if labeling is disabled
        if not label:
            string_start = "boundary\n("
            index_start = file.find(string_start) + len(string_start)
            string_end = "mergePatchPairs"
            index_end = file.find(string_end)
            file = file[:index_start] + "\n);\n\n" + file[index_end:]

        # Inserting blocks
        str10 = "blocks\n(\n"
        index9 = file.find(str10) + len(str10)
        file = file[:index9] + str(str_blocks) + file[index9:]

        # Gets rid of all commas in the file
        file = file.replace(",", "")

        # Returns the doc as a string
        with open("blockMeshDict", "w+") as a:
            return a.write(file)


def boundary(points, d, vert_unique):
    # Initialization
    boundary_points = np.zeros(len(points) * len(d) * 3).reshape(-1, 4, 3)

    # Creates all points at mesh boundary
    for i in range(len(points)):
        boundary_points[i] = points[i] + d

    boundary_points = boundary_points.reshape(-1, 3)

    # Creates tuples of each row and stores them in a numpy array
    # This allows for the vectorization of np.where
    vert_tuple = vert_unique.view([('', vert_unique.dtype)] *
                                  vert_unique.shape[1])
    boundary_tuple = boundary_points.view([('', boundary_points.dtype)] *
                                          boundary_points.shape[1])
    # Replaces vertices with corresponding index in vert_unique
    num = np.array([np.arange(0, len(vert_tuple))]).reshape(-1, 1)
    bool_mat = boundary_tuple[:, None] == vert_tuple
    faces = np.array([])

    # Replacment for np.where
    for i in bool_mat:
        faces = np.append(faces, num[i])

    faces = faces.astype(int).reshape(-1, 4)

    return faces


def stringify(thelist):
    string = str(thelist).replace("[", "(")
    string = string.replace("]", ")")
    string = "(" + string.strip("[()]") + ")"

    return string
