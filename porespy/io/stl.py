import numpy as np
from stl import mesh
import skimage.measure as ms


class stl():
    """
    """
    def save(im, vs, path):
        r"""
        Converts an array to an stl file.

        Parameters
        ----------
        im : 3D image
            The image of the porous material
        voxel_size : int
            The side length of the voxels (voxels  are cubic)
        path : string
            Path to output file
        """
        im = np.pad(im, pad_width=10, mode='constant', constant_values=True)
        vertices, faces, norms, values = ms.marching_cubes_lewiner(im)
        vertices *= vs
        # export the stl file
        export = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                export.vectors[i][j] = vertices[f[j], :]
        export.save(path+'.stl')
