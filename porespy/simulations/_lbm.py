
import os
import subprocess

import h5py
import imageio
import mplbm_utils as mplbm
import nest_asyncio
import numpy as np

import porespy as ps


def return_im_poro(file, file_type=None, shape=None):
    if type(file) == str:
        if file_type is None:
            file_type = file[-3:]
        if file_type == "hdf5":
            f_in = h5py.File(file, "r")
            im_key = f_in.keys()[
                0
            ]  # key for the image by default assumming it has one key e.g.: 'im'
            im_in = np.array(f_in[im_key][()])
            f_in.close()

        elif file_type == "raw":
            raw_im = np.fromfile(file, dtype=np.uint8)
            im = raw_im.reshape(shape[0], shape[1], shape[2])
            im = im == 0

        elif file_type == "bin":
            bin_im = np.fromfile(file, dtype=np.uint8)
            im = bin_im.reshape(shape[0], shape[1], shape[2])
            im = im == 0

        elif file_type == "tif":
            im = imageio.mimread(file)
            im = np.array(im, dtype=bool)
    else:  # ndarray image itself
        im = np.copy(file)
    porosity = ps.metrics.porosity(im)
    return im, porosity


def prep_im_LBM(im, direction):
    im_LBM = ~np.array(im, dtype="bool")
    if direction == "x":
        pass
    if direction == "y":
        # im_LBM = np.rot90(im_LBM)
        im_LBM = np.swapaxes(im_LBM, 0, 1)
    if direction == "z":
        # im_LBM = np.rot90(im_LBM, k=1, axes=(0, 2))
        im_LBM = np.swapaxes(im_LBM, 0, 2)
    return im_LBM


def write_im_to_file(im_LBM, direction):
    path = "tmp_" + direction
    if not os.path.isdir(path):
        os.makedirs(path)
    path = "input_" + direction
    if not os.path.isdir(path):
        os.makedirs(path)
    hf = h5py.File(path + "/im_LBM" + direction + ".hdf5", "w")
    hf.create_dataset(
        "im", im_LBM.shape, data=im_LBM, dtype="int", compression="gzip", compression_opts=9
    )
    hf.close()
    return


# create dict of info
# parse_input_file reads inputs from a yaml file or from a Python dictionary

def create_dict(
    im_LBM,
    direction,
    simulation_folder_path,
    descriptor="D3Q19",
    num_in_out_layers=2,
    simulation_settings=None,
    periodic_bounds=None,
):
    Nx, Ny, Nz = im_LBM.shape
    input_output_dict = {
        # Full path to simulation directory (run pwd command in simulation directory and paste output here)
        "simulation directory": simulation_folder_path,
        "input folder": "input_" + direction + "/",
        "output folder": "tmp_" + direction + "/",
    }

    geometry_size_dict = {"Nx": Nx, "Ny": Ny, "Nz": Nz}
    geometry_dict = {
        "file name": "im_LBM" + direction + ".hdf5",  # Name of the input geometry file
        "data type": "int",
        "geometry size": geometry_size_dict,
    }
    domain_size_dict = {"nx": Nx, "ny": Ny, "nz": Nz}

    periodic_boundary_dict = {"x": False, "y": False, "z": False}
    if periodic_bounds is not None:
        periodic_boundary_dict.update(periodic_bounds)
    domain_dict = {
        "geom name": "im_LBM"
        + direction,  # Name of .dat file, rename from original if you'd like. Do not include the file extension.
        "domain size": domain_size_dict,
        "periodic boundary": periodic_boundary_dict,
        "inlet and outlet layers": num_in_out_layers,
        "add mesh": False,  # Add neutral mesh, by default False --> Not yet implemented
        "swap xz": False,  # False by default
        "double geom resolution": False,  # False by default
    }

    simulation_dict = {
        "num procs": 2,  # Number of processors to run on
        "num geoms": 1,  # Total number of geometries / individual simulations (this will be used once two-phase rel perm python code is done)
        "pressure": 0.00005,
        "max iterations": 20000000,
        "convergence": 1e-6,
        "save vtks": True,
    }
    if simulation_settings is not None:
        simulation_dict.update(simulation_settings)
    input_dict = {
        "simulation type": "1-phase",
        "input output": input_output_dict,
        "geometry": geometry_dict,
        "domain": domain_dict,
        "simulation": simulation_dict,
        "descriptor": descriptor,
    }
    return input_dict


def run_1_phase_sim(inputs):
    if inputs["descriptor"] == "D3Q19":
        descriptor_path = "/home/amin/Code/MPLBM-UT/src/1-phase_LBM/permeability"
    elif inputs["descriptor"] == "MRT":
        descriptor_path = "/home/amin/Code/MPLBM-UT/src/1-phase_LBM/permeability_MRT"

    sim_directory = inputs["input output"]["simulation directory"]

    # 2) Create Palabos geometry
    # print("Creating efficient geometry for Palabos...")
    mplbm.create_geom_for_palabos(inputs)

    # 3) Create simulation input file
    # print("Creating input file...")
    mplbm.create_palabos_input_file(inputs)

    # 4) Run 1-phase simulation
    # print("Running 1-phase simulation...")
    num_procs = inputs["simulation"]["num procs"]
    input_dir = inputs["input output"]["input folder"]
    output_dir = inputs["input output"]["output folder"]
    simulation_command = (
        # f"mpirun -np {num_procs} {descriptor_path} {input_dir}1_phase_sim_input.xml > /dev/null 2>&1"
        f"mpirun -np {num_procs} {descriptor_path} {input_dir}1_phase_sim_input.xml > log.txt"
    )
    file = open(f"{sim_directory}/{input_dir}run_single_phase_sim.sh", "w")
    file.write(f"{simulation_command}")
    file.close()
    simulation_command_subproc = f"bash {sim_directory}/{input_dir}run_single_phase_sim.sh"
    subprocess.run(simulation_command_subproc.split(" "))
    # Reading a file into a dictionary
    file_path = f"{sim_directory}/{output_dir}/relPerm&vels.txt"
    file = open(file_path)
    resources = {}
    content = file.readlines()
    line = content[5]
    key, value = line.rstrip().split("=", 1)
    file.close()
    K_LBM = float(value)
    return K_LBM


def permeability_lbm(
    im,
    direction="x",
    voxel_size=1,
    descriptor="D3Q19",
    num_in_out_layers=2,
    simulation_settings=None,
    periodic_bounds=None,
):
    try:
        im, porosity = return_im_poro(im)
        im_LBM = prep_im_LBM(im, direction)
        write_im_to_file(im_LBM, direction)
        simulation_folder_path = os.getcwd()
        input_dict = create_dict(
            im_LBM,
            direction,
            simulation_folder_path=simulation_folder_path,
            descriptor=descriptor,
            num_in_out_layers=num_in_out_layers,
            simulation_settings=simulation_settings,
            periodic_bounds=periodic_bounds,
        )
        inputs = mplbm.parse_input_file(input_dict)     # Parse inputs
        K_LBM = run_1_phase_sim(inputs)                 # Run 1 phase sim
        perm = K_LBM * voxel_size**2
    except Exception as e:
        print(e)
        perm = np.nan
    
    return perm


if __name__ == "__main__":
    np.random.seed(6)
    im = ps.generators.blobs(shape=[40, 40, 40], porosity=0.7, blobiness=2)
    K = ps.simulations.permeability_lbm(im=im)
    print(f"Permeability: {K:.2e} m^2")
