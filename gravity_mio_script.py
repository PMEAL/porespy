# %%
import porespy as ps
import scipy.ndimage as spim
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D
import time
from porespy.filters import _erode_im, _dilate_im, dilate_im_gravity, eliminate_overlapping
from porespy.filters import Connectivity_Check


# %% Input Data Cockpit
start = time.time()

# Variables
gamma = 0.0728            # Interfacial tension [N/m]
theta = [ 0 ]             # Contact angle [°] -> Different Contact angles. Position in list is corresponding figure of solid phase
dr = 1.6 * 10 ** (-5)     # resolution [meter/pixel]
NwpR = "-x"               # Position of NWP-Reservoir [plane]
rhoWP = 1                 # Density of WP [kg/m^3]
rhoNWP = 1000             # Density of NWP [kg/m^3]
g = 9.81                  # gravitational acceleration [m/s^2]
gravity = True            # Regard gravity effects [Yes=True, No=False]
Sat_Curve = True          # Decides if Saturation Curve is created or specific intrusion pressure is selected
int_pressure = 407.5      # Pressure of intrusion if Sat_Curve==False [Pa]
saturation_steps = 40     # Number of Steps for creating the saturation curve
dirname = ''


# %% Generate an image
im_shape = [3000, 1000]
im = ps.generators.overlapping_spheres(shape=im_shape, radius=20, porosity=.6)


# %% Analyse Image
# Lists
R = []
Snwp = []
pc = []
images = []
images2 = []

direction = ["x", "-x", "z", "-z", "y", "-y"]
theta = np.array(theta)

Snwp.append(0)
pc.append(0)

if NwpR not in direction:
    raise Exception("NwpR has to be element of %s" % direction)
for item in theta:
    if item % 90 == 0 and item / 90 % 2 != 0:
        raise Exception("Theta is not allowed to be (2n-1)*90°, for n = 1,2,3,4,...")

# Convert Image in suitable format WP = 0, Solid Phase = 1, NWP > 1
im = im.copy()
im = (im - 1) * (-1)
im = im * 1
shape = im.shape

# Decide if Data is 2D or 3D
if len(shape) == 2:
    D2img = True
    Lx = shape[0]
    Ly = shape[1]
else:
    D2img = False
    Lx = shape[0]
    Ly = shape[1]
    Lz = shape[2]

# Porosity
P = ps.metrics.porosity(im)

# %% Calculate start conditions

# Get minimum distance to solid phase
# Approach: different im_distance arrays for each solid phase with
# specific contact angle costheta

img_different_solid_phase = im

solidelements = np.unique(img_different_solid_phase)
solidelements = np.delete(solidelements, np.where(solidelements == 0))
solidelements_amount = len(solidelements)

if D2img is True:
    im_distance = np.zeros([Lx, Ly, solidelements_amount])
    i = 0
    for item in solidelements:
        split_array = np.where(img_different_solid_phase == item, 0, 1)
        im_distance[:, :, i] = spim.distance_transform_edt(split_array)
        i += 1
else:
    im_distance = np.zeros([Lx, Ly, Lz, solidelements_amount])
    i = 0
    for item in solidelements:
        split_array = np.where(img_different_solid_phase == item, 0, 1)
        im_distance[:, :, :, i] = spim.distance_transform_edt(split_array)
        i += 1

# Get factor from contact angle
costheta_list = np.abs(np.cos(theta / 180 * np.pi))

# Calculate height of image
h_max = Lx * dr

# Count is variable which shows intrusion process with increasing pressure
# by increasing integer
# CountA must be smallest value which does not exist in the image

countA = np.max(solidelements) + 1
count = countA

if Sat_Curve is True:

    # Calculate maxmimum sphere diameter

    D_max = np.max(spim.distance_transform_edt(im)) * 2
    if gravity is True:
        # Minimum pressure to intrude into pore structure
        D_max2 = (4 * gamma) / (rhoWP * g * h_max * dr)
        if D_max2 < D_max:
            D_max = D_max2

    # Calculate minimum sphere diameter

    if gravity is True:
        # Maximum pressure to intrude into pore structure on heighest level
        # with resolution of 1 px
        det = np.min(costheta_list) - rhoNWP * g * h_max * D_max * dr / (4 * gamma)
        D_min2 = (4 * gamma) / (
            4 * gamma * np.min(costheta_list) + rhoNWP * g * h_max * dr
        )
        D_min = D_min2
    else:
        D_min = 1

    # Define resolution by calculating with minimum and maximum diameter in 100 steps

    resolution = (D_max - D_min) / saturation_steps

else:
    D_max = 4 * gamma / int_pressure / dr
    D_min = 1
    resolution = D_max


# Stepsize
# Approach: Bond Number = 0.01
# Gravity influences can be neglected and segment has then constant
# hydrostatic pressure
Bo = 0.01
step = Bo * 4 * gamma / (dr ** 2 * D_max * abs(rhoNWP - rhoWP) * g)
step = int(step)
if step < 1:
    step = int(1)
print(step)

# Define start diameter
D = D_max

# Define threshold value
upp = np.floor(D_max / resolution) + countA + 2

# Set im_saturated for illustrating NWP penetration
im_saturated = img_different_solid_phase.copy()
im_opened_static = np.zeros(shape)

# Shows progress of calculation
progress_var = 1

# %% Pressure Saturation Curve

while D > D_min:

    im_eroded = np.zeros(shape)

    # Calculate Erosion for each solidphase with contact angle in
    # costheta_list seperately
    for dis_split in range(solidelements_amount):

        # Calculate erosion diameter for each solid phase dis_split with
        # its wettability character
        costheta = costheta_list[dis_split]
        Deros0 = D * costheta

        # Calculate pressure
        p0 = 4 * gamma / (D * dr)

        if gravity is True:

            # Calculate the erosion diameter Reros(h) with its reliance to height h
            Reros = np.zeros(int(Lx / step) + 1)
            for i in range(0, Lx, step):
                lower = i
                higher = i + step

                # Calculate with the mean height of the scope between
                # lower and higher value
                h = (lower + step / 2) * dr
                temp = (costheta * 2 * gamma /
                        (p0 - (rhoWP * g * (h_max - h) + rhoNWP * g * h)) / dr)
                Reros[int(i / step)] = temp

                # Add eroded profile of solidphase with contact angle
                # costheta to main eroded profile im_eroded
                if D2img is True:
                    temp = _erode_im(im_distance[lower:higher, :, dis_split],
                                     Reros[int(i/step)], 0, upp, )
                    im_eroded[lower:higher, :] += temp
                else:
                    temp = _erode_im(im_distance[lower:higher, :, :, dis_split],
                                     Reros[int(i/step)], 0, upp, )
                    im_eroded[lower:higher, :, :] += temp
        else:
            if D2img is True:
                im_eroded += _erode_im(im_distance[:, :, dis_split],
                                       Deros0/2, 0, upp)
            else:
                temp = _erode_im(im_distance[:, :, :, dis_split],
                                 Deros0/2, 0, upp)
                im_eroded += temp

    threshold, upper, lower = 0, upp, 0
    im_eroded = np.where(im_eroded == threshold, upper, lower)

    # Check connectivity to intrusion side
    im_connected_bool = Connectivity_Check(im_eroded, NwpR, D2img)

    if type(im_connected_bool) == bool:
        # There is no entry and we can skip the dilation
        R.append(D/2)
        pc.append(p0)
        Snwp.append(0)

        # If there is no entry, the maximum diameter and the resolution
        # will be lowered
        if Sat_Curve is True:
            D_max = D_max - resolution
            D = D - resolution
            resolution = (D_max - D_min) / saturation_steps
            D_max = D_max + resolution
            D = D + resolution
        pass
    else:
        # Dilation of the eroded pore space which is connected to NWP
        # reservoir with diameter D
        im_distance2 = spim.distance_transform_edt(im_connected_bool)
        if gravity is True:
            im_opened_static = dilate_im_gravity(im_distance2, im,
                                                 step, Deros0, gamma,
                                                 rhoWP, rhoNWP,
                                                 h_max, g, p0, D2img, dr, upp)
        else:
            im_opened_static = _dilate_im(im_distance2, im, D/2, 0, upp)

        # Eliminate overlapping phase
        im_opened_static = eliminate_overlapping(im_opened_static, NwpR, D2img)

        # Create flow behaviour picture by multiply new intruded NWP
        # with value 'count'
        im_opened_dynamic = im_opened_static * count

        # Store the radius
        R.append(D / 2)
        pc.append(p0)

        # Calculate the NW phase saturation
        s = ps.metrics.porosity(im_opened_static)
        Snwp.append(s / P)

        # Add the original image
        # For illustration only
        temp = np.where((im_opened_dynamic - im_saturated) == count, count, 0)
        im_saturated += temp

        # Save eroded profile
        im_save = (im_connected_bool * (-1) + 1) * countA
        images2.append(np.where(im_eroded == upp, countA, 0) +
                       img_different_solid_phase)
        # Save flow status
        images.append(np.where(im_opened_static == 1, countA, 0) +
                      img_different_solid_phase)

        # Raise progress_var to show calculation progress in %
        progress_var += 1
        progress = int(np.floor(progress_var / saturation_steps * 100))
        print(progress)

    # Smaller diameter by resolution
    D = D - resolution
    # Set count to next step for flow behaviour image
    count += 1

# %% Bubble Points

bubble_array = np.where(im_saturated < countA, upp, 0)
bubble_array = im_saturated + bubble_array

if D2img is True:
    bubblepoints = np.zeros([4])
    # Bubblepoint -x
    bubblepoints[0] = np.min(bubble_array[0, :])
    # Bubblepoint x
    bubblepoints[1] = np.min(bubble_array[Lx - 1, :])
    # Bubblepoint -y
    bubblepoints[2] = np.min(bubble_array[:, 0])
    # Bubblepoint y
    bubblepoints[3] = np.min(bubble_array[:, Ly - 1])
else:
    bubblepoints = np.zeros([6])
    # Bubblepoint -x
    bubblepoints[0] = np.min(bubble_array[0, :, :])
    # Bubblepoint x
    bubblepoints[1] = np.min(bubble_array[Lx - 1, :, :])
    # Bubblepoint -y
    bubblepoints[2] = np.min(bubble_array[:, 0, :])
    # Bubblepoint y
    bubblepoints[3] = np.min(bubble_array[:, Ly - 1, :])
    # Bubblepoint -z
    bubblepoints[4] = np.min(bubble_array[:, :, 0])
    # Bubblepoint z
    bubblepoints[5] = np.min(bubble_array[:, :, Lz - 1])
    pass

bubblepoints = (D_max - (bubblepoints - countA) * resolution) / 2
bubblepoints2 = 2 * gamma / bubblepoints / dr

print(bubblepoints2)

# %%  Show Time

end = time.time()
diff = float(end) - float(start)
minutes = np.floor(diff)
seconds = diff - minutes
minutes = minutes / 60
seconds += (minutes - np.floor(minutes)) * 60
print("%d min %5.2f s" % (minutes, seconds))


# %% Save Images

path = os.path.join(dirname, "images")
try:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
except:
    raise Exception("Couldn't save images correctly")

start_i = len(R) - len(images) + 1

if D2img is True:
    cmap_im = "Blues"
    i = start_i
    for item in images:
        path2 = os.path.join(path, "dilated%s_%5.4f.png" % (i, R[i - 1]))
        plt.imsave(path2, item, origin="lower", cmap=cmap_im)
        i += 1
        pass
    i = start_i
    for item in images2:
        path2 = os.path.join(path, "eroded%s_%5.4f.png" % (i, R[i - 1]))
        plt.imsave(path2, item, origin="lower", cmap=cmap_im)
        i += 1
        pass

else:
    cmap_im = "Blues"
    i = start_i
    for item in images:
        for cut in range(0, Lz, int(Lz / 4)):
            path2 = os.path.join(path, "dilated%s_%5.4f_%s.png" % (i, R[i - 1], cut))
            plt.imsave(path2, item[:, :, cut], origin="lower", cmap=cmap_im)
        i += 1
        pass
    i = start_i
    for item in images2:
        for cut in range(0, Lz, int(Lz / 4)):
            path2 = os.path.join(path, "eroded%s_%5.4f_%s.png" % (i, R[i - 1], cut))
            plt.imsave(path2, item[:, :, cut], origin="lower", cmap=cmap_im)
        i += 1
        pass
    pass

# Create vtk file
ps.io.to_vtk(im_saturated, os.path.join(path, "vtk_file.vtk"))


# %% Plot
def graphplot(data, title, xlabel, ylabel):
    with plt.style.context("seaborn-whitegrid"):
        plt.rcParams.update({"font.size": 12})
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title, fontsize=14, fontweight=0, color="black", loc="left")
        ax.set_xlabel(xlabel, fontsize=12, fontweight=0, color="black")
        ax.set_ylabel(ylabel, fontsize=12, fontweight=0, color="black")
        ax.set_xlim(left=0, right=1)
        ax.set_ylim(bottom=0, top=max(data[1, :]) * 1.1)
        ax.plot(data[0, :], data[1, :])


d = namedtuple("xy_data", ("radius", "saturation"))

if Sat_Curve is True:
    R2 = []
    for item in R:
        R2.append(item * dr * 1000)

    data1 = np.zeros([2, len(Snwp) - 1])
    data1[0, :] = np.asarray(Snwp[1:len(Snwp)])
    data1[1, :] = np.asarray(R2)
    graphplot(data1, "Radius-Saturation-Diagramm", "Saturation [%]", "Radius [mm]")

    for item in bubblepoints:
        plt.plot([0, 1], [item * dr * 1000, item * dr * 1000])
        pass

    data2 = np.zeros([2, len(Snwp)])
    data2[0, :] = np.asarray(Snwp)
    data2[1, :] = np.asarray(pc)
    graphplot(data2,
              "Pressure-Saturation-Diagramm",
              "Saturation [%]",
              "Pressure [Pa]")

    for item in bubblepoints2:
        plt.plot([0, 1], [item, item])
        pass

if D2img is True:
    fig = plt.figure(figsize=(9, 6))
    plt.imshow(im_saturated[:, :], origin="lower")
else:
    fig = plt.figure(figsize=(9, 6))
    plt.imshow(im_saturated[:, :, int(im_saturated.shape[2] / 2)])

pressure_saturation = []
pressure_saturation.append(Snwp)
pressure_saturation.append(pc)
pressure_saturation = np.asarray(pressure_saturation)
path_s_p = os.path.join(dirname, "pressure_saturation.txt")
np.savetxt(path_s_p, np.ravel(pressure_saturation))

plt.show()
