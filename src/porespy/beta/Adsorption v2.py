# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 13:27:52 2025

@author: sebva
"""

import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
from edt import edt
import matplotlib.animation as animation
import imageio_ffmpeg
from matplotlib import rcParams

vol=True
#Physical Properties
T=77
R=8.314
gam=8.85*10**-3
vm=28.5*10**-6

Original_Image=ps.generators.blobs(shape=[500,500],porosity=0.6,blobiness=3) #create image of porous medium
Original_Image=ps.filters.fill_invalid_pores(Original_Image)
film_thickness=np.linspace(0.05,7,100) #Increasing film thickness
relative_pressure=10**((0.034-13.99/(100*film_thickness**2))/0.4343) #rearanged Harkins-Jura Equation for relative pressure
r=lambda p,th:-gam*vm/(R*T*np.log(p))+th # Kelvin Cohan equation to determine radius for capilary condensation

im=[Original_Image] #Array containing each image
epsilon=[] # Array which will hold porosity values for each image
adsorbed_volume=[]
dt=edt(Original_Image) #euclidian distance transform on initial image

for i in range(np.size(film_thickness)):
    capiliary_radius=film_thickness[i]+r(relative_pressure[i],film_thickness[i]) #radius of our structuring element for the capilary condensation

    film_adsoprtion=(dt<=film_thickness[i])*Original_Image #isolating the film around the pores

    #Closing opreation using structuring element of radius rn
    Closing_erosion=(edt(Original_Image*~film_adsoprtion)>capiliary_radius)
    Closing_dilation=(edt(~Closing_erosion))<=capiliary_radius

    adsorbed_volume.append(np.sum(Original_Image)-np.sum(Closing_dilation))
    epsilon.append(Closing_dilation.sum()/Closing_dilation.size)
    im.append(Closing_dilation)

fig,ax=plt.subplots(2,2)
ax[0,0].plot(relative_pressure,film_thickness)
ax[0,0].set_xlabel("Relative Pressure (p/po)")
ax[0,0].set_ylabel("Film Thickness")
if vol==True:
    ax[0,1].plot(relative_pressure,adsorbed_volume)
    ax[0,1].set_xlabel("Relative Pressure (p/po")
    ax[0,1].set_ylabel("Volume Adsorbed")
elif vol!=True:
    ax[0,1].plot(relative_pressure,epsilon)
    ax[0,1].set_xlabel("Relative Pressure (p/po)")
    ax[0,1].set_ylabel("Porosity")
ax[1,0].imshow(Original_Image)
ax[1,0].axis(False);
ax[1,1].imshow(Closing_dilation)
ax[1,1].axis(False);

fig2, ax2 = plt.subplots()
animated_image= ax2.imshow(im[0], animated=True)
ax2.axis("off")
def update(frame):
    animated_image.set_array(im[frame])
    return (animated_image,)

ani = animation.FuncAnimation(fig=fig2,func=update,frames=len(im), interval=60,blit=True)
rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
ani.save("Adsoprtion.mp4",writer='ffmpeg')
plt.show()
#%%
imfinal=Closing_dilation
adsrobed_volume2=[]
im2=[imfinal]
for i in range(np.size(film_thickness)):
    #print(t[-(i+1)])
    capiliary_radius=film_thickness[-(i+1)]+r(relative_pressure[-(i+1)],film_thickness[-(i+1)])
    #Opening using structuring element of radius rn
    Opening_erosion=(edt(Original_Image))>capiliary_radius
    Opening_dilation=edt(~Opening_erosion)<=capiliary_radius

    film=edt(Opening_dilation)>film_thickness[[-(i+1)]]
    filmc=ps.filters.fill_closed_pores(film)
    im2.append(filmc)
    adsrobed_volume2.append(np.sum(Original_Image)-np.sum(filmc))
    # plt.imshow(filmc/imfinal)
    # plt.show()

plt.plot(relative_pressure,adsorbed_volume,label='Adsorption')
plt.plot(relative_pressure,adsrobed_volume2[::-1],label='Desorption')
plt.legend()

fig2, ax2 = plt.subplots()
animated_image= ax2.imshow(im2[0], animated=True)
ax2.axis("off")
def update(frame):
    animated_image.set_array(im2[frame])
    return (animated_image,)

ani = animation.FuncAnimation(fig=fig2,func=update,frames=len(im2), interval=60)
plt.show()