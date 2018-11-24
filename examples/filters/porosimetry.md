# Porosimetry and Local Thickness

## Import Packages

``` python
import porespy as ps
import matplotlib.pyplot as plt
```

# Generate Artificial Image
 ```python
im = ps.generators.blobs(shape=[800,800], blobiness=[2,1.5],porosity=0.60)
fig, ax0 = plt.subplots(figsize=(10,10))
ax0.imshow(im)
ax0.axis('off')
```
![Imgur](https://i.imgur.com/u249RfT.png)

# Apply the Local Thickness and Porosimetry Filters
```python
lt = ps.filters.local_thickness(im,sizes=50)
psd = ps.filters.porosimetry(im,sizes=50,access_limited=True,mode='fft')

fig, (ax0,ax1) = plt.subplots(ncols=2,figsize=(20,10))
im0 = ax0.imshow(lt)
fig.colorbar(im0,ax=ax0,shrink=0.75)
im1 = ax1.imshow(psd)
fig.colorbar(im1,ax=ax1,shrink=0.75)
fig.savefig(r'../images_to_upload/psd/lt_psd',dpi=300)
```
![Imgur](https://i.imgur.com/pCYVg0N.png)

Note: the main difference between the ```local_thickness``` and ```porosimetry``` filters is the ```access_limited``` keyword. With it toggled ```True``` the filter allows shielding from smaller pores closer to the edges or inlets of the image. ```local_thickness``` gives the true pore or feature size distribution while the ```porosimetry``` filter simulates the invasion process similar to mercury intrusion porosimetry

# Calculate the Distributions and Analyze the Results

![Imgur](https://i.imgur.com/xsf7noz.png)

The difference in the shape of the curves shown above allows for the determination of the 'shielding' present in the material.
Alternatively if you prefer you can generate histograms of the probability density function of the materials.

![Imgur](https://i.imgur.com/uy6AkvT.png)




