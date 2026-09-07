import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from simulate_camera import simulate_camera

dataset = gdal.Open(r'./digital-terrain-models\DTEEC_048842_1985_048908_1985_U01.IMG')

band = dataset.GetRasterBand(1)
DTM = band.ReadAsArray()
nodata = band.GetNoDataValue()
valid = DTM != nodata
rows, cols = np.where(valid)
DTM = np.where(DTM == nodata, np.nan, DTM)
pixel_size = dataset.GetGeoTransform()[1]
height, width = DTM.shape
x = np.arange(width) * pixel_size
y = np.arange(height) * pixel_size

# Smaller selection of the DTM for visualization (2x2 km)
center_row = height // 2
center_col = width // 2
cropped_dtm_m = 2000
cropped_dtm_pixels = int(cropped_dtm_m / pixel_size) 
cropped_dtm_pixels_half = cropped_dtm_pixels // 2
crop_DTM = DTM[
    center_row - cropped_dtm_pixels_half:center_row + cropped_dtm_pixels_half,
    center_col - cropped_dtm_pixels_half:center_col + cropped_dtm_pixels_half
]

x_crop_min = (center_col - cropped_dtm_pixels_half) * pixel_size
x_crop_max = (center_col + cropped_dtm_pixels_half) * pixel_size
y_crop_min = (center_row - cropped_dtm_pixels_half) * pixel_size
y_crop_max = (center_row + cropped_dtm_pixels_half) * pixel_size

crop_height, crop_width = crop_DTM.shape

# Localized coordinate system for camera simulation 
# x and y start at 0, elevation remains the same as in the original DTM
x_local = np.arange(crop_width) * pixel_size
y_local = np.arange(crop_height) * pixel_size

fig1, ax1 = plt.subplots()
img1 = ax1.imshow(DTM, cmap='terrain', extent=(x[0], x[-1], y[0], y[-1]))
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Jezero Crater 15x7 km Digital Terrain Model')
fig1.colorbar(img1, label='Elevation (m)')
fig1.show()

fig2, ax2 = plt.subplots()
img2 = ax2.imshow(crop_DTM, cmap='terrain', extent=(x_crop_min, x_crop_max, y_crop_min, y_crop_max), vmin = np.nanmin(DTM), vmax = np.nanmax(DTM))
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('Jezero Crater 2x2 km (Cropped) Digital Terrain Model')
fig2.colorbar(img2, label='Elevation (m)') 
fig2.show()


fig3, ax3 = plt.subplots()
img3 = ax3.imshow(simulate_camera(crop_DTM, np.array([1000, 1000, 500]), pixel_size), 
                  cmap='terrain', 
                  vmin = np.nanmin(DTM), 
                  vmax = np.nanmax(DTM)
                )

plt.show()
# Camera simulation with HiRISE imagery instead of synthetic perlin terrain
