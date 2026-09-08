import matplotlib.pyplot as plt
import numpy as np
import tqdm as tq
from osgeo import gdal
from simulate_camera import simulate_camera
from hillshade import calculate_illumination, calculate_normals, get_hillshaded_terrain, get_hillshaded_camera_image
from matplotlib.animation import FuncAnimation
import cv2

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

illumination = get_hillshaded_terrain(crop_DTM, pixel_size)
fig4, ax4 = plt.subplots()
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.set_title('Jezero Crater 2x2 km (Cropped) Hillshaded Digital Terrain Model')
img4 = ax4.imshow(illumination, cmap='gray', extent=(x_crop_min, x_crop_max, y_crop_min, y_crop_max))
fig4.show()

fig3, ax3 = plt.subplots()

# Camera simulation with HiRISE imagery instead of synthetic perlin terrain

images = []
camera_pos = np.array([[1000, 1200, -1000]])

for pos in tq.tqdm(camera_pos):
    x_intersection, y_intersection, collided_rays = simulate_camera(crop_DTM, pos, pixel_size)
    image = get_hillshaded_camera_image(x_intersection, y_intersection, illumination, pixel_size)
    image[~collided_rays] = 0
    images.append(image)

img3 = ax3.imshow(images[0], cmap='gray')
ax3.axis("off")

def update(frame):
    img3.set_data(images[frame])
    return [img3]

animation = FuncAnimation(fig3, update, frames=len(images), interval=500)

# FEATURE DETECTION

reference_image = (illumination * 255).astype(np.uint8)
camera_image = (images[0] * 255).astype(np.uint8)
sift = cv2.SIFT_create()

reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image, None)
camera_keypoints, camera_descriptors = sift.detectAndCompute(camera_image, None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(camera_descriptors, reference_descriptors, k=2
)

good_matches = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append(m)

# RANSAC

src_pts = np.float32([camera_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([reference_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

inlier_matches = [m for m, inlier in zip(good_matches, mask.ravel()) if inlier]

print("Good matches:", len(good_matches))
print("RANSAC inliers:", len(inlier_matches))

match_image = cv2.drawMatches(
    camera_image,
    camera_keypoints,
    reference_image,
    reference_keypoints,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# vertical line separating the two images
divider_x = camera_image.shape[1]
cv2.line(match_image, (divider_x, 0), (divider_x, match_image.shape[0]), (255, 255, 255), 3)

fig5, ax5 = plt.subplots()
ax5.imshow(match_image)
ax5.axis("off")

terrain_points = []

for m in inlier_matches:

    # Reference image pixel coordinates
    u_ref, v_ref = reference_keypoints[m.trainIdx].pt

    # Convert reference pixel to DTM indices
    x_index = int(u_ref)
    y_index = int(v_ref)

    # Convert DTM indices to local terrain coordinates
    x = x_index * pixel_size
    y = (crop_height - 1 - y_index) * pixel_size
    z = crop_DTM[y_index, x_index]

    terrain_points.append([x, y, z])

terrain_points = np.array(terrain_points)

print("Terrain points:", terrain_points.shape)
print("First 5:")
print(terrain_points[:5])

# 2D camera points corresponding to the 3D terrain points
camera_points = np.float32([camera_keypoints[m.queryIdx].pt for m in inlier_matches])

terrain_points = np.float32(terrain_points)

# Input parameters
resolution_width = 1280
resolution_height = 780
Hfov_degrees = 60

Hfov = np.deg2rad(Hfov_degrees)
Vfov = 2*np.arctan(np.tan(Hfov/2) * (resolution_height/resolution_width))
fx = (resolution_width / 2) / np.tan(Hfov / 2)
fy = (resolution_height / 2) / np.tan(Vfov / 2)
cx = resolution_width / 2
cy = resolution_height / 2

# Camera intrinsic matrix
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float64)

# No lens distortion (for now)
dist_coeffs = np.zeros((4, 1))

success, rvec, tvec, inliers = cv2.solvePnPRansac(
    terrain_points,
    camera_points,
    K,
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)

print("PnP success:", success)
print("PnP inliers:", len(inliers))
print("rvec:", rvec.ravel())
print("tvec:", tvec.ravel())

R_camera, _ = cv2.Rodrigues(rvec)

camera_position_estimated = -R_camera.T @ tvec

print("Estimated camera position:")
print(camera_position_estimated.ravel())
print("Position error:")
print(np.linalg.norm(camera_position_estimated.ravel()-camera_pos[0]))
plt.show()