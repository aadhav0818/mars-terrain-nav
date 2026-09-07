from dbm import error
import time

import numpy as np
from noise import pnoise2
import matplotlib.pyplot as plt
from sympy import true
from matplotlib.animation import FuncAnimation
import tqdm as tq 

def terrain_height(x,y):
    if x < 0 or x >= width or y < 0 or y >= height:
        return None
    return normalized_terrain[int(y)][int(x)]

width = 1000
height = 1000
scale = 300.0
octaves = 6
persistence = 0.5
lacunarity = 3

terrain = np.zeros((height, width))

for i in range(height):
    for j in range(width):
        terrain[i][j] = pnoise2(i / scale,
                                j / scale,
                                octaves=octaves,
                                persistence=persistence,
                                lacunarity=lacunarity,
                                repeatx=width,
                                repeaty=height,
                                base=42)

# Normalize terrain values to 0–1000
normalized_terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 1000

# Grayscale heightmap
plt.figure(1)
plt.imshow(normalized_terrain, cmap='gray')
plt.title("Grayscale Heightmap")
plt.colorbar()
plt.show(block=False)

plt.figure(2)
x_range = np.linspace(0, 1000, height);
y_range = np.linspace(0, 1000, width);
x, y = np.meshgrid(x_range, y_range);
z = normalized_terrain;
minimum_altitude = np.min(z)
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
plt.show(block=False)

# Camera Simulation
# ASSUMPTIONS: square pixels, camera looking straight down, no distortions
# COORDINATE CONVENTION: +Z is forward relative to the optics
camera_pos = np.array([[500, 500, 2000], [500, 500, 1500], [500, 500, 1200], [500, 600, 1200], [500, 800, 1200], [500, 1000, 1200]])
resolution_width = 1280
resolution_height = 780

Hfov = np.deg2rad(90) 
Vfov = 2*np.arctan(np.tan(Hfov/2) * (resolution_height/resolution_width))

fx = (resolution_width / 2) / np.tan(Hfov / 2)
fy = (resolution_height / 2) / np.tan(Vfov / 2)

cx = resolution_width / 2
cy = resolution_height / 2

R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
pitch = np.deg2rad(5)
yaw = np.deg2rad(0)
roll = np.deg2rad(0)
R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
R_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
R_roll = np.array([[np.cos(roll), 0, np.sin(roll)], [0, 1, 0], [-np.sin(roll), 0, np.cos(roll)]])
R_orientation = R_roll @ R_pitch @ R_yaw @ R_x

epsilon = 0.1

## VECTORIZED VERSION

def simulate_camera(camera_pos):
    u, v = np.meshgrid(np.arange(resolution_width), np.arange(resolution_height))
    ray_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    ray_cam = ray_cam / np.linalg.norm(ray_cam, axis=-1, keepdims=True)
    ray_world = ray_cam @ R_orientation.T

    t = np.zeros((resolution_height, resolution_width))
    point = camera_pos + t[:, :, np.newaxis] * ray_world
    x = point[:, :, 0]
    y = point[:, :, 1]
    z = point[:, :, 2]
    x_index = np.clip(x.astype(int), 0, width - 1)
    y_index = np.clip(y.astype(int), 0, height - 1)
    terrain_z = normalized_terrain[y_index, x_index] # The row is technically the y coordinate, thus the notation

    t_low = np.zeros((resolution_height, resolution_width))
    t_high = np.zeros((resolution_height, resolution_width))

    F = z - terrain_z
    active_rays = np.ones((resolution_height, resolution_width), dtype=bool)
    new_ray_collisions = active_rays & (F <= 0)
    collided_rays = np.zeros((resolution_height, resolution_width), dtype=bool)

    step_size = 50
    while np.any(active_rays):
        t_previous = t.copy()
        t[active_rays] += step_size

        point = camera_pos + t[:, :, np.newaxis] * ray_world
        x = point[:, :, 0]
        y = point[:, :, 1]
        z = point[:, :, 2]

        map_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        active_rays[active_rays & ~map_bounds] = False

        x_index = np.clip(x.astype(int), 0, width - 1)
        y_index = np.clip(y.astype(int), 0, height - 1)
        terrain_z = normalized_terrain[y_index, x_index]

        F = z - terrain_z
        new_ray_collisions = active_rays & (F <= 0)
        collided_rays[new_ray_collisions] = True

        t_low[new_ray_collisions] = t_previous[new_ray_collisions]
        t_high[new_ray_collisions] = t[new_ray_collisions]
        
        active_rays[new_ray_collisions] = False

    epsilon = 0.1 # acceptable tolerance for this method. It's not perfect, but it should be good enough and computationally inexpensive for this simulation.

    while np.any(collided_rays & ((t_high - t_low) > epsilon)):
        t_mid = (t_low + t_high) / 2

        point = camera_pos + t_mid[:, :, np.newaxis] * ray_world
        x = point[:, :, 0]
        y = point[:, :, 1]
        z = point[:, :, 2]

        x_index = np.clip(x.astype(int), 0, width - 1)
        y_index = np.clip(y.astype(int), 0, height - 1)
        terrain_z = normalized_terrain[y_index, x_index]

        F = z - terrain_z

        negative = F <= 0
        positive = F > 0

        t_low[positive] = t_mid[positive]
        t_high[negative] = t_mid[negative]

    t_intersection = (t_low + t_high) / 2
    point = camera_pos + t_intersection[:, :, np.newaxis] * ray_world
    point_z = point[:, :, 2]
    image = np.zeros((resolution_height, resolution_width))
    image[collided_rays] = point_z[collided_rays]

    return image
    

images = []

for pos in tq.tqdm(camera_pos):
    images.append(simulate_camera(pos))

fig, ax = plt.subplots()

img = ax.imshow(images[0], cmap="viridis", vmin=0, vmax=1000)
ax.axis("off")

def update(frame):
    img.set_data(images[frame])
    return [img]

animation = FuncAnimation(fig, update, frames=len(images), interval=500)

plt.show()