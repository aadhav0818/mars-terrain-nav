from dbm import error
import time

import numpy as np
from noise import pnoise2
import matplotlib.pyplot as plt
from sympy import true
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
lacunarity = 1.8

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
resolution_width = 320
resolution_height = 180

Hfov = np.deg2rad(90) 
Vfov = 2*np.arctan(np.tan(Hfov/2) * (resolution_height/resolution_width))

fx = (resolution_width / 2) / np.tan(Hfov / 2)
fy = (resolution_height / 2) / np.tan(Vfov / 2)

cx = resolution_width / 2
cy = resolution_height / 2

R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
pitch = np.deg2rad(0)
yaw = np.deg2rad(0)
roll = np.deg2rad(0)
R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
R_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
R_roll = np.array([[np.cos(roll), 0, np.sin(roll)], [0, 1, 0], [-np.sin(roll), 0, np.cos(roll)]])
R_orientation = R_roll @ R_pitch @ R_yaw @ R_x

epsilon = 0.1

image = np.zeros((resolution_height, resolution_width))
fig, ax = plt.subplots()
img = ax.imshow(image, cmap="viridis", vmin=0, vmax=1000)
ax.axis("off")
plt.show(block=False)

# Camera Simulation to World Simulation
# COORDINATE CONVENTION: +Z is up relative to the terrain. Camera needs to rotate 180 degrees around the X-axis to look down at the terrain.
for pos in tq.tqdm(camera_pos):
    image.fill(0)
    camera_pos = pos
    for u in tq.tqdm(range(resolution_width)):
        for v in range(resolution_height):
            ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1])
            ray_cam = ray_cam / np.linalg.norm(ray_cam)
            ray_world = R_orientation @ ray_cam

            step_size = 50

            # Bracketing from [t_low, t_high] 
            t_low = 0
            t_high = None

            while True:
                point = camera_pos + t_low*ray_world
                point_x = point[0]
                point_y = point[1]
                point_z = point[2]
                # Check if ray exceeds image bounds (should not really be an issue with an actual map i think)
                if point_x < 0 or point_x >= width or point_y < 0 or point_y >= height:
                    break

                terrain_z = terrain_height(point_x, point_y)
                F = point_z - terrain_z
                if F <= 0:
                    t_high = t_low
                    t_low = t_low - step_size
                    break

                t_low += step_size

            if t_high is not None:
                while (t_high - t_low) > epsilon:

                    t_mid = (t_low + t_high) / 2

                    point = camera_pos + t_mid * ray_world

                    point_x = point[0]
                    point_y = point[1]
                    point_z = point[2]

                    terrain_z = terrain_height(point_x, point_y)

                    # F(t)
                    F = point_z - terrain_z

                    if F > 0:
                        # Ray is above terrain
                        t_low = t_mid

                    else:
                        # Ray is below terrain
                        t_high = t_mid

                t_intersection = (t_low + t_high) / 2

                point = camera_pos + t_intersection * ray_world

                point_x = point[0]
                point_y = point[1]

                image[v][u] = terrain_height(point_x, point_y)
                    
     
    img.set_data(image)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.5)

