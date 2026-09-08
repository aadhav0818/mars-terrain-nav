import numpy as np

# Input parameters
resolution_width = 1280
resolution_height = 780
Hfov_degrees = 60
pitch = np.deg2rad(0)
yaw = np.deg2rad(0)
roll = np.deg2rad(0)
epsilon = 0.1

Hfov = np.deg2rad(Hfov_degrees)
Vfov = 2*np.arctan(np.tan(Hfov/2) * (resolution_height/resolution_width))
fx = (resolution_width / 2) / np.tan(Hfov / 2)
fy = (resolution_height / 2) / np.tan(Vfov / 2)
cx = resolution_width / 2
cy = resolution_height / 2

R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
R_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
R_roll = np.array([[np.cos(roll), 0, np.sin(roll)], [0, 1, 0], [-np.sin(roll), 0, np.cos(roll)]])
R_orientation = R_roll @ R_pitch @ R_yaw @ R_x

def simulate_camera(terrain, camera_pos, pixel_size):
    u, v = np.meshgrid(np.arange(resolution_width), np.arange(resolution_height))
    ray_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    ray_cam = ray_cam / np.linalg.norm(ray_cam, axis=-1, keepdims=True)
    ray_world = ray_cam @ R_orientation.T

    height, width = terrain.shape
    t = np.zeros((resolution_height, resolution_width))
    point = camera_pos + t[:, :, np.newaxis] * ray_world
    x = point[:, :, 0]
    y = point[:, :, 1]
    z = point[:, :, 2]
    x_pixels = (x / pixel_size).astype(int)
    y_pixels = (y / pixel_size).astype(int)
    x_index = np.clip(x_pixels, 0, width - 1)
    y_index = np.clip(y_pixels, 0, height - 1)
    terrain_z = terrain[y_index, x_index] 

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

        map_bounds = (x >= 0) & (x < width * pixel_size) & (y >= 0) & (y < height * pixel_size)
        active_rays[active_rays & ~map_bounds] = False

        x_pixels = (x / pixel_size).astype(int)
        y_pixels = (y / pixel_size).astype(int)
        x_index = np.clip(x_pixels, 0, width - 1)
        y_index = np.clip(y_pixels, 0, height - 1)
        terrain_z = terrain[y_index, x_index]

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

        x_pixels = (x / pixel_size).astype(int)
        y_pixels = (y / pixel_size).astype(int)
        x_index = np.clip(x_pixels, 0, width - 1)
        y_index = np.clip(y_pixels, 0, height - 1)
        terrain_z = terrain[y_index, x_index]

        F = z - terrain_z

        negative = F <= 0
        positive = F > 0

        t_low[positive] = t_mid[positive]
        t_high[negative] = t_mid[negative]

    t_intersection = (t_low + t_high) / 2
    point = camera_pos + t_intersection[:, :, np.newaxis] * ray_world
    point_z = point[:, :, 2]
    point_x = point[:, :, 0]
    point_y = point[:, :, 1]
    # image = np.zeros((resolution_height, resolution_width))
    # image[collided_rays] = point_z[collided_rays]

    return point_x, point_y, collided_rays
    
