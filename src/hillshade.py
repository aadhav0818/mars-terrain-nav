import numpy as np

def calculate_normals(terrain, pixel_size):
    dz_dy, dz_dx = np.gradient(terrain, pixel_size, pixel_size);
    tangent_x = np.stack([np.ones_like(terrain), np.zeros_like(terrain), dz_dx], axis=-1)
    tangent_y = np.stack([np.zeros_like(terrain), np.ones_like(terrain), dz_dy], axis=-1)
    surface_normals = np.cross(tangent_x, tangent_y)
    surface_normals = surface_normals / np.linalg.norm(surface_normals, axis=-1, keepdims=True)
    return surface_normals

def calculate_illumination(surface_normals, sun_azimuth=315, sun_elevation=45):
    sun_azimuth = np.deg2rad(sun_azimuth)
    sun_elevation = np.deg2rad(sun_elevation)
    sun_direction = np.array([np.cos(sun_elevation) * np.sin(sun_azimuth), np.cos(sun_elevation) * np.cos(sun_azimuth), np.sin(sun_elevation)])
    illumination = np.sum(surface_normals * sun_direction, axis=-1)
    illumination = np.maximum(illumination, 0)
    return illumination

def get_hillshaded_terrain(terrain, pixel_size):
    normals = calculate_normals(terrain, pixel_size)
    illumination = calculate_illumination(normals)
    return illumination

def get_hillshaded_camera_image(x_intersection_m, y_intersection_m, illumination, pixel_size):
    x_pixels = (x_intersection_m / pixel_size).astype(int)
    y_pixels = (y_intersection_m / pixel_size).astype(int)
    height, width = illumination.shape
    x_index = np.clip(x_pixels, 0, width-1)
    y_index = np.clip(y_pixels, 0, height-1)
    image = illumination[y_index, x_index]
    return image
