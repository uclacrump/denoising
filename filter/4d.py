import numpy as np
from scipy.ndimage import gaussian_filter
import imageio

def import_data(file_path):
    with open(file_path, 'rb') as file:
        return np.fromfile(file, dtype=np.float32)

def load_images(file_path):
    data = import_data(file_path)
    vol = data.reshape((-1, 200, 380, 380))
    print(vol.shape)
    return vol

def apply_gaussian_filter(time_x_y_z_image_volume, sigma, radius):
    return gaussian_filter(time_x_y_z_image_volume, sigma=sigma, radius=radius)

def slice_4d_volume(volume_4d):
    return volume_4d[:, :, 144, :]

def create_video(volume_3d, output_path):
    # Adjust the slices for better visibility in the video
    slices_adjusted = (volume_3d - volume_3d.min()) / (volume_3d.max() - volume_3d.min()) * 255
    slices_adjusted = slices_adjusted.astype(np.uint8)

    num_loops = 5
    slices_loop = np.tile(slices_adjusted, (num_loops, 1, 1))

    imageio.mimsave(output_path, slices_loop, fps=20)

def load_and_filter_images(file_path, sigma, radius):
    time_x_y_z_image_volume = load_images(file_path)
    filtered_time_x_y_z_image_volume = apply_gaussian_filter(time_x_y_z_image_volume, sigma, radius)
    return time_x_y_z_image_volume, filtered_time_x_y_z_image_volume

file_path = "amide_export_t58783_feldkamp"
sigma = 4
radius = 9

original_4d_volume, filtered_4d_volume = load_and_filter_images(file_path, sigma, radius)
sliced_3d_volume = slice_4d_volume(original_4d_volume)
sliced_filtered_3d_volume = slice_4d_volume(filtered_4d_volume)

# # Create videos
create_video(sliced_3d_volume, 'original_video.mp4')
create_video(sliced_filtered_3d_volume, 'filtered_video.mp4')
