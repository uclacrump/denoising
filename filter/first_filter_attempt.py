import os
import glob
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
import imageio
import glob
import pdb

# Load and resize images. NOTE: This is wonky and just has to do with the source images being screenshots. Whole function is essentially ignorable.
def load_and_resize_images(dir_path):
    files = glob.glob(os.path.join(dir_path, "*.png"))
    dimensions = {}
    images = {}
    for file in files:
        print(file)
        image = Image.open(file)
        dimensions[file] = image.size
        images[file] = image
    min_width = min(dimensions[file][0] for file in files)
    min_height = min(dimensions[file][1] for file in files)
    resized_images = {file: image.resize((min_width, min_height)) for file, image in images.items()}
    return resized_images

# Apply Gaussian filter to an image volume
def apply_gaussian_filter(image_volume, sigma, radius):
    filtered_volume = np.empty_like(image_volume)
    for i in range(3):  # Apply the filter to each of the R, G, B channels separately
        filtered_volume[..., i] = gaussian_filter(image_volume[..., i], sigma, radius=radius)
    if image_volume.shape[-1] == 4:  # If the images have an alpha channel, copy it over without filtering
        # filtered_volume[..., 3] = image_volume[..., 3]
    return filtered_volume

# Save filtered images
def save_filtered_images(filtered_images, dir_path):
    for file, image in filtered_images.items():
        filename, ext = os.path.splitext(os.path.basename(file))
        output_filename = f"{filename}{ext}"
        output_path = os.path.join(dir_path, output_filename)
        image.save(output_path)

# Remove filtered images
def remove_filtered_images(dir_path):
    files = glob.glob(os.path.join(dir_path, "*_filtered.png"))
    for file in files:
        os.remove(file)

# Create movies
def create_movies(original_images, filtered_images, dir_path, repeat_times=30):
    original_movie_path = os.path.join(dir_path, 'original_movie.mp4')
    filtered_movie_path = os.path.join(dir_path, 'filtered_movie.mp4')

    # Sort original_images and filtered_images keys in numeric order
    sorted_original_keys = sorted(original_images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    sorted_filtered_keys = sorted(filtered_images, key=lambda x: int(os.path.splitext(os.path.basename(x.replace('_filtered', '')))[0]))

    print(f'Sorted original keys: {sorted_original_keys}')
    print(f'Sorted filtered keys: {sorted_filtered_keys}')

    # Create sorted original_images and sorted_filtered_images based on the sorted keys
    sorted_original_images = [original_images[key] for key in sorted_original_keys]
    sorted_filtered_images = [filtered_images[key] for key in sorted_filtered_keys]

    # Create repeated frames for original and filtered images
    repeated_original_images = sorted_original_images * repeat_times
    repeated_filtered_images = sorted_filtered_images * repeat_times

    # Create the actual movies
    imageio.mimsave(original_movie_path, [np.array(image) for image in repeated_original_images], fps=15, format='mp4')
    imageio.mimsave(filtered_movie_path, [np.array(image) for image in repeated_filtered_images], fps=15, format='mp4')

# Load and filter images
def load_and_filter_images(dir_path, sigma, radius):
    images = load_and_resize_images(dir_path)
    # Stack images to create a 3D volume
    image_volume = np.stack([np.array(image) for image in images.values()])

    # Apply gaussian filter to the volume
    filtered_volume = apply_gaussian_filter(image_volume, sigma, radius)

    
    # Apply median filter to the volume. For time dimension, we look one frame ahead and one behind. For the other dimensions, we just use the original value.
    # filtered_volume = median_filter(filtered_volume, size=(3, 1, 1, 1))
    
    # Convert filtered volume back to individual image dict
    filtered_images = {file.replace(".png", "_filtered.png"): Image.fromarray(filtered_volume[i]) for i, file in enumerate(images.keys())}
    return images, filtered_images

dir_path = 'mouse_images'

# Gaussian kernel params.
sigma = 4
radius = 9

original_images, filtered_images = load_and_filter_images(dir_path, sigma, radius)
save_filtered_images(filtered_images, dir_path)
create_movies(original_images, filtered_images, dir_path)

# Kind of hacky: These will hang around in the directory and get refiltered by subsequent runs unless they are removed at the end.
remove_filtered_images(dir_path)
