import numpy as np
import matplotlib.pyplot as plt
import os
from misc import calculate_cropped_index_of_interest, calculate_downsampled_index_of_interest, load_images, export_data
import sys

def crop_volume(volume, new_shape, index_of_interest, dim_of_interest):
    time, depth, rows, cols = volume.shape
    _, new_depth, new_rows, new_cols = new_shape
    
    start_depth = (depth - new_depth) // 2
    start_row = (rows - new_rows) // 2
    start_col = (cols - new_cols) // 2

    new_index = calculate_cropped_index_of_interest(index_of_interest, dim_of_interest, volume.shape, new_shape)
    
    if dim_of_interest == 'depth':
        new_index = index_of_interest - start_depth
    elif dim_of_interest == 'rows':
        new_index = index_of_interest - start_row
    elif dim_of_interest == 'cols':
        new_index = index_of_interest - start_col
    
    return volume[:, start_depth:start_depth + new_depth, start_row:start_row + new_rows, start_col:start_col + new_cols], new_index

def downsample_volume_average(volume, new_shape):
    depth, rows, cols = volume.shape
    new_depth, new_rows, new_cols = new_shape
    
    if depth % new_depth != 0 or rows % new_rows != 0 or cols % new_cols != 0:
        raise ValueError("The new shape must be a divisor of the original shape.")
        
    depth_factor = depth // new_depth
    row_factor = rows // new_rows
    col_factor = cols // new_cols
    
    new_volume = np.zeros(new_shape)
    
    for i in range(new_depth):
        for j in range(new_rows):
            for k in range(new_cols):
                subarray = volume[i*depth_factor:(i+1)*depth_factor, 
                                  j*row_factor:(j+1)*row_factor, 
                                  k*col_factor:(k+1)*col_factor]
                
                new_volume[i, j, k] = np.mean(subarray)
                
    return new_volume

def downsample_volume_random(volume, new_shape):
    depth, rows, cols = volume.shape
    new_depth, new_rows, new_cols = new_shape

    if depth % new_depth != 0 or rows % new_rows != 0 or cols % new_cols != 0:
        raise ValueError("The new shape must be a divisor of the original shape.")

    depth_factor = depth // new_depth
    row_factor = rows // new_rows
    col_factor = cols // new_cols

    new_volume = np.zeros(new_shape)

    for i in range(new_depth):
        for j in range(new_rows):
            for k in range(new_cols):
                subarray = volume[i*depth_factor:(i+1)*depth_factor,
                                  j*row_factor:(j+1)*row_factor,
                                  k*col_factor:(k+1)*col_factor]
                
                # Select a random value from the subarray
                random_value = np.random.choice(subarray.flatten())
                
                new_volume[i, j, k] = random_value

    return new_volume

def plot_slices(volume, index_of_interest, figure_number):
    plt.figure(figure_number)
    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    axes = axes.ravel()
    for idx, slice_img in enumerate(volume[:, :, index_of_interest, :]):
        axes[idx].imshow(slice_img, cmap='gray', clim=(slice_img.min(), slice_img.max()))
        axes[idx].set_title(f"Slice {idx + 1}")
        axes[idx].axis('off')
    plt.tight_layout()

NUM_GATES = 12
CORONAL_SLICE_INDEX = 144

if len(sys.argv) != 4:
    print("Usage: python downsample.py original_file_path cached_downsampled_average_file_path cached_downsampled_random_file_path")
    sys.exit(1)

original_file_path = sys.argv[1]
cached_downsampled_average_file_path = sys.argv[2]
cached_downsampled_random_file_path = sys.argv[3]

original_shape = (NUM_GATES, 200, 380, 380)
cropped_shape = (NUM_GATES, 200, 200, 200)
cropped_downsampled_shape = (NUM_GATES, 100, 100, 100)

if os.path.exists(cached_downsampled_average_file_path):
    cropped_downsampled_volume_average = load_images(cached_downsampled_average_file_path, cropped_downsampled_shape)
else:
    original_volume = load_images(original_file_path, original_shape)
    cropped_volume, cropped_coronal_slice_index = crop_volume(original_volume, cropped_shape, CORONAL_SLICE_INDEX, 'rows')
    
    cropped_downsampled_volume_average = np.zeros(cropped_downsampled_shape)
    
    for t in range(cropped_volume.shape[0]):
        print(t)
        cropped_downsampled_volume_average[t, :, :, :] = downsample_volume_average(cropped_volume[t, :, :, :], cropped_downsampled_shape[1:])
        
    export_data(cropped_downsampled_volume_average, cached_downsampled_average_file_path)

if os.path.exists(cached_downsampled_random_file_path):
    cropped_downsampled_volume_random = load_images(cached_downsampled_random_file_path, cropped_downsampled_shape)
else:
    original_volume = load_images(original_file_path, original_shape)
    cropped_volume, cropped_coronal_slice_index = crop_volume(original_volume, cropped_shape, CORONAL_SLICE_INDEX, 'rows')
    
    cropped_downsampled_volume_random = np.zeros(cropped_downsampled_shape)
    
    for t in range(cropped_volume.shape[0]):
        print(t)
        cropped_downsampled_volume_random[t, :, :, :] = downsample_volume_random(cropped_volume[t, :, :, :], cropped_downsampled_shape[1:])
        
    export_data(cropped_downsampled_volume_random, cached_downsampled_random_file_path)


original_volume = load_images(original_file_path, original_shape)
cropped_volume, cropped_coronal_slice_index = crop_volume(original_volume, cropped_shape, CORONAL_SLICE_INDEX, 'rows')

cropped_downsampled_coronal_slice_index = calculate_downsampled_index_of_interest(cropped_shape[1:], cropped_downsampled_shape[1:], 'rows', cropped_coronal_slice_index)


plot_slices(original_volume, CORONAL_SLICE_INDEX, 1)
plot_slices(cropped_volume, cropped_coronal_slice_index, 2)
plot_slices(cropped_downsampled_volume_average, cropped_downsampled_coronal_slice_index, 3)
plot_slices(cropped_downsampled_volume_random, cropped_downsampled_coronal_slice_index, 4)

plt.show()
