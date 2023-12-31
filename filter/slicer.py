import numpy as np
import matplotlib.pyplot as plt
import imageio
import pdb

def import_data(file_path):
    with open(file_path, 'rb') as file:
        return np.fromfile(file, dtype=np.float32)

file_path = "amide_export_t58783_feldkamp"
data = import_data(file_path)
vol = data.reshape((-1, 200, 380, 380))
print(vol.shape)

slices = vol[:, :, 144, :]

print(slices.shape)

# Adjust the slices for better visibility in the video
slices_adjusted = (slices - slices.min()) / (slices.max() - slices.min()) * 255
slices_adjusted = slices_adjusted.astype(np.uint8)

# # Display each adjusted 2D slice in a single window
# num_slices = slices.shape[0]

# Let's make a 3x4 grid to display the 12 slices (assuming there are 12 slices; adjust grid if different)
fig, axes = plt.subplots(3, 4, figsize=(13, 8))
axes = axes.ravel()  # Flatten axes to make indexing easier

for idx, slice_img in enumerate(slices):
    axes[idx].imshow(slice_img, cmap='gray', clim=(slice_img.min(), slice_img.max()))
    axes[idx].set_title(f"Slice {idx + 1}")
    axes[idx].axis('off')  # Turn off axis labels for cleaner visualization

plt.tight_layout()
plt.show()

# Save slices to an MP4 video with loops
fps = 10  # Adjust for desired frames per second
video_filename = 'output_video.mp4'
num_loops = 5  # Specify how many times you want the sequence to loop in the video

# Create a repeated sequence of slices for the desired number of loops
slices_loop = np.tile(slices_adjusted, (num_loops, 1, 1))

imageio.mimsave(video_filename, slices_loop, fps=fps)

print(f"Video saved as {video_filename}")
