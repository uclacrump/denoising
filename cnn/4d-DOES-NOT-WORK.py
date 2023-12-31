import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import TimeDistributed
import matplotlib.pyplot as plt

from misc import load_images, calculate_cropped_index_of_interest, calculate_downsampled_index_of_interest

# Load the data
cached_downsampled_average_file_path = "/Users/williamdriscoll/Dropbox/Projects/crump/data/old_amide_exports/amide_export_t58783_feldkamp_cropped_downsampled_average"
cached_downsampled_random_file_path = "/Users/williamdriscoll/Dropbox/Projects/crump/data/old_amide_exports/amide_export_t58783_feldkamp_cropped_downsampled_random"

downsampled_average_volume = load_images(cached_downsampled_average_file_path, (12, 100, 100, 100))
downsampled_random_volume = load_images(cached_downsampled_random_file_path, (12, 100, 100, 100))

min_value = min(np.min(downsampled_average_volume), np.min(downsampled_random_volume))
max_value = max(np.max(downsampled_average_volume), np.max(downsampled_random_volume))

normalized_average = (downsampled_average_volume - min_value) / (max_value - min_value)
normalized_random = (downsampled_random_volume - min_value) / (max_value - min_value)

normalized_average = np.expand_dims(normalized_average, axis=-1)
normalized_random = np.expand_dims(normalized_random, axis=-1)


# Build the model
model = models.Sequential([
    TimeDistributed(layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same'), input_shape=(12, 100, 100, 100, 1)),
    layers.BatchNormalization(),
    TimeDistributed(layers.MaxPooling3D((2, 2, 2), padding='same')),
    TimeDistributed(layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')),
    layers.BatchNormalization(),
    TimeDistributed(layers.MaxPooling3D((2, 2, 2), padding='same')),
    TimeDistributed(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')),
    layers.BatchNormalization(),
    TimeDistributed(layers.UpSampling3D((2, 2, 2))),
    TimeDistributed(layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')),
    layers.BatchNormalization(),
    TimeDistributed(layers.UpSampling3D((2, 2, 2))),
    TimeDistributed(layers.Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same'))
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(normalized_random, normalized_average, epochs=1, batch_size=1)

# Make a prediction
predicted_volume = model.predict(normalized_random)

# Get 3D volume from the predicted 4D volume
predicted_volume_0_index = predicted_volume[0]

NUM_GATES = 12
original_shape = (NUM_GATES, 200, 380, 380)
cropped_shape = (NUM_GATES, 200, 200, 200)
cropped_downsampled_shape = (NUM_GATES, 100, 100, 100)

cropped_coronal_slice_index = calculate_cropped_index_of_interest(144, 'rows', original_shape, cropped_shape)
cropped_downsampled_coronal_slice_index = calculate_downsampled_index_of_interest(cropped_shape[1:], cropped_downsampled_shape[1:], 'rows', cropped_coronal_slice_index)

coronal_slice_average_0_index = normalized_average[0, :, cropped_downsampled_coronal_slice_index, :, :]
coronal_slice_random_0_index = normalized_random[0, :, cropped_downsampled_coronal_slice_index, :, :]
denoised_image_slice = predicted_volume_0_index[:, cropped_downsampled_coronal_slice_index, :, :]

plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(coronal_slice_average_0_index[0, :, :, 0], cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Noisy')
plt.imshow(coronal_slice_random_0_index[0, :, :, 0], cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Denoised')
plt.imshow(denoised_image_slice[0, :, :, 0], cmap='gray')

plt.show()
