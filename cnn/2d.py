import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from misc import load_images, calculate_cropped_index_of_interest, calculate_downsampled_index_of_interest
import matplotlib.pyplot as plt

def train_model(average_dir, random_dir):
    average_files = sorted(os.listdir(average_dir))
    random_files = sorted(os.listdir(random_dir))

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(100, 100, 1)),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # first_pair = True
    test_pair = None

    for avg_file, rnd_file in zip(average_files, random_files):
        if avg_file == "t58783":
            test_pair = (avg_file, rnd_file)
            # first_pair = False
            continue

        print(avg_file)

        average_image_path = os.path.join(average_dir, avg_file)
        random_image_path = os.path.join(random_dir, rnd_file)

        average_volume = load_images(average_image_path, (12, 100, 100, 100))
        random_volume = load_images(random_image_path, (12, 100, 100, 100))

        cropped_coronal_slice_index = calculate_cropped_index_of_interest(144, 'rows', (12, 200, 380, 380), (12, 200, 200, 200))
        cropped_downsampled_coronal_slice_index = calculate_downsampled_index_of_interest((200, 200, 200), (100, 100, 100), 'rows', cropped_coronal_slice_index)

        coronal_slice_average = average_volume[:, :, cropped_downsampled_coronal_slice_index, :]
        coronal_slice_random = random_volume[:, :, cropped_downsampled_coronal_slice_index, :]

        coronal_slice_average_0_index = coronal_slice_average[0]
        coronal_slice_random_0_index = coronal_slice_random[0]

        coronal_slice_average_0_index = coronal_slice_average_0_index.reshape((1, 100, 100, 1)).astype('float32')
        coronal_slice_random_0_index = coronal_slice_random_0_index.reshape((1, 100, 100, 1)).astype('float32')

        max_value = max(coronal_slice_average_0_index.max(), coronal_slice_random_0_index.max())
        min_value = min(coronal_slice_average_0_index.min(), coronal_slice_random_0_index.min())

        coronal_slice_average_0_index = (coronal_slice_average_0_index - min_value) / (max_value - min_value)
        coronal_slice_random_0_index = (coronal_slice_random_0_index - min_value) / (max_value - min_value)

        model.fit(coronal_slice_random_0_index, coronal_slice_average_0_index, epochs=1000, batch_size=1, verbose=1)

    print("Model has been trained!")

    # Testing on the first pair
    avg_file, rnd_file = test_pair
    average_image_path = os.path.join(average_dir, avg_file)
    random_image_path = os.path.join(random_dir, rnd_file)

    average_volume = load_images(average_image_path, (12, 100, 100, 100))
    random_volume = load_images(random_image_path, (12, 100, 100, 100))

    cropped_coronal_slice_index = calculate_cropped_index_of_interest(144, 'rows', (12, 200, 380, 380), (12, 200, 200, 200))
    cropped_downsampled_coronal_slice_index = calculate_downsampled_index_of_interest((200, 200, 200), (100, 100, 100), 'rows', cropped_coronal_slice_index)

    coronal_slice_average = average_volume[:, :, cropped_downsampled_coronal_slice_index, :]
    coronal_slice_random = random_volume[:, :, cropped_downsampled_coronal_slice_index, :]

    coronal_slice_average_0_index = coronal_slice_average[0]
    coronal_slice_random_0_index = coronal_slice_random[0]

    coronal_slice_average_0_index = coronal_slice_average_0_index.reshape((1, 100, 100, 1)).astype('float32')
    coronal_slice_random_0_index = coronal_slice_random_0_index.reshape((1, 100, 100, 1)).astype('float32')

    max_value = max(coronal_slice_average_0_index.max(), coronal_slice_random_0_index.max())
    min_value = min(coronal_slice_average_0_index.min(), coronal_slice_random_0_index.min())

    coronal_slice_average_0_index = (coronal_slice_average_0_index - min_value) / (max_value - min_value)
    coronal_slice_random_0_index = (coronal_slice_random_0_index - min_value) / (max_value - min_value)

    print("Generating plots for test pair...")
    denoised_image = model.predict(coronal_slice_random_0_index)

    plt.figure(figsize=(10, 10))

    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(coronal_slice_average_0_index[0, :, :, 0], cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Noisy')
    plt.imshow(coronal_slice_random_0_index[0, :, :, 0], cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Denoised')
    plt.imshow(denoised_image[0, :, :, 0], cmap='gray')

    plt.show()
    print("All done!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_downsampled_average_dir> <path_to_downsampled_random_dir>")
        sys.exit(1)

    average_dir = sys.argv[1]
    random_dir = sys.argv[2]

    train_model(average_dir, random_dir)
