import pdb
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image
from datetime import datetime

from misc import load_images, calculate_cropped_index_of_interest, calculate_downsampled_index_of_interest

def get_current_moment():
    current_moment = datetime.now()
    formatted_moment = current_moment.strftime("%Y-%m-%d-%H-%M")
    return formatted_moment

class StopAtThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(StopAtThresholdCallback, self).__init__()
        self.threshold = threshold
        self.num_times_below_threshold = 0
    
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        if loss is not None and loss < self.threshold:
            self.num_times_below_threshold += 1
            if self.num_times_below_threshold > 5:
                self.model.stop_training = True
                print(f"Training stopped as loss {loss} is less than {self.threshold}")

def train_model_on_directory(noisy_downsampled_dir, clean_downsampled_dir):
    noisy_files = sorted(os.listdir(noisy_downsampled_dir))
    clean_files = sorted(os.listdir(clean_downsampled_dir))

    for i, (noisy_file, clean_file) in enumerate(zip(noisy_files, clean_files)):
        print("Training model on file", noisy_file)

        noisy_file_path = os.path.join(noisy_downsampled_dir, noisy_file)
        clean_file_path = os.path.join(clean_downsampled_dir, clean_file)

        downsampled_average_volume = load_images(clean_file_path, (12, 100, 100, 100))
        downsampled_random_volume = load_images(noisy_file_path, (12, 100, 100, 100))

        downsampled_average_volume = np.transpose(downsampled_average_volume, (2, 0, 1, 3))
        downsampled_random_volume = np.transpose(downsampled_random_volume, (2, 0, 1, 3))

        z_depth = downsampled_average_volume.shape[0]

        normalized_averages = []
        normalized_randoms = []

        for z in range(z_depth):
            min_value = min(np.min(downsampled_average_volume[z]), np.min(downsampled_random_volume[z]))
            max_value = max(np.max(downsampled_average_volume[z]), np.max(downsampled_random_volume[z]))


            normalized_average = (downsampled_average_volume[z] - min_value) / (max_value - min_value)
            normalized_random = (downsampled_random_volume[z] - min_value) / (max_value - min_value)
            
            normalized_averages.append(np.expand_dims(np.expand_dims(normalized_average, axis=0), axis=-1))
            normalized_randoms.append(np.expand_dims(np.expand_dims(normalized_random, axis=0), axis=-1))

        for z, (normalized_random, normalized_average) in enumerate(zip(normalized_randoms, normalized_averages)):
            model.fit(normalized_random, normalized_average, epochs=1, batch_size=1)
            
            # This demonstrates how to use a threshold callback. This is not necessary to use, but it can help cut down on unneeded epochs.

            #stop_at_threshold_callback = StopAtThresholdCallback(threshold=0.00009)
            # model.fit(normalized_random, normalized_average, epochs=175, batch_size=1, callbacks=[stop_at_threshold_callback])

        # We test denoise one of the images. Yes, this is one of the images we trained on. So we are denoising an image that is in the training data.
        # This is bad, I know. But you can fix it!

        if i == len(noisy_files) - 1:
            NUM_GATES = 12
            original_shape = (NUM_GATES, 200, 380, 380)
            cropped_shape = (NUM_GATES, 200, 200, 200)
            cropped_downsampled_shape = (NUM_GATES, 100, 100, 100)

            cropped_coronal_slice_index = calculate_cropped_index_of_interest(144, 'rows', original_shape, cropped_shape)
            z = calculate_downsampled_index_of_interest(cropped_shape[1:], cropped_downsampled_shape[1:], 'rows', cropped_coronal_slice_index)

            print('Denoising...')
            
            predicted_volume = model.predict(normalized_randoms[z])

            plot_results(normalized_averages[z], normalized_randoms[z], predicted_volume, noisy_file)


def plot_results(original_clean_volume, noisy_volume, denoised_volume, src_file_name):
    plt.figure(figsize=(10, 10))

    clean_volume_plottable_img = original_clean_volume[0, 0, :, :, 0]
    noisy_volume_plottable_img = noisy_volume[0, 0, :, :, 0]
    denoised_volume_plottable_img = denoised_volume[0, 0, :, :, 0]

    # Normalize images to [0, 255] and convert to uint8 for saving as PNG
    def normalize_and_save(img, path):
        img = np.squeeze(img)  # Remove singleton dimensions if any
        img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255.0
        img = img.astype(np.uint8)
        Image.fromarray(img, 'L').save(path)

    folder_path = 'data/runs/3d-xytime/' + get_current_moment() + '/' + src_file_name + '/'
    os.makedirs(folder_path, exist_ok=True)
    
    clean_image_path = os.path.join(folder_path, 'clean.png')
    noisy_image_path = os.path.join(folder_path, 'noisy.png')
    denoised_image_path = os.path.join(folder_path, 'denoised.png')

    normalize_and_save(clean_volume_plottable_img, clean_image_path)
    normalize_and_save(noisy_volume_plottable_img, noisy_image_path)
    normalize_and_save(denoised_volume_plottable_img, denoised_image_path)

    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(clean_volume_plottable_img, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Noisy')
    plt.imshow(noisy_volume_plottable_img, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Denoised')
    plt.imshow(denoised_volume_plottable_img, cmap='gray')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with noisy and clean directories.')
    parser.add_argument('noisy_downsampled_dir', type=str, help='Path to the directory containing noisy downsampled volumes.')
    parser.add_argument('clean_downsampled_dir', type=str, help='Path to the directory containing clean downsampled volumes.')

    args = parser.parse_args()

    noisy_downsampled_dir = args.noisy_downsampled_dir
    clean_downsampled_dir = args.clean_downsampled_dir

    model = models.Sequential([
        layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same', input_shape=(12, 100, 100, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2), padding='same'),
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2), padding='same'),
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.UpSampling3D((2, 2, 2)),
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.UpSampling3D((2, 2, 2)),
        layers.Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')
    ])

    model.summary()

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    train_model_on_directory(noisy_downsampled_dir, clean_downsampled_dir)
