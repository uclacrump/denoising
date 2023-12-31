import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import os
import argparse

from misc import load_images, calculate_cropped_index_of_interest, calculate_downsampled_index_of_interest

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
        # if noisy_file != "t58783":
        #     continue

        print(noisy_file)

        noisy_file_path = os.path.join(noisy_downsampled_dir, noisy_file)
        clean_file_path = os.path.join(clean_downsampled_dir, clean_file)

        downsampled_average_volume = load_images(clean_file_path, (12, 100, 100, 100))
        downsampled_random_volume = load_images(noisy_file_path, (12, 100, 100, 100))

        normalized_averages = []
        normalized_randoms = []

        for j in range(12):
            min_value = min(np.min(downsampled_average_volume[j]), np.min(downsampled_random_volume[j]))
            max_value = max(np.max(downsampled_average_volume[j]), np.max(downsampled_random_volume[j]))

            normalized_average = (downsampled_average_volume[j] - min_value) / (max_value - min_value)
            normalized_random = (downsampled_random_volume[j] - min_value) / (max_value - min_value)

            normalized_averages.append(np.expand_dims(np.expand_dims(normalized_average, axis=0), axis=-1))
            normalized_randoms.append(np.expand_dims(np.expand_dims(normalized_random, axis=0), axis=-1))

        for j, (normalized_random, normalized_average) in enumerate(zip(normalized_randoms, normalized_averages)):
            print(j)
            # early_stopping = callbacks.EarlyStopping(monitor='loss', min_delta=0.000005, patience=5, restore_best_weights=True)

            stop_at_threshold_callback = StopAtThresholdCallback(threshold=0.00009)
            model.fit(normalized_random, normalized_average, epochs=1, batch_size=1, callbacks=[stop_at_threshold_callback])

            print('downsampled value dimensions expanded dim')
            print(normalized_random.shape)
            print(normalized_average.shape)
            # model.fit(normalized_random, normalized_average, epochs=5000, batch_size=1)

        if i == len(noisy_files) - 1:
        # if noisy_file == "t58783":
            print('shape of what we are predicting')
            print(normalized_randoms[0].shape)
            predicted_volume = model.predict(normalized_randoms[0])
            plot_results(normalized_averages[0], normalized_randoms[0], predicted_volume)


def plot_results(normalized_average, normalized_random, predicted_volume):
    print ('plot results')
    print(normalized_average.shape)
    print(normalized_random.shape)
    print(predicted_volume.shape)
    NUM_GATES = 12
    original_shape = (NUM_GATES, 200, 380, 380)
    cropped_shape = (NUM_GATES, 200, 200, 200)
    cropped_downsampled_shape = (NUM_GATES, 100, 100, 100)

    cropped_coronal_slice_index = calculate_cropped_index_of_interest(144, 'rows', original_shape, cropped_shape)
    cropped_downsampled_coronal_slice_index = calculate_downsampled_index_of_interest(cropped_shape[1:], cropped_downsampled_shape[1:], 'rows', cropped_coronal_slice_index)

    coronal_slice_average_0_index = normalized_average[:, :, cropped_downsampled_coronal_slice_index, :, :]
    coronal_slice_random_0_index = normalized_random[:, :, cropped_downsampled_coronal_slice_index, :, :]
    denoised_image_slice = predicted_volume[:, :, cropped_downsampled_coronal_slice_index, :, :]

    print ('goddamn it')
    print(np.all(coronal_slice_average_0_index == normalized_average[:, :, cropped_downsampled_coronal_slice_index, :, :]))
    print(coronal_slice_average_0_index.shape)
    print((normalized_average[:, :, cropped_downsampled_coronal_slice_index, :, :]).shape)

    print('0 indeices shapes')
    print(coronal_slice_random_0_index.shape)

    print('plottable shapes')
    print(coronal_slice_average_0_index[0, :, :, 0].shape)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with noisy and clean directories.')
    parser.add_argument('noisy_downsampled_dir', type=str, help='Path to the directory containing noisy downsampled volumes.')
    parser.add_argument('clean_downsampled_dir', type=str, help='Path to the directory containing clean downsampled volumes.')

    args = parser.parse_args()

    noisy_downsampled_dir = args.noisy_downsampled_dir
    clean_downsampled_dir = args.clean_downsampled_dir

    model = models.Sequential([
        layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same', input_shape=(100, 100, 100, 1)),
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

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    train_model_on_directory(noisy_downsampled_dir, clean_downsampled_dir)
