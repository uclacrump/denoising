import cv2
import numpy as np
import argparse

def calculate_mse(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must be of the same shape")
    
    mse = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    mse /= float(image1.shape[0] * image1.shape[1])
    
    return mse

def noise_score(image, ground_truth):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    if len(ground_truth.shape) == 3:
        gray_ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    else:
        gray_ground_truth = ground_truth
    
    mse = calculate_mse(gray_image, gray_ground_truth)
    
    return mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate noise scores for images.")
    parser.add_argument("clean_image", help="Path to the clean image (ground truth)")
    parser.add_argument("noisy_image", help="Path to the noisy image")
    parser.add_argument("denoised_image", help="Path to the denoised image")

    args = parser.parse_args()

    # Read in the images
    clean_image = cv2.imread(args.clean_image, cv2.IMREAD_GRAYSCALE)
    noisy_image = cv2.imread(args.noisy_image, cv2.IMREAD_GRAYSCALE)
    denoised_image = cv2.imread(args.denoised_image, cv2.IMREAD_GRAYSCALE)
    
    # Calculate the noise scores
    noise_score_noisy = noise_score(noisy_image, clean_image)
    noise_score_denoised = noise_score(denoised_image, clean_image)

    print(f"Noise Score for Noisy Image: {noise_score_noisy}")
    print(f"Noise Score for Denoised Image: {noise_score_denoised}")
