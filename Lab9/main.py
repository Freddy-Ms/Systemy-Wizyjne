import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_convert_to_gray(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return gray

def compute_histogram(gray_image):
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    return hist

def equalize_histogram(gray_image):
    return cv2.equalizeHist(gray_image)

def save_image(path, image):
    cv2.imwrite(path, image)

def plot_2x2_matrix(original, original_hist, equalized, equalized_hist, save_path=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].plot(original_hist, color='black')
    axs[0, 1].set_title('Original Histogram')

    axs[1, 0].imshow(equalized, cmap='gray')
    axs[1, 0].set_title('Equalized Image')
    axs[1, 0].axis('off')

    axs[1, 1].plot(equalized_hist, color='black')
    axs[1, 1].set_title('Equalized Histogram')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def process_image(input_path, output_image_path, output_plot_path):
    gray = load_and_convert_to_gray(input_path)
    gray_hist = compute_histogram(gray)

    equalized = equalize_histogram(gray)
    equalized_hist = compute_histogram(equalized)

    save_image(output_image_path, equalized)
    plot_2x2_matrix(gray, gray_hist, equalized, equalized_hist, save_path=output_plot_path)


if __name__ == "__main__":
    process_image('image.jpg', 'output_equalized.jpg', 'histograms_and_images.png')
