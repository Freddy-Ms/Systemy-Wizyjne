import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, image_bgr

def convert_to_hsv(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return h, s, v

def create_mask(h_channel, lower_hue, upper_hue):
    mask = cv2.inRange(h_channel, lower_hue, upper_hue)
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    return mask_clean

def apply_mask(image_rgb, mask):
    return cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

def show_results(image_rgb, h, mask, result):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title('Obraz RGB')
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Kanał H')
    plt.imshow(h, cmap='hsv')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('Maska binarna')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Wyizolowany kwiat')
    plt.imshow(result)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main(image_path, lower_hue=150, upper_hue=180):
    image_rgb, image_bgr = load_image(image_path)
    h, s, v = convert_to_hsv(image_bgr)
    mask = create_mask(h, lower_hue, upper_hue)
    result = apply_mask(image_rgb, mask)
    show_results(image_rgb, h, mask, result)


if __name__ == "__main__":
    main("image.jpg")  
