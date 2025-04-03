import cv2
import numpy as np

def extract_roi(image, x, y, w, h):
    return image[y:y+h, x:x+w]

def simple_threshold(image, threshold=127):
    _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def adaptive_threshold(image, block_size=11, C=2):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, C)

def otsu_threshold(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def show_and_save_image(image, window_name, file_name):
    cv2.imshow(window_name, image)
    cv2.imwrite(file_name, image)

def main():
    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    x, y, w, h = 50, 100, 550, 400  
    roi = extract_roi(image, x, y, w, h)

    thresh_simple = simple_threshold(roi)
    thresh_adaptive = adaptive_threshold(roi)
    thresh_otsu = otsu_threshold(roi)

    show_and_save_image(roi, 'Original ROI', 'roi_output.png')
    show_and_save_image(thresh_simple, 'Simple Threshold', 'simple_threshold.png')
    show_and_save_image(thresh_adaptive, 'Adaptive Threshold', 'adaptive_threshold.png')
    show_and_save_image(thresh_otsu, 'Otsu Threshold', 'otsu_threshold.png')
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

