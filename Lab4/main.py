import cv2
import numpy as np

def apply_contouring(image):
    return cv2.Canny(image, 100, 200)

def apply_threshold(image):
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return thresh

def remove_artifacts(image):
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def resize_image(image, max_size=800):
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Błąd: Nie można wczytać obrazu {image_path}")
        return
    
    contoured = apply_contouring(image)
    thresholded = apply_threshold(image)
    final_image = remove_artifacts(image)
    
    contoured_resized = resize_image(contoured)
    thresholded_resized = resize_image(thresholded)
    final_image_resized = resize_image(final_image)
    
    cv2.imwrite('contoured.png', contoured)
    cv2.imwrite('thresholded.png', thresholded)
    cv2.imwrite('final_processed_image.png', final_image)
    
    cv2.imshow('Contoured Image', contoured_resized)
    cv2.imshow('Thresholded Image', thresholded_resized)
    cv2.imshow('Final Processed Image', final_image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


process_image('image.JPG')  
