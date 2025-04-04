import cv2
import numpy as np

def apply_contouring(image):
    return cv2.Canny(image, 100, 200)

def apply_threshold(image):
     return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def remove_artifacts(image):
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def show_and_save_image(image, window_name, file_name):
    cv2.imshow(window_name, image)
    cv2.imwrite(file_name, image)


def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Błąd: Nie można wczytać obrazu {image_path}")
        return
    
    contoured = apply_contouring(image)
    thresholded = apply_threshold(image)
    art_image = remove_artifacts(image)
    
    show_and_save_image(contoured, 'Contoured Image', 'contoured.png')
    show_and_save_image(thresholded, 'Thresholded Image', 'thresholded.png')
    show_and_save_image(art_image, 'Artifact Removed Image', 'artifact_removed.png')
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_image('image.JPG')  
