import cv2
import numpy as np

def load_and_gray(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def blur_image(gray):
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)
    return blurred

def threshold_image(blurred):
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_edges(thresh):
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    return edges

def compute_rotation_angle(edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
    if lines is None:
        return 0

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)
    
    median_angle = np.median(angles)
    return median_angle

def draw_lines(image, lines):
    image_with_lines = image.copy()
    height, width = image.shape[:2]
    length = int(np.hypot(width, height)) 

    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + length * -b), int(y0 + length * a))
            pt2 = (int(x0 - length * -b), int(y0 - length * a))
            cv2.line(image_with_lines, pt1, pt2, (0, 0, 255), 2)
    return image_with_lines


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def find_largest_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def warp_perspective(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image[y:y+h, x:x+w]
    return cropped

def show_and_save_image(image, window_name, file_name):
    cv2.imshow(window_name, image)
    cv2.imwrite(file_name, image)

def process_image(image_path):
    image, gray = load_and_gray(image_path)
    blurred = blur_image(gray)
    thresh = threshold_image(blurred)
    edges = detect_edges(thresh)
    
    angle = compute_rotation_angle(edges)
    rotated_image = rotate_image(image, -angle)

    rotated_gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    rotated_blurred = blur_image(rotated_gray)
    rotated_thresh = threshold_image(rotated_blurred)
    rotated_edges = detect_edges(rotated_thresh)

    largest_contour = find_largest_contour(rotated_thresh)
    if largest_contour is None:
        raise ValueError("Nie znaleziono konturu kartki.")

    cropped = warp_perspective(rotated_image, largest_contour)

    return image, rotated_image, cropped, thresh, edges


if __name__ == "__main__":
    original, rotated, cropped, thresholded, edge_map = process_image("image.jpg")

    show_and_save_image(thresholded, "Progowanie", "progowanie.jpg")
    show_and_save_image(edge_map, "Krawędzie", "krawedzie.jpg")
    show_and_save_image(rotated, "Obrócony obraz", "obrocona.jpg")
    show_and_save_image(cropped, "Wykadrowana kartka", "kartka_wyprostowana.jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
