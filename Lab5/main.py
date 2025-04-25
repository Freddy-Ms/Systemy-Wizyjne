import cv2
import numpy as np


def show_and_save_image(image, window_name, file_name):
    cv2.imshow(window_name, image)
    cv2.imwrite(file_name, image)

def load_and_gray(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def blur_image(gray):
    return cv2.GaussianBlur(gray, (51, 51), 0)

def threshold_image(blurred):
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_edges(thresh):
    return cv2.Canny(thresh, 50, 150, apertureSize=3)

def detect_lines(edges):
    return cv2.HoughLines(edges, 1, np.pi / 180, 200)

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

def compute_average_angle(lines):
    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.rad2deg(theta)
            if angle > 90:
                angle -= 180
            angles.append(angle)
        return np.mean(angles)
    return 0

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def process_image(image_path):
    image, gray = load_and_gray(image_path)
    blurred = blur_image(gray)
    thresh = threshold_image(blurred)
    edges = detect_edges(thresh)
    lines = detect_lines(edges)
    lines_image = draw_lines(image, lines)
    avg_angle = compute_average_angle(lines)
    print(f"Kąt rotacji: {avg_angle:.2f} stopni")
    rotated = rotate_image(image, -avg_angle)
    return image, rotated, thresh, edges, lines_image


if __name__ == "__main__":
    original, rotated, thresholded, edge_map, lines_image = process_image("image.jpg")

    show_and_save_image(original, "Oryginalny", "original.jpg")
    show_and_save_image(rotated, "Wyrównany", "rotated.jpg")
    show_and_save_image(thresholded, "Progowanie", "thresholded.jpg")
    show_and_save_image(edge_map, "Krawędzie", "edges.jpg")
    show_and_save_image(lines_image, "Wykryte linie", "lines.jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
