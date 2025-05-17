import cv2
import numpy as np
import os

IMAGE_NAME = "rynek_frag.jpg"
ROTATION_ANGLE = 44 
MAX_CORNERS = 2700
OUTPUT_REPORT = "report.txt"

def show_and_save_image(image, window_name, file_name):
    cv2.imshow(window_name, image)
    cv2.imwrite(file_name, image)

def detect_corners(gray_img):
    return cv2.goodFeaturesToTrack(gray_img, maxCorners=MAX_CORNERS, qualityLevel=0.01, minDistance=10)

def rotate_image(image, angle, center_point):
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center_point, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated, rotation_matrix

def transform_points(points, matrix):
    points = np.squeeze(points)
    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=0)
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = np.dot(matrix, homogeneous.T).T
    return transformed

def compare_points(original, transformed, threshold=5):
    match_count = 0
    for p1 in original:
        for p2 in transformed:
            distance = np.linalg.norm(p1 - p2)
            if distance < threshold:
                match_count += 1
                break
    return match_count

def save_report(num_before, num_after, matched, angle):
    with open(OUTPUT_REPORT, "w") as f:
        f.write("REPORT: Corner Comparison Before and After Rotation\n")
        f.write(f"\nRotation angle: {angle} degrees")
        f.write(f"\nCorners before rotation: {num_before}")
        f.write(f"\nCorners after rotation: {num_after}")
        f.write(f"\nMatched corners (after reverse transformation): {matched}")
        f.write(f"\nPercentage preserved: {matched / num_before * 100:.2f}%\n")

def main():
    image = cv2.imread(IMAGE_NAME)
    if image is None:
        print("Error: image not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners_before = detect_corners(gray)
    print(f"Found {len(corners_before)} corners before rotation.")

    reference_point = (image.shape[1] // 4, image.shape[0] // 2)
    rotated_image, rot_matrix = rotate_image(image, ROTATION_ANGLE, reference_point)
    gray_rotated = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    corners_after = detect_corners(gray_rotated)
    print(f"Found {len(corners_after)} corners after rotation.")

    inverse_matrix = cv2.invertAffineTransform(rot_matrix)
    corners_after_inverse = transform_points(corners_after, inverse_matrix)

    matched = compare_points(np.squeeze(corners_before), corners_after_inverse)
    print(f"Matched corners: {matched} / {len(corners_before)}")

    save_report(len(corners_before), len(corners_after), matched, ROTATION_ANGLE)

    original_with_corners = image.copy()
    for corner in corners_before:
        x, y = corner.ravel()
        cv2.circle(original_with_corners, (int(x), int(y)), 3, (0, 255, 0), -1)
    show_and_save_image(original_with_corners, "Original Image with Corners", "original_with_corners.jpg")

    rotated_with_corners = rotated_image.copy()
    for corner in corners_after:
        x, y = corner.ravel()
        cv2.circle(rotated_with_corners, (int(x), int(y)), 3, (0, 255, 0), -1)
    show_and_save_image(rotated_with_corners, "Rotated Image with Corners", "rotated_with_corners.jpg")


if __name__ == "__main__":
    main()
