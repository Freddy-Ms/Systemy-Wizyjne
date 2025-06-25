import cv2
import numpy as np

def white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    avg_a = np.mean(a)
    avg_b = np.mean(b)

    a = a - ((avg_a - 128) * (l / 255.0) * 1.1)
    b = b - ((avg_b - 128) * (l / 255.0) * 1.1)

    lab = cv2.merge([l, a.astype(np.uint8), b.astype(np.uint8)])
    balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return balanced

def main():
    image = cv2.imread("image.jpg")

    corrected = white_balance(image)

    cv2.imwrite("output_corrected.jpg", corrected)

    cv2.imshow("Original", image)
    cv2.imshow("Corrected White Balance", corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()