import cv2
import numpy as np
import sys 
import os

def load_image(image="image.jpg"):
    if image == "image.jpg":
        print("No image path provided, using default image")
    if not os.path.exists(image):
        print(f"Error: File {image} not found")
        sys.exit(1)
    return cv2.imread(image)

def show_image(title,image):
    cv2.imshow(title, image)

def create_matrix(n):
    if n<0:
        print("Invalid input")
        return
    size = 2 * n + 1
    print(f"Size of matrix: {size}x{size}, enter the values row-wise")
    matrix = []
    for i in range(size):
        while True:
            try:
                row = list(map(int, input().split()))
                if len(row) != size:
                    print("Invalid row size, enter again")
                    continue
                matrix.append(row)
                break
            except:
                print("Invalid input, enter again")
    return np.array(matrix, dtype=np.float32)

def apply_convolution(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def main():
    if len(sys.argv) > 1:
        image = load_image(sys.argv[1])
    else:
        image = load_image()
    show_image("Preprocessed", image)
    while True:
        n = int(input("Enter the value of n - positive number: "))
        if n>0:
            break
    kernel = create_matrix(n)
    post_image = apply_convolution(image, kernel)
    show_image("Postprocessed", post_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output.jpg", post_image)

if __name__ == "__main__":
    main()
    #To run this code, use the following command:
    # python main.py image.jpg
    # where image.jpg is the path to the image file