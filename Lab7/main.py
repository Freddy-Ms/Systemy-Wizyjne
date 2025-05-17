import os
import cv2
import numpy as np
import imageio
from pathlib import Path

INPUT_FOLDER = 'images'       
OUTPUT_FOLDER = 'output'      
GIF_PATH = f'{OUTPUT_FOLDER}/animation15FPS.gif'
REPORT_PATH = f'{OUTPUT_FOLDER}/report.txt'
GIF_FPS = 15

def load_images_from_folder(folder):
    image_paths = sorted(Path(folder).glob("*.jpg"))
    images = [cv2.imread(str(p)) for p in image_paths]
    return images, image_paths

def detect_laser_positions(images):
    positions = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(thresh)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = 0, 0
        positions.append((cx, cy))
    return positions


def create_gif(images, positions, gif_path, fps):
    gif_frames = []
    for img, pos in zip(images, positions):
        frame = img.copy()
        cv2.circle(frame, pos, 10, (0, 255, 0), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gif_frames.append(frame_rgb)
    imageio.mimsave(gif_path, gif_frames, fps=fps)

def save_report(positions, report_path):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Laser Pointer Tracking Report (Frame by Frame)\n")
        f.write("Frame\tX\tY\tdX\tdY\n")

        prev_x, prev_y = positions[0]
        f.write(f"0\t{prev_x}\t{prev_y}\t0\t0\n") 
        for i in range(1, len(positions)):
            x, y = positions[i]
            dx = x - prev_x
            dy = y - prev_y
            f.write(f"{i}\t{x}\t{y}\t{dx}\t{dy}\n")
            prev_x, prev_y = x, y


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    images, _ = load_images_from_folder(INPUT_FOLDER)
    positions = detect_laser_positions(images)
    create_gif(images, positions, GIF_PATH, GIF_FPS)
    save_report(positions, REPORT_PATH)
    print(f"GIF saved to: {GIF_PATH}")
    print(f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
