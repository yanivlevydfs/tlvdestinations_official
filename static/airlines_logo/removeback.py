import os
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = r"C:\Users\YanivLevy\Documents\GitHub\tlvdestinations_official\static\airlines_logo\airline_logos"
MAX_WORKERS = min(8, os.cpu_count() or 4)

def remove_bg_opencv(path: str):
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return f"✖ cannot read: {path}"

        # Handle grayscale / RGB / RGBA safely
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.shape[2] == 4:
            bgr = img[:, :, :3]
        else:
            bgr = img

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold (robust to off-white backgrounds)
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5
        )

        # Clean edges
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # Create RGBA
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = mask

        # Overwrite same file (lossless WebP)
        Image.fromarray(rgba).save(path, "WEBP", lossless=True)

        return f"✔ processed: {path}"

    except Exception as e:
        return f"✖ failed: {path} → {e}"

def collect_files(root: str):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".webp"):
                files.append(os.path.join(dirpath, name))
    return files

if __name__ == "__main__":
    files = collect_files(ROOT_DIR)
    print(f"Found {len(files)} .webp files")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(remove_bg_opencv, f) for f in files]

        for future in as_completed(futures):
            print(future.result())
