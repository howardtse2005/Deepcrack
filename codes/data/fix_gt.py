import cv2
import os
import numpy as np

input_dir = '/home/fyp/DeepCrack/codes/data/NonCrack/test/gt'
output_dir = '/home/fyp/DeepCrack/codes/data/NonCrack/test/gt_fixed'
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(input_dir, fname)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Binarize: threshold at 127
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(output_dir, fname), mask_bin)
        print(f"Binarized {fname}")