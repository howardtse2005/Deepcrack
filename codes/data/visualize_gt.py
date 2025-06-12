import cv2
import numpy as np
mask = cv2.imread('/home/fyp/DeepCrack/codes/data/Crack/test/gt_fixed/9_GT.png', cv2.IMREAD_GRAYSCALE)
print(np.unique(mask))