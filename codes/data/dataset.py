import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import random


def readIndex(index_path, shuffle=False):
    img_list = []
    with open(index_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    if shuffle is True:
        random.shuffle(img_list)
    return img_list

class dataReadPip(object):

    def __init__(self, crop=True, transforms=None, target_size=(448, 448), min_size = 448, crop_size=(448, 448), num_crops=40, num_crops_with_cracks=20):
        self.transforms = transforms
        self.crop = crop
        self.target_size = target_size
        self.min_size = min_size
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.num_crops_with_cracks = num_crops_with_cracks

    def __call__(self, item):
        img = cv2.imread(item[0])
        lab = cv2.imread(item[1])

        if len(lab.shape) != 2:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        # Ensure mask dimensions match image dimensions
        img_h, img_w = img.shape[:2]
        mask_h, mask_w = lab.shape[:2]
        
        # Check if dimensions are rotated relative to each other (width/height swapped)
        rotated = False
        if img_h != mask_h or img_w != mask_w:
            # Check if aspect ratios are inverted (potential 90-degree rotation)
            img_ratio = img_w / img_h
            mask_ratio = mask_w / mask_h
            
            # If image is wide but mask is tall or vice versa
            if (img_ratio > 1 and mask_ratio < 1) or (img_ratio < 1 and mask_ratio > 1):
                print(f"Detected potential 90-degree rotation. Img ratio: {img_ratio:.2f}, Mask ratio: {mask_ratio:.2f}")
                # Rotate image 90 degrees to match mask orientation
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                img_h, img_w = img.shape[:2]  # Update dimensions after rotation
                rotated = True
                print(f"Image rotated. New dimensions: {img_w}x{img_h}")
        
        # After potential rotation correction, resize mask if needed
        if img_h != mask_h or img_w != mask_w:
            print(f"Warning: Image dimensions ({img_w}x{img_h}) don't match mask dimensions ({mask_w}x{mask_h}). Resizing mask to match.")
            lab = cv2.resize(lab, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        if not self.crop:
            ### Pad the image to become square instead of performing random cropping
            # Add padding to make the image square
            h, w = img.shape[:2]
            if h != w:
                max_dim = max(h, w)
                # Create square canvas with padding
                img_padded = np.zeros((max_dim, max_dim, 3), dtype=img.dtype)
                mask_padded = np.zeros((max_dim, max_dim), dtype=lab.dtype)
                pad_h = (max_dim - h) // 2
                pad_w = (max_dim - w) // 2
                img_padded[pad_h:pad_h+h, pad_w:pad_w+w, :] = img
                mask_padded[pad_h:pad_h+h, pad_w:pad_w+w] = lab
                img, lab = img_padded, mask_padded
                
            # Resize images and labels to the target size
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
            lab = cv2.resize(lab, self.target_size, interpolation=cv2.INTER_NEAREST)

            if self.transforms is not None:
                img, lab = self.transforms(img, lab)

            img = _preprocess_img(img)
            lab = _preprocess_lab(lab)
            return [img, lab]
        
        ### If self.crop is True, perform random cropping 
        # Resize if either dimension is less than crop_size, maintaining aspect ratio
        h, w = img.shape[:2]
        crop_small = False
        if h < self.min_size or w < self.min_size:
            if h < w:
                scale = self.min_size / h
            else:
                scale = self.min_size / w
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            lab = cv2.resize(lab, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            print(f"Resized image dimensions: {img.shape[:2]}")
            print(f"Resized mask dimensions: {lab.shape[:2]}")
            crop_small = True
        elif h == self.min_size or w == self.min_size:
            crop_small = True

        # Perform random cropping
        crops_with_cracks = []
        crops_without_cracks = []
        max_attempts = 1000
        attempts = 0

        if crop_small:
            # Only crop one image if either dimension was less than 448
            all_crops = [(img[:self.crop_size[0], :self.crop_size[1]], lab[:self.crop_size[0], :self.crop_size[1]])]
        
        else:
            # Perform random cropping with filtering and fallback after max_attempts 
            while (len(crops_with_cracks) < self.num_crops_with_cracks or len(crops_without_cracks) < (self.num_crops - self.num_crops_with_cracks)) and attempts < max_attempts:
                h, w = img.shape[:2]
                crop_h, crop_w = self.crop_size
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)

                img_crop = img[top:top + crop_h, left:left + crop_w]
                lab_crop = lab[top:top + crop_h, left:left + crop_w]

                # Resize images and labels to the target size
                if self.target_size != self.crop_size:
                    img_crop = cv2.resize(img_crop, self.target_size, interpolation=cv2.INTER_LINEAR)
                    lab_crop = cv2.resize(lab_crop, self.target_size, interpolation=cv2.INTER_NEAREST)

                # Check if the crop contains cracks (pixels with value 255 in the label)
                if np.any(lab_crop == 255) and len(crops_with_cracks) < self.num_crops_with_cracks:
                    crops_with_cracks.append((img_crop, lab_crop))
                elif not np.any(lab_crop == 255) and len(crops_without_cracks) < (self.num_crops - self.num_crops_with_cracks):
                    crops_without_cracks.append((img_crop, lab_crop))
                
                attempts += 1

            # If not enough, fill the rest with random crops (no filtering)
            total_needed = self.num_crops - (len(crops_with_cracks) + len(crops_without_cracks))
            for _ in range(total_needed):
                h, w = img.shape[:2]
                crop_h, crop_w = self.crop_size
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                img_crop = img[top:top + crop_h, left:left + crop_w]
                lab_crop = lab[top:top + crop_h, left:left + crop_w]
                # Add to whichever list is not full, or just append to crops_without_white
                if len(crops_with_cracks) < self.num_crops_with_cracks:
                    crops_with_cracks.append((img_crop, lab_crop))
                else:
                    crops_without_cracks.append((img_crop, lab_crop))

            # Combine crops and preprocess
            all_crops = crops_with_cracks + crops_without_cracks
        
        processed_crops = [(_preprocess_img(crop[0]), _preprocess_lab(crop[1])) for crop in all_crops]

        return processed_crops

def _preprocess_img(cvImage):
        '''
        :param cvImage: numpy HWC BGR 0~255
        :return: tensor img CHW BGR  float32 cpu 0~1
        '''

        cvImage = cvImage.transpose(2, 0, 1).astype(np.float32) / 255


        return torch.from_numpy(cvImage)

def _preprocess_lab(cvImage):
        '''
        :param cvImage: numpy 0(background) or 255(crack pixel)
        :return: tensor 0 or 1 float32
        '''
        cvImage = cvImage.astype(np.float32) / 255

        return torch.from_numpy(cvImage)


class loadedDataset(Dataset):
    """
    Create a torch Dataset from data
    """

    def __init__(self, dataset, preprocess=None):
        super(loadedDataset, self).__init__()
        self.samples = []
        if preprocess is None:
            preprocess = lambda x: x
        
        # Flatten all crops from all images
        print("Preprocessing dataset...")
        for item in dataset:
            crops = preprocess(item)
            if isinstance(crops, list):
                self.samples.extend(crops)
            else:
                self.samples.append(crops)
        print(f"Total samples after processing: {len(self.samples)}")

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

