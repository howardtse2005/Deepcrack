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

    def __init__(self, transforms=None, target_size=(512, 512)):
        self.transforms = transforms
        self.target_size = target_size  # Add target size for resizing

    def __call__(self, item):

        img = cv2.imread(item[0])
        mask = cv2.imread(item[1])

        if len(mask.shape) != 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Ensure mask dimensions match image dimensions
        img_h, img_w = img.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        if img_h != mask_h or img_w != mask_w:
            print(f"Warning: Image dimensions ({img_w}x{img_h}) don't match mask dimensions ({mask_w}x{mask_h}). Resizing mask to match.")
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # Add padding to make the image square
        h, w = img.shape[:2]
        if h != w:
            max_dim = max(h, w)
            # Create square canvas with padding
            img_padded = np.zeros((max_dim, max_dim, 3), dtype=img.dtype)
            mask_padded = np.zeros((max_dim, max_dim), dtype=mask.dtype)
            pad_h = (max_dim - h) // 2
            pad_w = (max_dim - w) // 2
            img_padded[pad_h:pad_h+h, pad_w:pad_w+w, :] = img
            mask_padded[pad_h:pad_h+h, pad_w:pad_w+w] = mask
            img, mask = img_padded, mask_padded
            
        # Resize images and labels to the target size
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        img = _preprocess_img(img)
        mask = _preprocess_lab(mask)
        return img, mask


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
        self.dataset = dataset
        if preprocess is None:
            preprocess = lambda x: x
        self.preprocess = preprocess

    def __getitem__(self, index):
        return self.preprocess(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

