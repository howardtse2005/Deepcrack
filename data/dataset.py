import data.preprocess_pipeline as pp
from torch.utils.data import Dataset
from os.path import exists, join
import os, cv2
import numpy as np
import torch
class CrackDataset(Dataset):
    """
    Create a torch Dataset from data
    """

    def __init__(self, dataset_img_path:str, dataset_mask_path:str, augmentations:list[pp.Augmentation]=[],
                 mask_postfix:str='_GT.png'):
        super().__init__()
        self.pp = pp.PreprocessPipeline(augmentations) 
        if not exists(dataset_img_path):
            raise FileNotFoundError(f"Dataset path {dataset_img_path} does not exist.")
        if not exists(dataset_mask_path):
            raise FileNotFoundError(f"Dataset path {dataset_mask_path} does not exist.")
        
        image_files, mask_files = self._get_files(dataset_img_path, dataset_mask_path)
        imgs, masks = self._get_images(image_files, mask_files, dataset_img_path, dataset_mask_path, mask_postfix)
        
        print("images and masks loaded, applying augmentations...")
        self.imgs, self.masks = self.pp(imgs, masks) # apply augmentations
        print(f"Dataset initialized with {len(self.imgs)} images and masks.")
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        if index < 0 or index >= len(self.imgs):
            raise IndexError("Index out of range.")
        img, mask = torch.from_numpy(self.imgs[index]).permute(2, 0, 1), torch.from_numpy(self.masks[index])
        return img, mask # img out is (C, H, W) and mask out is (H, W) in tensor

    def _get_files(self, img_dir, mask_dir):
        '''
        Get all image and mask files in the specified directories. and check validity
        '''
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            raise FileExistsError(f"No image files found in {img_dir}")
        if not mask_files:
            raise FileExistsError(f"No mask files found in {mask_dir}")
        if len(image_files) != len(mask_files):
            raise ValueError("Number of images and masks do not match.")
        return image_files, mask_files
    
    def _get_images(self, image_files, mask_files, dataset_img_path, dataset_mask_path, mask_postfix):
        imgs, masks = [], []
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}{mask_postfix}"
            if mask_file in mask_files:
                img_path = join(dataset_img_path, img_file)
                mask_path = join(dataset_mask_path, mask_file)
                img, mask = self._read_image(img_path, mask_path)
                imgs.append(img)
                masks.append(mask)
            else:
                raise FileNotFoundError(f"Mask file {mask_file} not found for image {img_file}.")
        return imgs, masks
    
    def _read_image(self, img_path, mask_path):
        '''
        Read an image and its corresponding mask. And check if they are valid.
        images are assumed to be binary in single channel
        '''
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        if img is None:
            raise ValueError(f"Image {img_path} could not be read.")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask {mask_path} could not be read.")
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image {img_path} and mask {mask_path} dimensions do not match.")
        if mask.max() > 1:
            mask = mask // 255  # Normalize mask to 0-1 range
        mask = np.ceil(mask).astype(np.uint8)
        return img, mask
    
   
if __name__ == "__main__":
    # Example usage
    transforms = [
        pp.Crop(range_crop_len=(200, 1000), n_copy=10),
        pp.Resize(target_size=(448, 448))
    ]

    dataset = CrackDataset(
        dataset_img_path="img_debug",
        dataset_mask_path="mask_debug",
        augmentations=transforms
    )

    print(f"Dataset length: {len(dataset)}")
    for i in range(len(dataset)):
        img, mask = dataset[i]
        print(f"Image {i} shape: {img.shape}, Mask {i} shape: {mask.shape}")

        # Convert mask to 3-channel for concatenation with image
        mask_3ch = np.expand_dims(mask, axis=2)  # Add channel dimension
        mask_3ch = np.repeat(mask_3ch, 3, axis=2)  # Repeat to get 3 channels
        mask_3ch = mask_3ch * 255  # Scale to 0-255 for visibility

        # Concatenate image and mask horizontally
        out = np.concatenate((img,mask_3ch), axis=1)
        cv2.imshow("Output", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()