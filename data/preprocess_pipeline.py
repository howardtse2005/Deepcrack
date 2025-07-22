import cv2
import random
import numpy as np

class Augmentation:
    # Base class for all augmentations
    def __init__(self, name:str, n_copy:int=1, use_raw:bool=False):
        '''
        Base class for augmentation modules in the preprocessing pipeline.
        Args:
            name (str): Name of the augmentation.
            n_copy (int): Number of copies to generate for each input image.
            use_raw (bool): If True, the output from the augmentation module will be generated base on raw images.
                by the end of the pipeline. All raw images would be ditched
        '''
        self.name = name
        self.n_copy = n_copy
        self.use_raw = use_raw

class PreprocessPipeline:
    def __init__(self, augmentations: list[Augmentation]=[]):
        self.augmentations = augmentations
        

    def __call__(self, imgs:list[np.ndarray], masks:list[np.ndarray]):        
        if self.augmentations!= []:
            imgs_out, masks_out = [], []
            for aug in self.augmentations:
                if aug.use_raw:
                    imgs_aug, masks_aug = aug(imgs, masks)
                    imgs_out.extend(imgs_aug)
                    masks_out.extend(masks_aug)
                else:
                    imgs_aug, masks_aug = aug(imgs_out, masks_out)
                    imgs_out = imgs_aug
                    masks_out = masks_aug
        return imgs_out, masks_out

class Resize(Augmentation):
    def __init__(self, target_size:tuple):
        '''
        Resizes the input images and masks to the target size. For use at the end of the pipeline.
        Args:
            target_size (tuple): Target size for resizing (width, height).
        '''
        super().__init__('Resize', n_copy=1, use_raw=False)
        self.target_size = target_size

    def __call__(self, img:list[np.ndarray], mask:list[np.ndarray]):
        img = [cv2.resize(i, self.target_size) for i in img]
        mask = [cv2.resize(m, self.target_size) for m in mask]
        return img, mask

class Crop(Augmentation):
    """
        Randomly crops the input images and masks.
        Args:
            range_crop_len (tuple): Range of crop sizes (min, max).
            n_copy (int): Number of copies to generate for each input image.
            each_has_crack (float): Probability that each cropped image will contain a crack.
            use_raw (bool): If True, the output from the augmentation module will be generated based on raw images.
    """
    def __init__(self, range_crop_len:tuple,  n_copy:int=1, each_has_crack:float=0.9, use_raw:bool=True):
        super().__init__('RandomCrop', n_copy=n_copy)
        self.range_crop_len = range_crop_len
        self.each_has_crack = each_has_crack
        self.use_raw = use_raw
    
    def __call__(self, img:list[np.ndarray], mask:list[np.ndarray]):
        imgs, masks = [], []
        for img, mask in zip(img, mask):
            for _ in range(self.n_copy):
                crop_size = random.randint(self.range_crop_len[0], self.range_crop_len[1])
                if random.random() < self.each_has_crack:
                    # Ensure the crop contains a crack
                    centers = np.argwhere(mask > 0)
                    if len(centers) == 0:
                        center_y = random.randint(crop_size // 2, img.shape[0] - crop_size // 2)
                        center_x = random.randint(crop_size // 2, img.shape[1] - crop_size // 2)
                    else:
                        center = random.choice(centers)
                        center_y, center_x = center[0], center[1]
                else:
                    center_y = random.randint(crop_size // 2, img.shape[0] - crop_size // 2)
                    center_x = random.randint(crop_size // 2, img.shape[1] - crop_size // 2)

                img_cropped, mask_cropped = self.center_crop(img, mask, (center_y, center_x), crop_size)
                imgs.append(img_cropped)
                masks.append(mask_cropped)
            
        return imgs, masks

    def center_crop(self, img:np.ndarray, mask:np.ndarray, center:tuple, crop_size:int):
        center_y, center_x = center
        start_y = max(center_y - crop_size // 2, 0)
        start_x = max(center_x - crop_size // 2, 0)
        end_y = min(start_y + crop_size, img.shape[0])
        end_x = min(start_x + crop_size, img.shape[1])

        img_cropped = img[start_y:end_y, start_x:end_x]
        mask_cropped = mask[start_y:end_y, start_x:end_x]
        
        return img_cropped, mask_cropped
        
        
if __name__ == "__main__":
    # Example usage
    img = cv2.imread('img_debug/3.jpg')
    mask = cv2.imread('mask_debug/3_GT.png', cv2.IMREAD_GRAYSCALE)
    
    pipeline = PreprocessPipeline([
        Crop(range_crop_len=(200,1000), n_copy=10),
    ])
    
    imgs_out, masks_out = pipeline([img], [mask])
    
    for i, (out_img, out_mask) in enumerate(zip(imgs_out, masks_out)):
        cv2.imwrite(f'output_img_{i}.jpg', out_img)
        cv2.imwrite(f'output_mask_{i}.png', out_mask)