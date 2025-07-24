import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
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
    def __init__(self, augmentations:list[Augmentation]=[]):
        self.augmentations = augmentations
        

    def __call__(self, imgs:list[np.ndarray], masks:list[np.ndarray]):        
        imgs_out, masks_out = [], []
        
        # If no augmentations, return original images
        if not self.augmentations:
            return imgs, masks
            
        # Initialize with original images for the first augmentation
        imgs_out, masks_out = imgs[:], masks[:]

        with tqdm.tqdm(total=len(self.augmentations), desc='Applying augmentations') as pbar:
            for aug in self.augmentations:
                if aug.use_raw:
                    imgs_aug, masks_aug = aug(imgs, masks)
                    imgs_out.extend(imgs_aug)
                    masks_out.extend(masks_aug)
                pbar.update(1)
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
        
        
class RandomJitter(Augmentation):
    """
        Randomly jitters the input images and masks.
        Args:
            jitter_strength (float): Strength of the jittering effect.
            n_copy (int): Number of copies to generate for each input image.
            use_raw (bool): If True, the output from the augmentation module will be generated based on raw images.
            prop_apply (float): Probability that the jittering will be applied to each copy.
    """
    def __init__(self, jitter_strength:float=0.2, n_copy:int=1, use_raw:bool=False, prop_apply:float=0.5):
        super().__init__('RandomJitter', n_copy=n_copy, use_raw=use_raw)
        self.jitter_strength = jitter_strength
        self.prop_apply = prop_apply

    def __call__(self, imgs:list[np.ndarray], masks:list[np.ndarray]):
        imgs_out, masks_out = [], []
        for i in range(len(imgs)):
            for _ in range(self.n_copy):
                if random.random() < self.prop_apply:
                    # Apply jittering
                    jittered_img = imgs[i] + np.random.uniform(-self.jitter_strength * 255 , self.jitter_strength * 255, imgs[i].shape)
                    jittered_img = np.clip(jittered_img, 0, 255)
                    jittered_img = jittered_img.astype(np.uint8)
                else:
                    # No jittering
                    jittered_img = imgs[i]
                
                imgs_out.append(jittered_img)
                masks_out.append(masks[i])
        return imgs_out, masks_out
    
class SlidingWindowCrop:
    """
    Perform inference on large images using sliding window approach
    """
    def __init__(self, window_size=448, overlap=0.2):
        """
        Args:
            window_size: Size of sliding window (default 448)
            overlap: Overlap ratio between windows (default 0.2)
        """
        self.window_size = window_size
        self.overlap = overlap
    
    def __call__(self, model, image):
        """
        Args:
            model: Trained model
            image: Input image tensor (C, H, W) or numpy array (H, W, C)
        
        Returns:
            prediction: Full resolution prediction tensor
        """
        device = next(model.parameters()).device
        
        # Handle both numpy arrays and tensors
        if isinstance(image, np.ndarray):
            # Convert numpy array to tensor
            if len(image.shape) == 3 and image.shape[2] == 3:  # H, W, C format
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            else:
                raise ValueError("Input image should be in H, W, C format for numpy arrays")
        
        # Resize to multiples of 32 for model compatibility
        C, H, W = image.shape
        new_h = (H // 32) * 32
        new_w = (W // 32) * 32
        
        if H != new_h or W != new_w:
            image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
            H, W = new_h, new_w

        stride = int(self.window_size * (1 - self.overlap))
        
        # Ensure complete coverage by calculating windows differently
        h_windows = (H + stride - 1) // stride
        w_windows = (W + stride - 1) // stride
        
        # Initialize prediction and weight maps
        prediction = torch.zeros((1, H, W), device=device)
        weight_map = torch.zeros((H, W), device=device)
        
        # Create Gaussian weight for blending
        gaussian_weight = torch.ones((self.window_size, self.window_size), device=device)
        center = self.window_size // 2
        for i in range(self.window_size):
            for j in range(self.window_size):
                dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                gaussian_weight[i, j] = np.exp(-(dist ** 2) / (2 * (center / 3) ** 2))
        
        model.eval()
        with torch.no_grad():
            for h_idx in range(h_windows):
                for w_idx in range(w_windows):
                    # Calculate window coordinates ensuring full coverage
                    h_start = min(h_idx * stride, H - self.window_size)
                    w_start = min(w_idx * stride, W - self.window_size)
                    
                    # Ensure we don't go beyond image boundaries
                    h_start = max(0, h_start)
                    w_start = max(0, w_start)
                    h_end = min(h_start + self.window_size, H)
                    w_end = min(w_start + self.window_size, W)
                    
                    # Extract window with proper padding if needed
                    if h_end - h_start < self.window_size or w_end - w_start < self.window_size:
                        # Pad the window to ensure it's exactly window_size x window_size
                        window = image[:, h_start:h_end, w_start:w_end]
                        pad_h = self.window_size - (h_end - h_start)
                        pad_w = self.window_size - (w_end - w_start)
                        
                        # Use reflection padding instead of zero padding
                        window = F.pad(window, (0, pad_w, 0, pad_h), mode='reflect')
                    else:
                        window = image[:, h_start:h_end, w_start:w_end]
                    
                    window_batch = window.unsqueeze(0).to(device)
                    
                    # Get prediction for window - this will be handled by the caller
                    window_pred = self._predict_window(model, window_batch)
                    window_pred = window_pred.squeeze(0)
                    
                    # Crop prediction back to actual window size if we padded
                    actual_h = h_end - h_start
                    actual_w = w_end - w_start
                    window_pred = window_pred[:, :actual_h, :actual_w]
                    
                    # Create corresponding weight map for this window
                    current_weight = gaussian_weight[:actual_h, :actual_w]
                    
                    # Apply Gaussian weighting
                    weighted_pred = window_pred * current_weight
                    
                    # Add to full prediction with weights
                    prediction[:, h_start:h_end, w_start:w_end] += weighted_pred
                    weight_map[h_start:h_end, w_start:w_end] += current_weight
        
        # Normalize by weights to handle overlaps
        weight_map[weight_map == 0] = 1  # Avoid division by zero
        prediction = prediction / weight_map.unsqueeze(0)
        
        return prediction.squeeze(0)
    
    def _predict_window(self, model, window_batch):
        """
        Predict on a single window - to be overridden or configured based on model type
        """
        # Default implementation - assumes single output model
        return torch.sigmoid(model(window_batch))
    
    def set_model_predictor(self, predictor_func):
        """
        Set custom prediction function for different model types
        """
        self._predict_window = predictor_func

if __name__ == "__main__":
    # Example usage
    img = cv2.imread('img_debug/3.jpg')
    mask = cv2.imread('mask_debug/3_GT.png', cv2.IMREAD_GRAYSCALE)
    
    pipeline = PreprocessPipeline([
        RandomJitter(use_raw=True, jitter_strength=0.1, n_copy=5),
    ])
    
    imgs_out, masks_out = pipeline([img], [mask])
    
    for i, (out_img, out_mask) in enumerate(zip(imgs_out, masks_out)):
        cv2.imwrite(f'output_img_{i}.jpg', out_img)
        cv2.imwrite(f'output_mask_{i}.png', out_mask)