import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from os.path import exists, join
from os import listdir, makedirs, remove
class Augmentation:
    # Base class for all augmentations
    def __init__(self, name:str, n_copy:int=1):
        '''
        Base class for augmentation modules in the preprocessing pipeline.
        Args:
            name (str): Name of the augmentation.
            n_copy (int): Number of copies to generate for each input image.
            use_raw (bool): If True, the output from the augmentation module will be generated base on raw images.
                by the end of the pipeline. All raw images would be ditched
                For example, if the input to an pipeline is 10, with 2 modules using n_copies=100,
                In first scenario, 
                set the first module use_raw=True and second to use_raw=True, the total output after 1st module will be 1000(10 * 100) 
                the total output after the 2nd module will be 2000(1000 + 100*10). So there will be 2000 by the end
                
                In Second sceneario,
                if the first module use_raw=True and second to use_raw=False, the total output after 1st module will be 1000(10 * 100)
                the total output after the 2nd module will be 100000 (1000 * 100)
        '''
        self.name = name
        self.n_copy = n_copy

class PreprocessPipeline:
    def __init__(self, temp_dir:str, keep_temp:bool = False, augmentations:list[Augmentation]=[]):
        self.augmentations = augmentations
        self.save_id = 0
        self.img_dir = temp_dir + '/imgs'
        self.mask_dir = temp_dir + '/masks'
        self.keep_temp = keep_temp
    
    def __call__(self, img, mask):        
        imgs_out, masks_out = [], []
        if not isinstance(img, np.ndarray) or not isinstance(mask, np.ndarray):
            raise TypeError("Input img and mask must be cv2 images")
        
        for i, aug in enumerate(self.augmentations):
            if isinstance(aug, (Crop, SlideCrop)):
                if i != 0:
                    raise ValueError("Crop and SlideCrop should be the first augmentation in the pipeline if you wish to use it")
                else:
                    imgs, masks = aug([img], [mask])
                imgs_out.extend(imgs)
                masks_out.extend(masks)
            elif isinstance(aug, Augmentation):
                if len(imgs_out) == 0:
                    imgs_out, masks_out = aug([img], [mask])
                else:
                    imgs_out, masks_out = aug(imgs_out, masks_out)
            else:
                raise TypeError(f"Unsupported augmentation type: {type(aug)}")
            
        if len(imgs_out) == 0 or len(masks_out) == 0:
            imgs_out, masks_out = [img], [mask]
        self._save(imgs_out, masks_out)

    def __del__(self):
        if not self.keep_temp:
            if exists(self.img_dir):
                for f in listdir(self.img_dir):
                    remove(join(self.img_dir, f))
            if exists(self.mask_dir):
                for f in listdir(self.mask_dir):
                    remove(join(self.mask_dir, f))
            print(f"Temporary files in {self.img_dir} and {self.mask_dir} have been deleted.")
        else:
            print(f"Temporary files in {self.img_dir} and {self.mask_dir} are kept as keep_temp is set to True.")
            
    def _save(self, imgs, masks):
        if not exists(self.img_dir):
            makedirs(self.img_dir)
        if not exists(self.mask_dir):
            makedirs(self.mask_dir)
        
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            img_path = f"{self.img_dir}/{self.save_id}.png"
            mask_path = f"{self.mask_dir}/{self.save_id}.png"
            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)
            self.save_id += 1
            
            

class Resize(Augmentation):
    def __init__(self, target_size:tuple):
        '''
        Resizes the input images and masks to the target size. For use at the end of the pipeline.
        Args:
            target_size (tuple): Target size for resizing (width, height).
        '''
        super().__init__('Resize', n_copy=1)
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
    def __init__(self, range_crop_len:tuple,  n_copy:int=1, each_has_crack:float=0.9):
        super().__init__('RandomCrop', n_copy=n_copy)
        self.range_crop_len = range_crop_len
        self.each_has_crack = each_has_crack
    
    def __call__(self, img:list[np.ndarray], mask:list[np.ndarray]):
        imgs, masks = [], []
        for img, mask in zip(img, mask):
            for _ in range(self.n_copy):
                crop_size = random.randint(self.range_crop_len[0], self.range_crop_len[1])
                crop_size = min(crop_size, img.shape[0], img.shape[1])  # Ensure crop size does not exceed image dimensions
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
        
class RandomRotate(Augmentation):
    """
        Randomly rotates the input images and masks.
        Args:
            angle_range (tuple): Range of angles for rotation (min_angle, max_angle).
            n_copy (int): Number of copies to generate for each input image.
            use_raw (bool): If True, the output from the augmentation module will be generated based on raw images.
    """
    def __init__(self, angle_range:tuple=(-90, 90), n_copy:int=1, p_apply:float=0.5):
        super().__init__('RandomRotate', n_copy=n_copy)
        self.angle_range = angle_range
        self.p_apply = p_apply

    def __call__(self, imgs:list[np.ndarray], masks:list[np.ndarray]):
        imgs_out, masks_out = [], []
        for img, mask in zip(imgs, masks):
            if random.random() < self.p_apply:
                # Rotate the image and mask
                angle = random.uniform(self.angle_range[0], self.angle_range[1])
                M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1)
                img_rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                mask_rotated = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
                imgs_out.append(img_rotated)
                masks_out.append(mask_rotated)
            else:
                imgs_out.append(img)
                masks_out.append(mask)
        return imgs_out, masks_out
    
    
class RandomJitter(Augmentation):
    """
        Randomly jitters the input images and masks.
        Args:
            jitter_strength (float): Strength of the jittering effect.
            n_copy (int): Number of copies to generate for each input image.
            use_raw (bool): If True, the output from the augmentation module will be generated based on raw images.
            prop_apply (float): Probability that the jittering will be applied to each copy.
    """
    def __init__(self, jitter_strength:float=0.1, n_copy:int=1, prop_apply:float=0.2):
        super().__init__('RandomJitter', n_copy=n_copy)
        self.jitter_strength = jitter_strength
        self.prop_apply = prop_apply

    def __call__(self, imgs:list[np.ndarray], masks:list[np.ndarray]):
        imgs_out, masks_out = [], []
        for i in range(len(imgs)):
            if random.random() < self.prop_apply:
                # Convert to float32 for arithmetic
                img_float = imgs[i].astype(np.float32)
                
                # Apply stronger jitter with normal distribution for more natural look
                jitter = np.random.normal(0, self.jitter_strength * 255, imgs[i].shape)
                jittered_img = img_float + jitter
                jittered_img = np.clip(jittered_img, 0, 255).astype(np.uint8)
            else:
                jittered_img = imgs[i]
        
            imgs_out.append(jittered_img)
            masks_out.append(masks[i])
        return imgs_out, masks_out
    
class RandomGaussianNoise(Augmentation):
    """
        Adds random Gaussian noise to the input images.
        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
            n_copy (int): Number of copies to generate for each input image.
            use_raw (bool): If True, the output from the augmentation module will be generated based on raw images.
    """
    def __init__(self, mean:float=0, std:float=0.1, n_copy:int=1, p_apply:float=0.2):
        super().__init__('RandomGaussianNoise', n_copy=n_copy)
        self.mean = mean
        self.std = std
        self.p_apply = p_apply

    def __call__(self, imgs:list[np.ndarray], masks:list[np.ndarray]):
        imgs_out, masks_out = [], []
        for i, img in enumerate(imgs):
            if random.random() < self.p_apply:
                # Generate noise in float32 to preserve negative values
                noise = np.random.normal(self.mean, self.std * 255, img.shape).astype(np.float32)
                
                # Convert image to float32, add noise, then clip and convert back
                img_float = img.astype(np.float32)
                noisy_img = img_float + noise
                noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            else:
                noisy_img = img
        
            imgs_out.append(noisy_img)
            masks_out.append(masks[i])  # Use corresponding mask index
        return imgs_out, masks_out
    
class SlideCrop(Augmentation):
    def __init__(self, crop_size:int=448, step_size:int=224):
        '''
        Slides a crop of size crop_size over the input images and masks.
        Args:
            crop_size (int): Size of the crop.
            step_size (int): Step size for sliding the crop.
        '''
        super().__init__('SlideCrop', n_copy=1)
        self.crop_size = crop_size
        self.step_size = step_size

    def __call__(self, img:list[np.ndarray], mask:list[np.ndarray]):
        imgs_out, masks_out = [], []
        for i in range(len(img)):
            h, w = img[i].shape[:2]
            for y in range(0, h - self.crop_size + 1, self.step_size):
                for x in range(0, w - self.crop_size + 1, self.step_size):
                    img_crop = img[i][y:y + self.crop_size, x:x + self.crop_size]
                    mask_crop = mask[i][y:y + self.crop_size, x:x + self.crop_size]
                    imgs_out.append(img_crop)
                    masks_out.append(mask_crop)
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
    img = cv2.imread('data/1.jpg')
    mask = cv2.imread('data/1_GT.png', cv2.IMREAD_GRAYSCALE)
    
    pipeline = PreprocessPipeline(
        augmentations=[
            SlideCrop(crop_size=448, step_size=224),
            RandomJitter(jitter_strength=0.1, n_copy=5),
            Resize(target_size=(448, 448))
        ], 
        save_path='data/preprocess_output', 
        keep_temp=True
    )
    
    pipeline(img, mask)