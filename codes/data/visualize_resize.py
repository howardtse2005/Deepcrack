import cv2
import numpy as np
import random
import argparse

def resize_and_pad(img, mask, target_size=(448, 448)):
    # Resize to target size
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    return img_resized, mask_resized

def visualize(image_path, mask_path):
    # Load image and mask
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask from {mask_path}")

    print(f"Original image dimensions: {img.shape[:2]}")
    print(f"Original mask dimensions: {mask.shape[:2]}")

    # Ensure mask dimensions match image dimensions
    img_h, img_w = img.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    
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
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # Resize if either dimension is less than 448, maintaining aspect ratio
    min_size = 448
    h, w = img.shape[:2]
    crop_small = False
    if h < min_size or w < min_size:
        if h < w:
            scale = min_size / h
        else:
            scale = min_size / w
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        print(f"Resized image dimensions: {img.shape[:2]}")
        print(f"Resized mask dimensions: {mask.shape[:2]}")
        crop_small = True
        
    if h == min_size or w == min_size:
        crop_small = True

    crop_size = (448, 448)
    num_crops_with_white = 21
    num_crops_total = 42
    crops_with_white = []
    crops_without_white = []
    max_attempts = 1000
    attempts = 0
    if crop_small:
        # Only crop one image if either dimension was less than 448
        crops = [(img[:crop_size[0], :crop_size[1]], mask[:crop_size[0], :crop_size[1]])]
    else:
        # Perform random cropping with filtering and fallback after 1000 iterations
        while (len(crops_with_white) < num_crops_with_white or len(crops_without_white) < (num_crops_total - num_crops_with_white)) and attempts < max_attempts:
            h, w = img.shape[:2]
            crop_h, crop_w = crop_size
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)

            img_crop = img[top:top + crop_h, left:left + crop_w]
            mask_crop = mask[top:top + crop_h, left:left + crop_w]

            if np.any(mask_crop == 255) and len(crops_with_white) < num_crops_with_white:
                crops_with_white.append((img_crop, mask_crop))
            elif not np.any(mask_crop == 255) and len(crops_without_white) < (num_crops_total - num_crops_with_white):
                crops_without_white.append((img_crop, mask_crop))
            attempts += 1

        # If not enough, fill the rest with random crops (no filtering)
        total_needed = num_crops_total - (len(crops_with_white) + len(crops_without_white))
        for _ in range(total_needed):
            h, w = img.shape[:2]
            crop_h, crop_w = crop_size
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            img_crop = img[top:top + crop_h, left:left + crop_w]
            mask_crop = mask[top:top + crop_h, left:left + crop_w]
            # Add to whichever list is not full, or just append to crops_without_white
            if len(crops_with_white) < num_crops_with_white:
                crops_with_white.append((img_crop, mask_crop))
            else:
                crops_without_white.append((img_crop, mask_crop))

        crops = crops_with_white + crops_without_white

    # Resize crops to smaller dimensions for visualization
    display_size = (128, 128)  # Define the smaller display size
    resized_crops = [(cv2.resize(crop[0], display_size, interpolation=cv2.INTER_LINEAR),
                      cv2.resize(crop[1], display_size, interpolation=cv2.INTER_NEAREST)) for crop in crops]

    # Arrange resized crops in a matrix layout with red borders
    grid_size = int(np.ceil(np.sqrt(num_crops_total)))  # Determine grid size
    border_thickness = 2  # Thickness of the border
    blank_img = np.zeros((display_size[0] * grid_size + border_thickness * (grid_size - 1),
                          display_size[1] * grid_size + border_thickness * (grid_size - 1), 3), dtype=np.uint8)
    blank_img[:, :] = [255, 255, 255]  # Set the background to red for borders
    blank_mask = np.zeros((display_size[0] * grid_size + border_thickness * (grid_size - 1),
                           display_size[1] * grid_size + border_thickness * (grid_size - 1)), dtype=np.uint8)
    blank_mask[:, :] = 255  # Set the background to white for borders

    for idx, (img_crop, mask_crop) in enumerate(resized_crops):
        row = idx // grid_size
        col = idx % grid_size
        start_row = row * (display_size[0] + border_thickness)
        start_col = col * (display_size[1] + border_thickness)
        blank_img[start_row:start_row + display_size[0], start_col:start_col + display_size[1]] = img_crop
        blank_mask[start_row:start_row + display_size[0], start_col:start_col + display_size[1]] = mask_crop

    cv2.imshow("Cropped Images Matrix", blank_img)
    cv2.imshow("Cropped GT Matrix", blank_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize resized and padded image-GT pair.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("mask_path", type=str, help="Path to the input ground truth mask.")
    args = parser.parse_args()

    visualize(args.image_path, args.mask_path)
