import cv2
import numpy as np
import argparse

def resize_and_pad(img, mask, target_size=(512, 512)):
    if img is None:
        raise ValueError(f"Failed to load image")
        
    # Convert from BGR (OpenCV default) to RGB for proper training
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # Check if image has 3 channels (is RGB)
    if len(img.shape) != 3 or img.shape[2] != 3:
        print(f"Warning: Image is not RGB (shape: {img.shape}). Converting to RGB.")
        # If image is grayscale, convert to RGB by duplicating the single channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
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
    
    # Apply resizing and padding
    img_resized, mask_resized = resize_and_pad(img, mask)

    # Visualize the results
    cv2.imshow("Resized Image", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
    cv2.imshow("Resized Mask", mask_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize resized and padded image-GT pair.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("mask_path", type=str, help="Path to the input ground truth mask.")
    args = parser.parse_args()

    visualize(args.image_path, args.mask_path)
