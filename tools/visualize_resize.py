import cv2
import numpy as np
import random
import argparse
import os

def resize_and_pad(img, mask, target_size=(448, 448)):
    # Resize to target size
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    return img_resized, mask_resized

def process_directories(images_dir, gt_dir, save_dir):
    """
    Process all image-mask pairs in the specified directories
    
    Args:
        images_dir: Directory containing input images
        gt_dir: Directory containing ground truth masks
        save_dir: Directory to save processed crops
    """
    # Create save directory structure
    os.makedirs(save_dir, exist_ok=True)
    crops_img_dir = os.path.join(save_dir, 'crops/rgb')
    crops_mask_dir = os.path.join(save_dir, 'crops/gt')
    os.makedirs(crops_img_dir, exist_ok=True)
    os.makedirs(crops_mask_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file in os.listdir(images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    print(f"Found {len(image_files)} images to process")
    
    # Global counter for sequential naming
    global_crop_counter = 1
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    
    for image_file in sorted(image_files):
        image_path = os.path.join(images_dir, image_file)
        basename = os.path.splitext(image_file)[0]
        
        # Try different mask naming conventions
        mask_candidates = [
            f"{basename}_GT.png",
            f"{basename}.png",
            f"{basename}_gt.png",
            f"{basename}.jpg",
            f"{basename}.jpeg"
        ]
        
        mask_path = None
        for mask_candidate in mask_candidates:
            candidate_path = os.path.join(gt_dir, mask_candidate)
            if os.path.exists(candidate_path):
                mask_path = candidate_path
                break
        
        if mask_path is None:
            print(f"⚠️  Warning: No matching mask found for {image_file}")
            print(f"   Looked for: {mask_candidates}")
            skipped_count += 1
            continue
        
        try:
            print(f"\n📷 Processing {processed_count + 1}/{len(image_files)}: {image_file}")
            # Process this image-mask pair without visualization
            global_crop_counter = process_single_pair(image_path, mask_path, save_dir, 
                                                    show_visualization=False, 
                                                    crop_counter=global_crop_counter)
            processed_count += 1
        except Exception as e:
            print(f"❌ Error processing {image_file}: {str(e)}")
            skipped_count += 1
            continue
    
    print(f"\n✅ Processing complete!")
    print(f"   Successfully processed: {processed_count} files")
    print(f"   Total crops generated: {global_crop_counter - 1}")
    print(f"   Skipped: {skipped_count} files")
    print(f"   Crops saved to: {os.path.join(save_dir, 'crops')}")

def process_single_pair(image_path, mask_path, save_dir, show_visualization=True, crop_counter=1):
    """
    Process a single image-mask pair (extracted from original visualize function)
    
    Args:
        image_path: Path to input image
        mask_path: Path to input mask
        save_dir: Directory to save crops
        show_visualization: Whether to show CV2 windows
        crop_counter: Starting number for sequential crop naming
        
    Returns:
        Updated crop_counter for next batch
    """
    # Create save directory structure
    crops_img_dir = os.path.join(save_dir, 'crops/rgb')
    crops_mask_dir = os.path.join(save_dir, 'crops/gt')
    os.makedirs(crops_img_dir, exist_ok=True)
    os.makedirs(crops_mask_dir, exist_ok=True)

    # Load image and mask
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask from {mask_path}")

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
            # Rotate image 90 degrees to match mask orientation
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img_h, img_w = img.shape[:2]  # Update dimensions after rotation
            rotated = True
    
    # After potential rotation correction, resize mask if needed
    if img_h != mask_h or img_w != mask_w:
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
        crop_small = True
        
    if h == min_size or w == min_size:
        crop_small = True

    crop_size = (448, 448)
    num_crops_with_white = 1
    num_crops_total = 2
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

    # Save individual crops with sequential numbering
    current_counter = crop_counter
    
    for i, (img_crop, mask_crop) in enumerate(crops):
        # Save crop image with sequential numbering: 1.jpg, 2.jpg, 3.jpg, ...
        img_crop
        crop_img_path = os.path.join(crops_img_dir, f"{current_counter}.jpg")
        cv2.imwrite(crop_img_path, img_crop)
        
        # Save crop mask with sequential numbering: 1_GT.png, 2_GT.png, 3_GT.png, ...
        crop_mask_path = os.path.join(crops_mask_dir, f"{current_counter}_GT.png")
        cv2.imwrite(crop_mask_path, mask_crop)
        
        current_counter += 1

    # Show visualization only if requested
    if show_visualization:
        # Resize crops to smaller dimensions for visualization
        display_size = (128, 128)
        resized_crops = [(cv2.resize(crop[0], display_size, interpolation=cv2.INTER_LINEAR),
                          cv2.resize(crop[1], display_size, interpolation=cv2.INTER_NEAREST)) for crop in crops]

        # Arrange resized crops in a matrix layout with red borders
        grid_size = int(np.ceil(np.sqrt(num_crops_total)))
        border_thickness = 2
        blank_img = np.zeros((display_size[0] * grid_size + border_thickness * (grid_size - 1),
                              display_size[1] * grid_size + border_thickness * (grid_size - 1), 3), dtype=np.uint8)
        blank_img[:, :] = [255, 255, 255]
        blank_mask = np.zeros((display_size[0] * grid_size + border_thickness * (grid_size - 1),
                               display_size[1] * grid_size + border_thickness * (grid_size - 1)), dtype=np.uint8)
        blank_mask[:, :] = 255

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
    
    # Return the updated counter for the next batch
    return current_counter

def visualize(image_path, mask_path, save_dir=None):
    """Original single-file visualization function (kept for backward compatibility)"""
    # ...existing code...
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        img_save_dir = os.path.join(save_dir, 'rgb')
        mask_save_dir = os.path.join(save_dir, 'gt')
        # Add directories for crops
        crops_img_dir = os.path.join(save_dir, 'crops/rgb')
        crops_mask_dir = os.path.join(save_dir, 'crops/gt')
        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)
        os.makedirs(crops_img_dir, exist_ok=True)
        os.makedirs(crops_mask_dir, exist_ok=True)
        print(f"Will save aligned pairs to {save_dir}")
        print(f"Will save individual crops to {os.path.join(save_dir, 'crops')}")

    # Load image and mask
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")

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

    # Save the aligned image and mask if requested
    if save_dir:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        # Save aligned image
        img_save_path = os.path.join(img_save_dir, f"{basename}.jpg")
        cv2.imwrite(img_save_path, img)
        
        # Save aligned mask
        # If mask is binary (0/255), ensure it stays that way
        if np.max(mask) > 1:
            # Ensure mask is binary (0/255)
            mask_binary = np.zeros_like(mask)
            mask_binary[mask > 127] = 255
            mask = mask_binary
            
        mask_save_path = os.path.join(mask_save_dir, f"{basename}_GT.png")
        cv2.imwrite(mask_save_path, mask)
        
        print(f"Saved aligned image to: {img_save_path}")
        print(f"Saved aligned mask to: {mask_save_path}")

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
    num_crops_with_white = 1
    num_crops_total = 2
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

    # Save individual crops if a save directory was specified
    if save_dir:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Saving {len(crops)} crops for {basename}...")
        
        for i, (img_crop, mask_crop) in enumerate(crops):
            # Determine if this crop contains cracks
            has_crack = np.any(mask_crop == 255)
            crop_type = "crack" if has_crack else "no_crack"
            
            # Save crop image
            crop_img_path = os.path.join(crops_img_dir, f"{basename}_crop{i:03d}_{crop_type}.jpg")
            cv2.imwrite(crop_img_path, img_crop)
            
            # Save crop mask
            crop_mask_path = os.path.join(crops_mask_dir, f"{basename}_crop{i:03d}_{crop_type}_GT.png")
            cv2.imwrite(crop_mask_path, mask_crop)
        
        print(f"Saved {len(crops)} crops to {os.path.join(save_dir, 'crops')}")

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
    parser = argparse.ArgumentParser(description="Visualize resized and padded image-GT pair or process entire directories. Can also save aligned pairs.")
    parser.add_argument("input_path", type=str, help="Path to the input image or images directory.")
    parser.add_argument("mask_path", type=str, help="Path to the input ground truth mask or GT directory.")
    parser.add_argument("--save_dir", type=str, help="Directory to save processed crops (required for directory processing).")
    parser.add_argument("--batch", action="store_true", help="Process entire directories instead of single files.")
    args = parser.parse_args()

    if args.batch:
        # Directory processing mode
        if not args.save_dir:
            print("❌ Error: --save_dir is required when using --batch mode")
            exit(1)
        
        if not os.path.isdir(args.input_path):
            print(f"❌ Error: {args.input_path} is not a valid directory")
            exit(1)
        
        if not os.path.isdir(args.mask_path):
            print(f"❌ Error: {args.mask_path} is not a valid directory")
            exit(1)
        
        print(f"🚀 Starting batch processing...")
        print(f"   Images directory: {args.input_path}")
        print(f"   GT directory: {args.mask_path}")
        print(f"   Output directory: {args.save_dir}")
        
        process_directories(args.input_path, args.mask_path, args.save_dir)
    else:
        # Single file processing mode (original behavior)
        visualize(args.input_path, args.mask_path, args.save_dir)
