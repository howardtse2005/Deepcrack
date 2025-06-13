import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def is_valid_gt(gt_path, min_white_pixels=10, max_white_percent=50):
    """
    Check if a ground truth image is valid.
    
    Args:
        gt_path: Path to the ground truth image
        min_white_pixels: Minimum number of white pixels required
        max_white_percent: Maximum percentage of white pixels allowed (to detect over-annotation)
        
    Returns:
        tuple: (is_valid, message, statistics)
    """
    # Check if file exists
    if not os.path.exists(gt_path):
        return False, f"File doesn't exist", {}
    
    # Try to load the image
    try:
        img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        return False, f"Failed to load image: {e}", {}
    
    if img is None:
        return False, f"Failed to load image", {}
    
    # Get image statistics
    stats = {}
    stats['dimensions'] = img.shape
    stats['unique_values'] = len(np.unique(img))
    
    # Check if image is binary or near-binary
    unique_values = np.unique(img)
    if len(unique_values) > 10:  # Allow some gray values but not too many
        return False, f"Not a binary mask, found {len(unique_values)} unique pixel values", stats
    
    # Check if image has any white pixels (cracks)
    white_pixels = np.sum(img > 127)  # Threshold at 127 to accommodate for compression artifacts
    stats['white_pixels'] = int(white_pixels)
    stats['white_percent'] = (white_pixels / (img.shape[0] * img.shape[1])) * 100
    
    if white_pixels < min_white_pixels:
        return False, f"Too few white pixels: {white_pixels} (min: {min_white_pixels})", stats
    
    # Check if image has too many white pixels (possible error in annotation)
    if stats['white_percent'] > max_white_percent:
        return False, f"Too many white pixels: {stats['white_percent']:.2f}% (max: {max_white_percent}%)", stats
    
    return True, "Valid GT image", stats

def validate_gt_directory(gt_dir, img_dir=None, output_csv=None, visualize=False, fix_issues=False):
    """
    Validate all ground truth images in a directory.
    
    Args:
        gt_dir: Directory containing ground truth images
        img_dir: Optional directory containing input images to check matching dimensions
        output_csv: Optional path to save results as CSV
        visualize: Whether to show problematic images
        fix_issues: Whether to attempt to fix issues in GT images
    """
    if not os.path.exists(gt_dir):
        print(f"Error: GT directory does not exist: {gt_dir}")
        return
    
    # Get all image files in the GT directory
    gt_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        gt_files.extend([f for f in os.listdir(gt_dir) if f.endswith(ext)])
    
    if not gt_files:
        print(f"No image files found in {gt_dir}")
        return
    
    print(f"Found {len(gt_files)} GT images to validate")
    
    # Track statistics
    valid_count = 0
    issues = {}  # type of issue -> count
    problematic_files = []
    
    # Create directory for fixed images if needed
    if fix_issues:
        fixed_dir = os.path.join(gt_dir, 'fixed')
        os.makedirs(fixed_dir, exist_ok=True)
    
    # Process each GT image
    for gt_file in tqdm(gt_files):
        gt_path = os.path.join(gt_dir, gt_file)
        
        # Validate the GT image
        is_valid, message, stats = is_valid_gt(gt_path)
        
        # Check if dimensions match corresponding image if img_dir is provided
        if is_valid and img_dir:
            # First try to find corresponding image with same name
            base_name = os.path.splitext(gt_file)[0]
            if base_name.endswith('_GT'):
                base_name = base_name[:-3]
                
            # Look for matching input image with various extensions
            img_found = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                img_path = os.path.join(img_dir, base_name + ext)
                if os.path.exists(img_path):
                    img_found = True
                    img = cv2.imread(img_path)
                    if img.shape[:2] != stats['dimensions']:
                        is_valid = False
                        message = f"Dimension mismatch: Input {img.shape[:2]}, GT {stats['dimensions']}"
                        if 'dimension_mismatch' not in issues:
                            issues['dimension_mismatch'] = 0
                        issues['dimension_mismatch'] += 1
                    break
                    
            if not img_found:
                if 'no_matching_image' not in issues:
                    issues['no_matching_image'] = 0
                issues['no_matching_image'] += 1
                message += " (No matching input image found)"
        
        if is_valid:
            valid_count += 1
        else:
            # Track the issue type
            issue_type = message.split(':')[0]
            if issue_type not in issues:
                issues[issue_type] = 0
            issues[issue_type] += 1
            
            problematic_files.append((gt_path, message, stats))
            
            # Try to fix issues if requested
            if fix_issues:
                try:
                    # Read the image
                    img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Convert to binary (anything above 127 becomes 255, below becomes 0)
                        binary = np.zeros_like(img)
                        binary[img > 127] = 255
                        
                        # Save the fixed image
                        fixed_path = os.path.join(fixed_dir, gt_file)
                        cv2.imwrite(fixed_path, binary)
                        print(f"Fixed and saved: {fixed_path}")
                except Exception as e:
                    print(f"Failed to fix {gt_file}: {e}")
            
            # Visualize problematic images if requested
            if visualize and len(problematic_files) <= 5:  # Limit to first 5 to avoid too many windows
                try:
                    img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        plt.figure(figsize=(10, 4))
                        plt.subplot(1, 2, 1)
                        plt.imshow(img, cmap='gray')
                        plt.title(f"GT: {gt_file}")
                        
                        # If there's a corresponding input image, show it
                        if img_dir and img_found:
                            rgb_img = cv2.imread(img_path)
                            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                            plt.subplot(1, 2, 2)
                            plt.imshow(rgb_img)
                            plt.title(f"Input: {os.path.basename(img_path)}")
                        
                        plt.suptitle(f"Issue: {message}")
                        plt.tight_layout()
                        plt.show(block=False)
                except Exception as e:
                    print(f"Failed to visualize {gt_file}: {e}")
    
    # Print summary
    print("\n--- GT Validation Summary ---")
    print(f"Total GT images: {len(gt_files)}")
    print(f"Valid GT images: {valid_count} ({valid_count/len(gt_files)*100:.1f}%)")
    print(f"Problematic GT images: {len(gt_files) - valid_count} ({(len(gt_files) - valid_count)/len(gt_files)*100:.1f}%)")
    
    # Print issues breakdown
    if issues:
        print("\nIssues breakdown:")
        for issue_type, count in issues.items():
            print(f"  - {issue_type}: {count} files")
    
    # Save results to CSV if requested
    if output_csv:
        import csv
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File', 'Valid', 'Message', 'White Pixels', 'White Percent'])
            
            # Write valid files
            for gt_file in gt_files:
                gt_path = os.path.join(gt_dir, gt_file)
                is_valid, message, stats = is_valid_gt(gt_path)
                if 'white_pixels' in stats:
                    writer.writerow([gt_file, is_valid, message, stats['white_pixels'], f"{stats['white_percent']:.2f}%"])
                else:
                    writer.writerow([gt_file, is_valid, message, 'N/A', 'N/A'])
            
        print(f"\nResults saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ground truth images for DeepCrack")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth images")
    parser.add_argument("--img_dir", type=str, help="Optional directory containing corresponding input images")
    parser.add_argument("--output_csv", type=str, help="Save results to CSV file")
    parser.add_argument("--visualize", action="store_true", help="Visualize problematic images")
    parser.add_argument("--fix", action="store_true", help="Try to fix issues (converts to binary)")
    args = parser.parse_args()
    
    validate_gt_directory(args.gt_dir, args.img_dir, args.output_csv, args.visualize, args.fix)
