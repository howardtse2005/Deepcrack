import os

def create_dataset_file():
    # Paths
    image_dir = "/home/fyp/DeepCrack/codes/data/NonCrack/test/rgb"
    mask_dir = "/home/fyp/DeepCrack/codes/data/NonCrack/test/gt_fixed"
    output_file = "/home/fyp/DeepCrack/codes/data/train_example.txt"
    
    # Get all image files
    try:
        # Fixed: Use a tuple to check for multiple extensions
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])
    except FileNotFoundError:
        print(f"Error: Directory not found: {image_dir}")
        return
        
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
        
    print(f"Found {len(image_files)} image files")
    
    # Write image-mask pairs to output file
    pairs_count = 0
    with open(output_file, 'w') as f:
        for img_file in image_files:
            # Construct full image path
            img_path = os.path.join(image_dir, img_file)
            
            # Extract base name without extension and construct mask filename
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}_GT.png"
            mask_path = os.path.join(mask_dir, mask_file)
            
            # Check if corresponding mask exists
            if not os.path.exists(mask_path):
                print(f"Warning: No matching mask found for {img_file} (looked for {mask_file})")
                continue
                
            # Write the pair to the output file
            f.write(f"{img_path} {mask_path}\n")
            pairs_count += 1
            
    print(f"Dataset file created at {output_file}")
    print(f"Total pairs written: {pairs_count}")

if __name__ == "__main__":
    create_dataset_file()
