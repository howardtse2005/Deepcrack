import os

def create_dataset_file():
    # Paths
    image_dir = "/home/fyp/DeepCrack/codes/data/images"
    mask_dir = "/home/fyp/DeepCrack/codes/data/masks"
    output_file = "/home/fyp/DeepCrack/codes/data/train_example.txt"
    # Get all image files
    try:
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    except FileNotFoundError:
        print(f"Error: Directory not found: {image_dir}")
        return
    if not image_files:
        print(f"No jpg files found in {image_dir}")
        return
    print(f"Found {len(image_files)} image files")
    # Write image-mask pairs to output file
    with open(output_file, 'w') as f:
        for img_file in image_files:
            # Construct full paths
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)
            # Check if corresponding mask exists
            if not os.path.exists(mask_path):
                print(f"Warning: No matching mask found for {img_file}")
                continue
            # Write the pair to the output file
            f.write(f"{img_path} {mask_path}\n")
    print(f"Dataset file created at {output_file}")
    print(f"Total pairs written: {len(image_files)}")

if __name__ == "__main__":
    create_dataset_file()
