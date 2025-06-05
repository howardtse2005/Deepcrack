file_path = "/home/intern/DeepCrack/codes/data/train_example.txt"

with open(file_path, "a") as f:
    for i in range(6300, 6305):
        image_path = f"/home/intern/DeepCrack/codes/data/CrackTree260/image/{i}.jpg"
        mask_path = f"/home/intern/DeepCrack/codes/data/CrackTree260/gt/{i}.bmp"
        f.write(f"{image_path} {mask_path}\n")

with open(file_path, "a") as f:
    for i in range(6306, 6320):
        image_path = f"/home/intern/DeepCrack/codes/data/CrackTree260/image/{i}.jpg"
        mask_path = f"/home/intern/DeepCrack/codes/data/CrackTree260/gt/{i}.bmp"
        f.write(f"{image_path} {mask_path}\n")
