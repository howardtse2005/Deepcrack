How to train:
1. Save the RGB and Ground Truth (mask) images at /home/fyp/DeepCrack/codes/data/OurImages and /home/fyp/DeepCrack/codes/data/OurMasks respectively
2. Include the training dataset path in /home/fyp/DeepCrack/codes/data/train_example.txt. The first column is the RGB and the second column is the Ground Truth.
3. Include the val dataset path in /home/fyp/DeepCrack/codes/data/val_example.txt. The first column is the RGB and the second column is the Ground Truth.
4. Run the training by python3 train.py
5. The pth files (per epoch) will be located in /home/fyp/DeepCrack/codes/checkpoints/DeepCrack_CT260_FT1

How to test:
1. Include the testing dataset path in /home/fyp/DeepCrack/codes/data/test_example.txt. Both the first and second column is the RGB.
2. Include the pth path in the test.py
3. Run the testing by python3 test.py
4. See the results in /home/fyp/DeepCrack/codes/deepcrack_results

How to preprocess the images:
1. Resize the ground truth image (mask) to be the same size (resolution) to the rgb image
2. Add paddings to both the rgb and ground truth images to be square size
3. Resize both the rgb and ground truth (already padded) to become 512x512