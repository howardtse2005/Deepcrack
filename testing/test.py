from data.dataset import CrackDataset
from data.preprocess_pipeline import SlidingWindowCrop
from model.deepcrack import DeepCrack
from model.hnet import HNet
from model.unet import UNet 
from model.attention_unet import AttentionUNet
from model.segformer import SegFormer 
from trainer import DeepCrackTrainer, UNetTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
import datetime
from config import Config as cfg
import torch.nn.functional as F

from visualization import create_visualization
from benchmark import calculate_metrics, write_evaluation_results

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

test_img_path = 'data/july2025_imgs/img_raw_ts'
test_mask_path = 'data/july2025_imgs/masks_raw_ts'
checkpoint_path = 'checkpoints/hnet3_july.pth'
    
#--------------------- Main Test Function ---------------------

def test(test_data_path='data/test_example.txt',
         save_path='results/images',
         eval_path='results/eval',
         pretrained_model=checkpoint_path,
         test_img_path = test_img_path,
         test_mask_path = test_mask_path,
         threshold=0.5):
    
    # Create timestamp for folder names
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    os.makedirs(eval_path, exist_ok=True)
    result_folder_name = f"evaluation_{timestamp}"
    timestamped_save_path = os.path.join(save_path, result_folder_name)
    os.makedirs(timestamped_save_path, exist_ok=True)
    
    print(f"Results will be saved to: {timestamped_save_path}")
    
    # Load dataset
    test_dataset = CrackDataset(
        dataset_img_path=test_img_path,
        dataset_mask_path=test_mask_path,
    )

    # Build model and trainer
    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()
    
    if cfg.model_type == 'hnet':
        model = HNet()
        print("Using HNet architecture")
    elif cfg.model_type == 'unet':
        model = UNet()
        print("Using UNet architecture")
    elif cfg.model_type == 'attention_unet':
        model = AttentionUNet()
        print("Using Attention UNet architecture") 
    elif cfg.model_type == 'segformer':
        model = SegFormer(num_classes=1, phi=cfg.segformer_variant, pretrained=cfg.segformer_pretrained)
        print(f"Using SegFormer {cfg.segformer_variant} architecture")
    else:
        model = DeepCrack()
        print("Using DeepCrack architecture")
        
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)
    
    if cfg.model_type in ['hnet', 'unet', 'attention_unet', 'segformer']:
        trainer = UNetTrainer(model).to(device)
        print("Using UNetTrainer (single output)")
    else:
        trainer = DeepCrackTrainer(model).to(device)
        print("Using DeepCrackTrainer (multi-output)")

    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=True))
    model.eval()
    
    # Initialize sliding window crop inference
    sliding_crop = SlidingWindowCrop(window_size=448, overlap=0.0)
    
    # Set model-specific prediction function
    if cfg.model_type in ['hnet', 'unet', 'attention_unet', 'segformer']:
        sliding_crop.set_model_predictor(lambda model, batch: torch.sigmoid(model(batch)))
    else:
        sliding_crop.set_model_predictor(lambda model, batch: torch.sigmoid(model(batch)[0]))
    
    # Store predictions and ground truths
    all_predictions = []  # List of prediction arrays, each (H, W) with values 0-1
    all_groundtruths = []  # List of ground truth arrays, each (H, W) with values 0-1
    
    print("Processing full-resolution images with sliding window inference...")
    
    # Process each image
    for idx in tqdm(range(len(test_dataset))):
        # Get image and mask from dataset
        img_tensor, gt_tensor = test_dataset[idx]
        
        # Convert tensors back to numpy arrays for processing
        # img_tensor is (C, H, W), convert to (H, W, C) for cv2
        img = img_tensor.permute(1, 2, 0).numpy() * 255.0
        img = img.astype(np.uint8)
        
        # gt_tensor is (H, W)
        gt = gt_tensor.numpy()
        
        # Perform inference using sliding window class
        with torch.no_grad():
            pred_tensor = sliding_crop(model, img)  # Returns tensor (H, W)
        
        # Resize ground truth to match prediction size
        pred_h, pred_w = pred_tensor.shape
        gt_resized = cv2.resize(gt, (pred_w, pred_h))
        gt_np = gt_resized.astype(np.float32)  # Ground truth array (H, W)
        
        # Store results
        pred_np = pred_tensor.cpu().numpy()  # Prediction array (H, W) with values 0-1
        all_predictions.append(pred_np)
        all_groundtruths.append(gt_np)
        
        # Create and save visualization
        visualization = create_visualization(img, gt_np, pred_np)
        save_name = os.path.join(timestamped_save_path, f"fullres_{idx:04d}.png")
        cv2.imwrite(save_name, visualization)
    
    # Calculate all metrics
    print(f"Calculating metrics for {len(all_predictions)} images...")
    metrics_results = calculate_metrics(all_predictions, all_groundtruths, threshold)
    
    # Write evaluation results
    eval_file = os.path.join(eval_path, f"{result_folder_name}.txt")
    write_evaluation_results(eval_file, metrics_results, timestamp, pretrained_model, timestamped_save_path)
    
    print(f"Evaluation results saved to {eval_file}")
    print(f"Images saved to {timestamped_save_path}")


if __name__ == '__main__':
    test()