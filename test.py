from data.dataset import CrackDataset
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

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

test_img_path = 'data/july2025_imgs/img_raw_ts'
test_mask_path = 'data/july2025_imgs/masks_raw_ts'
checkpoint_path = 'checkpoints/unet_july.pth'

def sliding_window_inference(model, image, window_size=448, overlap=0.2):
    """
    Perform inference on large images using sliding window approach
    
    Args:
        model: Trained model
        image: Input image tensor (C, H, W) or numpy array (H, W, C)
        window_size: Size of sliding window (default 448)
        overlap: Overlap ratio between windows (default 0.2)
    
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

    stride = int(window_size * (1 - overlap))
    
    # Ensure complete coverage by calculating windows differently
    h_windows = (H + stride - 1) // stride
    w_windows = (W + stride - 1) // stride
    
    # Initialize prediction and weight maps
    prediction = torch.zeros((1, H, W), device=device)
    weight_map = torch.zeros((H, W), device=device)
    
    # Create Gaussian weight for blending
    gaussian_weight = torch.ones((window_size, window_size), device=device)
    center = window_size // 2
    for i in range(window_size):
        for j in range(window_size):
            dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
            gaussian_weight[i, j] = np.exp(-(dist ** 2) / (2 * (center / 3) ** 2))
    
    model.eval()
    with torch.no_grad():
        for h_idx in range(h_windows):
            for w_idx in range(w_windows):
                # Calculate window coordinates ensuring full coverage
                h_start = min(h_idx * stride, H - window_size)
                w_start = min(w_idx * stride, W - window_size)
                
                # Ensure we don't go beyond image boundaries
                h_start = max(0, h_start)
                w_start = max(0, w_start)
                h_end = min(h_start + window_size, H)
                w_end = min(w_start + window_size, W)
                
                # Extract window with proper padding if needed
                if h_end - h_start < window_size or w_end - w_start < window_size:
                    # Pad the window to ensure it's exactly window_size x window_size
                    window = image[:, h_start:h_end, w_start:w_end]
                    pad_h = window_size - (h_end - h_start)
                    pad_w = window_size - (w_end - w_start)
                    
                    # Use reflection padding instead of zero padding
                    window = F.pad(window, (0, pad_w, 0, pad_h), mode='reflect')
                else:
                    window = image[:, h_start:h_end, w_start:w_end]
                
                window_batch = window.unsqueeze(0).to(device)
                
                # Get prediction for window
                if cfg.model_type in ['hnet', 'unet', 'attention_unet', 'segformer']:
                    window_pred = torch.sigmoid(model(window_batch))
                else:
                    window_pred = torch.sigmoid(model(window_batch)[0])
                
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

def draw_text_with_outline(img, text, position, font, font_scale, text_color, outline_color, thickness=1, outline_thickness=3):
    """
    Draw text with an outline to make it more visible on various backgrounds
    """
    # Draw the outline (thicker text in outline_color)
    cv2.putText(img, text, position, font, font_scale, outline_color, thickness + outline_thickness, cv2.LINE_AA)
    # Draw the inner text (normal thickness in text_color)
    cv2.putText(img, text, position, font, font_scale, text_color, thickness, cv2.LINE_AA)


def create_visualization(original_img, ground_truth, prediction, max_display_size=1200):
    """
    Create a 3-panel visualization showing original image, ground truth, and prediction
    """
    # Convert prediction and ground truth to proper format
    if len(ground_truth.shape) == 2:
        display_gt = (ground_truth * 255).astype(np.uint8)
    else:
        display_gt = ground_truth.astype(np.uint8)
        
    if len(prediction.shape) == 2:
        display_pred = (prediction * 255).astype(np.uint8)
    else:
        display_pred = prediction.astype(np.uint8)
    
    # Resize for display if too large
    h, w = display_pred.shape
    if max(h, w) > max_display_size:
        scale = max_display_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        display_img = cv2.resize(original_img, (new_w, new_h))
        display_pred = cv2.resize(display_pred, (new_w, new_h))
        display_gt = cv2.resize(display_gt, (new_w, new_h))
    else:
        # Resize original image to match prediction size
        display_img = cv2.resize(original_img, (w, h))
    
    h_disp, w_disp = display_pred.shape
    border_thickness = 5
    
    # Create 3-panel display with white borders
    total_height = h_disp * 3 + border_thickness * 2
    display_combined = np.ones((total_height, w_disp, 3), dtype=np.uint8) * 255
    
    # Original image (top)
    display_combined[:h_disp, :, :] = display_img
    
    # Ground truth (middle)
    gt_start = h_disp + border_thickness
    gt_end = gt_start + h_disp
    display_combined[gt_start:gt_end, :, :] = np.stack([display_gt] * 3, axis=2)
    
    # Prediction (bottom)
    pred_start = gt_end + border_thickness
    pred_end = pred_start + h_disp
    display_combined[pred_start:pred_end, :, :] = np.stack([display_pred] * 3, axis=2)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    text_color = (255, 255, 255)
    outline_color = (0, 0, 0)
    thickness = 6
    
    draw_text_with_outline(display_combined, "Original Image", (10, 60), 
                          font, font_scale, text_color, outline_color, thickness)
    draw_text_with_outline(display_combined, "Ground Truth", (10, gt_start + 60), 
                          font, font_scale, text_color, outline_color, thickness)
    draw_text_with_outline(display_combined, "Prediction", (10, pred_start + 60), 
                          font, font_scale, text_color, outline_color, thickness)
    
    return display_combined

def calculate_metrics(predictions, groundtruths, threshold=0.5):
    """
    Calculate comprehensive metrics for all images in the dataset
    """
    if not predictions or not groundtruths:
        print("Warning: No prediction or ground truth data for metric calculation")
        return {}
    
    # Store per-image metrics
    per_image_metrics = []
    
    # Global counters for overall metrics
    total_tp = total_fp = total_tn = total_fn = 0
    
    # Calculate metrics for each image
    for idx, (pred, gt) in enumerate(zip(predictions, groundtruths)):
        # Convert tensors to numpy arrays if needed
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
        
        # Binarize prediction
        pred_binary = (pred > threshold).astype(np.uint8)
        gt_binary = (gt > 0.5).astype(np.uint8)
        
        # Calculate confusion matrix values
        tp = np.sum((pred_binary == 1) & (gt_binary == 1))
        tn = np.sum((pred_binary == 0) & (gt_binary == 0))
        fp = np.sum((pred_binary == 1) & (gt_binary == 0))
        fn = np.sum((pred_binary == 0) & (gt_binary == 1))
        
        # Accumulate for global metrics
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        
        # Calculate per-image metrics
        total_pixels = pred_binary.size
        correct_pixels = tp + tn
        accuracy = correct_pixels / total_pixels
        
        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0
        
        # IoU
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        
        # Class-specific accuracy
        crack_accuracy = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        non_crack_accuracy = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        
        # Store per-image metrics
        image_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou,
            'crack_accuracy': crack_accuracy,
            'non_crack_accuracy': non_crack_accuracy,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }
        per_image_metrics.append(image_metrics)
    
    # Calculate average metrics
    num_images = len(per_image_metrics)
    avg_metrics = {
        'accuracy': sum(m['accuracy'] for m in per_image_metrics) / num_images,
        'precision': sum(m['precision'] for m in per_image_metrics) / num_images,
        'recall': sum(m['recall'] for m in per_image_metrics) / num_images,
        'f1_score': sum(m['f1_score'] for m in per_image_metrics) / num_images,
        'iou': sum(m['iou'] for m in per_image_metrics) / num_images,
    }
    
    # Calculate global metrics from total pixel counts
    total_pixels = total_tp + total_fp + total_tn + total_fn
    global_metrics = {
        'accuracy': (total_tp + total_tn) / total_pixels,
        'precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
        'f1_score': 0,
        'iou': total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
    }
    
    # Calculate global F1
    if global_metrics['precision'] + global_metrics['recall'] > 0:
        global_metrics['f1_score'] = 2 * (global_metrics['precision'] * global_metrics['recall']) / (global_metrics['precision'] + global_metrics['recall'])
    
    # Calculate benchmark metrics
    thresholds = np.linspace(0.01, 0.99, 99)
    benchmark_metrics = calculate_pr_metrics_at_thresholds(predictions, groundtruths, thresholds)
    
    return {
        'per_image_metrics': per_image_metrics,
        'average_metrics': avg_metrics,
        'global_metrics': global_metrics,
        'total_confusion_matrix': {
            'TP': int(total_tp),
            'FP': int(total_fp),
            'TN': int(total_tn),
            'FN': int(total_fn)
        },
        'benchmark_metrics': benchmark_metrics,
        'total_images': num_images
    }

def calculate_pr_metrics_at_thresholds(predictions, groundtruths, thresholds):
    """
    Calculate ODS and OIS benchmark metrics for multiple thresholds
    
    Args:
        predictions: List of prediction arrays (values between 0-1)
        groundtruths: List of ground truth arrays (values between 0-1)
        thresholds: List of thresholds to evaluate
        
    Returns:
        dict: Dictionary containing ODS and OIS metrics
    """
    # Check if we have data to process
    if not predictions or not groundtruths:
        print("Warning: No prediction or ground truth data for benchmark calculation")
        return {
            'ODS': 0.0,
            'ODS_threshold': thresholds[0] if thresholds else 0.5,
            'OIS': 0.0
        }
        
    num_thresholds = len(thresholds)
    f_measures = np.zeros(num_thresholds)
    
    # For ODS: aggregate TP, FP, FN across all images
    total_tp = np.zeros(num_thresholds)
    total_fp = np.zeros(num_thresholds)
    total_fn = np.zeros(num_thresholds)
    
    # For OIS: store best F-measure for each image
    image_best_f_measures = []
    
    # For each image
    for pred, gt in zip(predictions, groundtruths):
        image_f_measures = np.zeros(num_thresholds)
        
        for i, threshold in enumerate(thresholds):
            # Binarize at this threshold
            pred_binary = (pred > threshold).astype(np.uint8)
            gt_binary = (gt > 0.5).astype(np.uint8)
            
            # Calculate metrics
            tp = np.sum((pred_binary == 1) & (gt_binary == 1))
            fp = np.sum((pred_binary == 1) & (gt_binary == 0))
            fn = np.sum((pred_binary == 0) & (gt_binary == 1))
            
            # Update for ODS calculation
            total_tp[i] += tp
            total_fp[i] += fp
            total_fn[i] += fn
            
            # Calculate F-measure for OIS
            image_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            image_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
            if image_precision + image_recall > 0:
                image_f = 2 * (image_precision * image_recall) / (image_precision + image_recall)
            else:
                image_f = 0
                
            image_f_measures[i] = image_f
        
        # Find best F-measure for this image (for OIS)
        best_f = np.max(image_f_measures)
        image_best_f_measures.append(best_f)
    
    # Calculate ODS F-measures
    for i in range(num_thresholds):
        precision = total_tp[i] / (total_tp[i] + total_fp[i]) if (total_tp[i] + total_fp[i]) > 0 else 0
        recall = total_tp[i] / (total_tp[i] + total_fn[i]) if (total_tp[i] + total_fn[i]) > 0 else 0
        
        if precision + recall > 0:
            f_measures[i] = 2 * (precision * recall) / (precision + recall)
        else:
            f_measures[i] = 0
    
    # Calculate OIS (average of best F-measures per image)
    ois = np.mean(image_best_f_measures)
    
    # Get ODS (optimal dataset scale - best F-measure across all thresholds)
    ods_idx = np.argmax(f_measures)
    ods_threshold = thresholds[ods_idx]
    ods = f_measures[ods_idx]
    
    return {
        'ODS': ods,
        'ODS_threshold': ods_threshold,
        'OIS': ois
    }


def write_evaluation_results(eval_file, metrics_results, timestamp, pretrained_model, timestamped_save_path):
    """
    Write comprehensive evaluation results to a text file
    """
    with open(eval_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"DeepCrack Full-Resolution Evaluation Results - {timestamp}\n")
        f.write(f"Model: {pretrained_model}\n")
        f.write(f"Images saved in: {timestamped_save_path}\n")
        f.write("=" * 80 + "\n\n")
        
        # Per-image metrics
        f.write("Per-Image Metrics:\n")
        f.write("-" * 40 + "\n")
        
        for idx, metrics in enumerate(metrics_results['per_image_metrics']):
            f.write(f"Image #{idx}:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1_score']:.4f}\n")
            f.write(f"  IoU: {metrics['iou']:.4f}\n")
            f.write(f"  Crack Pixel Accuracy: {metrics['crack_accuracy']:.4f}\n")
            f.write(f"  Non-Crack Pixel Accuracy: {metrics['non_crack_accuracy']:.4f}\n")
            f.write(f"  TP: {metrics['TP']}, FP: {metrics['FP']}, TN: {metrics['TN']}, FN: {metrics['FN']}\n\n")
        
        # Average metrics
        avg_metrics = metrics_results['average_metrics']
        f.write("Average Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Accuracy: {avg_metrics['accuracy']:.4f}\n")
        f.write(f"Average Precision: {avg_metrics['precision']:.4f}\n")
        f.write(f"Average Recall: {avg_metrics['recall']:.4f}\n")
        f.write(f"Average F1 Score: {avg_metrics['f1_score']:.4f}\n")
        f.write(f"Average IoU: {avg_metrics['iou']:.4f} (calculated from all {metrics_results['total_images']} images)\n")
        
        # Total confusion matrix
        cm = metrics_results['total_confusion_matrix']
        f.write(f"\nTotal Confusion Matrix:\n")
        f.write(f"  True Positives: {cm['TP']}\n")
        f.write(f"  False Positives: {cm['FP']}\n")
        f.write(f"  True Negatives: {cm['TN']}\n")
        f.write(f"  False Negatives: {cm['FN']}\n")
        
        # Global metrics
        global_metrics = metrics_results['global_metrics']
        f.write(f"\nGlobal Metrics (calculated from total pixel counts):\n")
        f.write(f"  Global Accuracy: {global_metrics['accuracy']:.4f}\n")
        f.write(f"  Global Precision: {global_metrics['precision']:.4f}\n")
        f.write(f"  Global Recall: {global_metrics['recall']:.4f}\n")
        f.write(f"  Global F1 Score: {global_metrics['f1_score']:.4f}\n")
        f.write(f"  Global IoU: {global_metrics['iou']:.4f}\n")
        
        # Benchmark metrics
        benchmark = metrics_results['benchmark_metrics']
        f.write("\nBoundary Detection Benchmark Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"ODS (Optimal Dataset Scale): {benchmark['ODS']:.4f} at threshold {benchmark['ODS_threshold']:.2f}\n")
        f.write(f"OIS (Optimal Image Scale): {benchmark['OIS']:.4f}\n")

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
    
    # Store predictions and ground truths
    all_predictions = []
    all_groundtruths = []
    
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
        
        # Perform inference (resizing handled inside sliding_window_inference)
        with torch.no_grad():
            pred_tensor = sliding_window_inference(model, img, window_size=448, overlap=0.0)
        
        # Resize ground truth to match prediction size
        pred_h, pred_w = pred_tensor.shape
        gt_resized = cv2.resize(gt, (pred_w, pred_h))
        gt_np = gt_resized.astype(np.float32)
        
        # Store results
        pred_np = pred_tensor.cpu().numpy()
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