from data.dataset import readIndex, dataReadPip, loadedDataset
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


def sliding_window_inference(model, image, window_size=448, overlap=0.2):
    """
    Perform inference on large images using sliding window approach
    
    Args:
        model: Trained model
        image: Input image tensor (C, H, W)
        window_size: Size of sliding window (default 448)
        overlap: Overlap ratio between windows (default 0.2)
    
    Returns:
        prediction: Full resolution prediction tensor
    """
    device = next(model.parameters()).device
    C, H, W = image.shape
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


def calculate_metrics(pred, gt, threshold=0.5):
    """
    Calculate accuracy metrics between prediction and ground truth
    
    Args:
        pred: Prediction tensor or numpy array (values between 0-1)
        gt: Ground truth tensor or numpy array (values between 0-1)
        threshold: Threshold to binarize prediction
        
    Returns:
        dict: Dictionary containing various metrics
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
        
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    # Binarize prediction
    pred_binary = (pred > threshold).astype(np.uint8)
    gt_binary = (gt > 0.5).astype(np.uint8)
    
    # Calculate metrics
    metrics = {}
    
    # Total pixel accuracy
    total_pixels = pred_binary.size
    correct_pixels = np.sum(pred_binary == gt_binary)
    metrics['accuracy'] = correct_pixels / total_pixels
    
    # True positives, false positives, true negatives, false negatives
    true_positive = np.sum((pred_binary == 1) & (gt_binary == 1))
    true_negative = np.sum((pred_binary == 0) & (gt_binary == 0))
    false_positive = np.sum((pred_binary == 1) & (gt_binary == 0))
    false_negative = np.sum((pred_binary == 0) & (gt_binary == 1))
    
    metrics['TP'] = int(true_positive)
    metrics['TN'] = int(true_negative)
    metrics['FP'] = int(false_positive)
    metrics['FN'] = int(false_negative)
    
    # Calculate precision, recall, F1 score
    if true_positive + false_positive > 0:
        metrics['precision'] = true_positive / (true_positive + false_positive)
    else:
        metrics['precision'] = 0.0
        
    if true_positive + false_negative > 0:
        metrics['recall'] = true_positive / (true_positive + false_negative)
    else:
        metrics['recall'] = 0.0
        
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0.0
    
    # IoU (Intersection over Union) for positive class
    has_positive_pixels = true_positive + false_negative > 0
    
    if has_positive_pixels:
        metrics['iou'] = true_positive / (true_positive + false_positive + false_negative)
        metrics['has_positive_pixels'] = True  # Flag to indicate valid IoU
    else:
        metrics['iou'] = float('nan')  # Skip from IoU averaging if no positive pixels
        metrics['has_positive_pixels'] = False
    
    # Class-specific accuracy
    positive_pixels = true_positive + false_negative
    if positive_pixels > 0:
        metrics['crack_accuracy'] = true_positive / positive_pixels
    else:
        metrics['crack_accuracy'] = float('nan')
        
    negative_pixels = true_negative + false_positive
    if negative_pixels > 0:
        metrics['non_crack_accuracy'] = true_negative / negative_pixels
    else:
        metrics['non_crack_accuracy'] = float('nan')
    
    return metrics


def calculate_pr_metrics_at_thresholds(predictions, groundtruths, thresholds):
    """
    Calculate precision, recall, and F-measure for multiple thresholds
    
    Args:
        predictions: List of prediction arrays (values between 0-1)
        groundtruths: List of ground truth arrays (values between 0-1)
        thresholds: List of thresholds to evaluate
        
    Returns:
        tuple: Arrays of precision, recall, and F-measure at each threshold
    """
    # Check if we have data to process
    if not predictions or not groundtruths:
        print("Warning: No prediction or ground truth data for benchmark calculation")
        return {
            'precisions': np.zeros(len(thresholds)),
            'recalls': np.zeros(len(thresholds)),
            'f_measures': np.zeros(len(thresholds)),
            'ODS': 0.0,
            'ODS_threshold': thresholds[0],
            'OIS': 0.0,
            'AP': 0.0
        }
        
    num_thresholds = len(thresholds)
    precisions = np.zeros(num_thresholds)
    recalls = np.zeros(num_thresholds)
    f_measures = np.zeros(num_thresholds)
    
    # For ODS: aggregate TP, FP, FN across all images
    total_tp = np.zeros(num_thresholds)
    total_fp = np.zeros(num_thresholds)
    total_fn = np.zeros(num_thresholds)
    
    # For OIS: store best F-measure for each image
    image_best_f_measures = []
    
    # For each image
    for pred, gt in zip(predictions, groundtruths):
        # Calculate TP, FP, FN at each threshold for this image
        image_tp = np.zeros(num_thresholds)
        image_fp = np.zeros(num_thresholds)
        image_fn = np.zeros(num_thresholds)
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
            
            # Calculate for OIS
            if tp + fp > 0:
                image_precision = tp / (tp + fp)
            else:
                image_precision = 0
                
            if tp + fn > 0:
                image_recall = tp / (tp + fn)
            else:
                image_recall = 0
                
            if image_precision + image_recall > 0:
                image_f = 2 * (image_precision * image_recall) / (image_precision + image_recall)
            else:
                image_f = 0
                
            image_f_measures[i] = image_f
            
            image_tp[i] = tp
            image_fp[i] = fp
            image_fn[i] = fn
        
        # Find best F-measure for this image (for OIS)
        best_f = np.max(image_f_measures)
        image_best_f_measures.append(best_f)
    # Calculate ODS
    for i in range(num_thresholds):
        if total_tp[i] + total_fp[i] > 0:
            precisions[i] = total_tp[i] / (total_tp[i] + total_fp[i])
        else:
            precisions[i] = 0
            
        if total_tp[i] + total_fn[i] > 0:
            recalls[i] = total_tp[i] / (total_tp[i] + total_fn[i])
        else:
            recalls[i] = 0
            
        if precisions[i] + recalls[i] > 0:
            f_measures[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
        else:
            f_measures[i] = 0
    
    # Calculate OIS
    ois = np.mean(image_best_f_measures)
    
    # Calculate AP - Fixed implementation
    # Sort by increasing recall for proper AP calculation
    sorted_indices = np.argsort(recalls)
    sorted_recalls = recalls[sorted_indices]
    sorted_precisions = precisions[sorted_indices]
    
    # Remove duplicate recall values and interpolate precision
    unique_recalls = []
    interpolated_precisions = []
    
    for i in range(len(sorted_recalls)):
        if i == 0 or sorted_recalls[i] != sorted_recalls[i-1]:
            unique_recalls.append(sorted_recalls[i])
            # For this recall level, find the maximum precision at this recall or higher
            max_precision = 0
            for j in range(i, len(sorted_recalls)):
                if sorted_recalls[j] >= sorted_recalls[i]:
                    max_precision = max(max_precision, sorted_precisions[j])
            interpolated_precisions.append(max_precision)
    
    unique_recalls = np.array(unique_recalls)
    interpolated_precisions = np.array(interpolated_precisions)
    
    # Calculate AP using trapezoid rule on interpolated curve
    ap = 0
    for i in range(1, len(unique_recalls)):
        recall_diff = unique_recalls[i] - unique_recalls[i-1]
        avg_precision = (interpolated_precisions[i] + interpolated_precisions[i-1]) / 2
        ap += recall_diff * avg_precision
    
    # Alternative: Use standard AP calculation method
    # Start from recall=0 with precision=1 (if we have perfect precision at low recall)
    if len(unique_recalls) > 0 and unique_recalls[0] > 0:
        # Add point at (0, max_precision) if it doesn't exist
        ap += unique_recalls[0] * interpolated_precisions[0]
    
    # Get ODS (optimal dataset scale)
    ods_idx = np.argmax(f_measures)
    ods_threshold = thresholds[ods_idx]
    ods = f_measures[ods_idx]
    
    return {
        'precisions': precisions,
        'recalls': recalls,
        'f_measures': f_measures,
        'ODS': ods,
        'ODS_threshold': ods_threshold,
        'OIS': ois,
        'AP': ap,
        'debug_unique_recalls': unique_recalls,
        'debug_interpolated_precisions': interpolated_precisions
    }


def test(test_data_path='data/test_example.txt',
         save_path='deepcrack_results/images',
         eval_path='deepcrack_results/eval',
         pretrained_model='checkpoints/hnet4_july.pth',
         threshold=0.5):
    
    # Create timestamp for folder names
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories if they don't exist
    os.makedirs(eval_path, exist_ok=True)
    
    # Create a timestamped subfolder inside save_path
    result_folder_name = f"evaluation_{timestamp}"
    timestamped_save_path = os.path.join(save_path, result_folder_name)
    os.makedirs(timestamped_save_path, exist_ok=True)
    
    print(f"Results will be saved to: {timestamped_save_path}")
    
    # Read test data directly without dataset preprocessing
    test_list = readIndex(test_data_path)

    # -------------------- build trainer --------------------- #
    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()
    
    # Select model based on config
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
    
    # Select trainer based on config
    if cfg.model_type in ['hnet', 'unet', 'attention_unet', 'segformer']:
        trainer = UNetTrainer(model).to(device)
        print("Using UNetTrainer (single output)")
    else:
        trainer = DeepCrackTrainer(model).to(device)
        print("Using DeepCrackTrainer (multi-output)")

    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=True))
    model.eval()
    
    # Store metrics for all images
    all_metrics = []
    all_predictions = []
    all_groundtruths = []
    
    # Create evaluation file path with matching timestamp
    eval_file = os.path.join(eval_path, f"{result_folder_name}.txt")
    
    print("Processing full-resolution images with sliding window inference...")
    
    for idx, item in enumerate(tqdm(test_list)):
        img_path, gt_path = item
        
        # Load images - use color for input, grayscale for ground truth
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or gt is None:
            print(f"Failed to load image pair: {img_path}, {gt_path}")
            continue
        
        # Resize images to be multiples of 32 for model compatibility
        h, w = img.shape[:2]
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32
        
        img_resized = cv2.resize(img, (new_w, new_h))
        gt_resized = cv2.resize(gt, (new_w, new_h))
        
        # normalize
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        gt_tensor = torch.from_numpy(gt_resized).float() / 255.0
        
        print(f"Processing image {idx}: {img_tensor.shape[1]}x{img_tensor.shape[2]} pixels")
        
        # Perform sliding window inference
        with torch.no_grad():
            pred_tensor = sliding_window_inference(model, img_tensor, window_size=448, overlap=0.0)
        
        # Convert to numpy
        pred_np = pred_tensor.cpu().numpy()
        gt_np = gt_tensor.cpu().numpy()
        
        # Store for benchmark calculation
        all_predictions.append(pred_np)
        all_groundtruths.append(gt_np)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_np, gt_np, threshold=threshold)
        all_metrics.append(metrics)
        
        # Create visualization (downsample for display if too large)
        max_display_size = 1200
        h, w = pred_np.shape
        
        if max(h, w) > max_display_size:
            scale = max_display_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Use original color image for display
            display_img = cv2.resize(img, (new_w, new_h))
            display_pred = cv2.resize((pred_np * 255).astype(np.uint8), (new_w, new_h))
            display_gt = cv2.resize((gt_np * 255).astype(np.uint8), (new_w, new_h))
        else:
            display_img = img
            display_pred = (pred_np * 255).astype(np.uint8)
            display_gt = (gt_np * 255).astype(np.uint8)
        
        h_disp, w_disp = display_pred.shape
        
        # Add white border thickness
        border_thickness = 5
        
        # Create 3-panel display with white borders: Original (top), Ground Truth (middle), Prediction (bottom)
        total_height = h_disp * 3 + border_thickness * 2  # Add space for 2 borders
        display_combined = np.ones((total_height, w_disp, 3), dtype=np.uint8) * 255  # Initialize with white
        
        # Top panel: original image
        display_combined[:h_disp, :, :] = display_img
        
        # White border between original and ground truth (already white from initialization)
        
        # Middle panel: ground truth (starting after first border)
        gt_start = h_disp + border_thickness
        gt_end = gt_start + h_disp
        display_combined[gt_start:gt_end, :, 0] = display_gt
        display_combined[gt_start:gt_end, :, 1] = display_gt
        display_combined[gt_start:gt_end, :, 2] = display_gt
        
        # White border between ground truth and prediction (already white from initialization)
        
        # Bottom panel: prediction (starting after second border)
        pred_start = gt_end + border_thickness
        pred_end = pred_start + h_disp
        display_combined[pred_start:pred_end, :, 0] = display_pred
        display_combined[pred_start:pred_end, :, 1] = display_pred
        display_combined[pred_start:pred_end, :, 2] = display_pred
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)
        thickness = 6
        
        # Draw labels (adjust positions for borders)
        draw_text_with_outline(display_combined, "Original Image", (10, 60), 
                              font, font_scale, text_color, outline_color, thickness)
        
        draw_text_with_outline(display_combined, "Ground Truth", (10, gt_start + 60), 
                              font, font_scale, text_color, outline_color, thickness)
        
        draw_text_with_outline(display_combined, "Prediction", (10, pred_start + 60), 
                              font, font_scale, text_color, outline_color, thickness)
        
        # Add metrics (adjust position for borders)
        acc_text = f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, IoU: {metrics['iou']:.4f}"
        draw_text_with_outline(display_combined, acc_text, (10, pred_start + 120), 
                              font, font_scale * 0.8, text_color, outline_color, thickness)
        
        # Save image
        save_name = os.path.join(timestamped_save_path, f"fullres_{idx:04d}.png")
        cv2.imwrite(save_name, display_combined)
    
    # Calculate benchmark metrics (ODS, OIS, AP)
    print(f"Collected {len(all_predictions)} images for benchmark evaluation")
    thresholds = np.linspace(0.01, 0.99, 99)
    benchmark_metrics = calculate_pr_metrics_at_thresholds(all_predictions, all_groundtruths, thresholds)
    
    # Calculate and save average metrics
    with open(eval_file, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write(f"DeepCrack Full-Resolution Evaluation Results - {timestamp}\n")
        f.write(f"Model: {pretrained_model}\n")
        f.write(f"Images saved in: {timestamped_save_path}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write per-image metrics
        f.write("Per-Image Metrics:\n")
        f.write("-" * 40 + "\n")
        
        for idx, metrics in enumerate(all_metrics):
            f.write(f"Image #{idx}:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1_score']:.4f}\n")
            f.write(f"  IoU: {metrics['iou']:.4f}\n")
            f.write(f"  Crack Pixel Accuracy: {metrics['crack_accuracy']:.4f}\n")
            f.write(f"  Non-Crack Pixel Accuracy: {metrics['non_crack_accuracy']:.4f}\n")
            f.write(f"  TP: {metrics['TP']}, FP: {metrics['FP']}, TN: {metrics['TN']}, FN: {metrics['FN']}\n\n")
          # Calculate and write average metrics
        f.write("\nAverage Metrics:\n")
        f.write("-" * 40 + "\n")
        
        # Calculate average IoU only for images with positive pixels
        valid_iou_scores = [m['iou'] for m in all_metrics if m['has_positive_pixels'] and not np.isnan(m['iou'])]
        num_images_with_cracks = len(valid_iou_scores)
        avg_iou = sum(valid_iou_scores) / num_images_with_cracks if num_images_with_cracks > 0 else 0.0
        
        avg_metrics = {
            'accuracy': sum(m['accuracy'] for m in all_metrics) / len(all_metrics),
            'precision': sum(m['precision'] for m in all_metrics) / len(all_metrics),
            'recall': sum(m['recall'] for m in all_metrics) / len(all_metrics),
            'f1_score': sum(m['f1_score'] for m in all_metrics) / len(all_metrics),
            'iou': avg_iou,
        }
        
        f.write(f"Average Accuracy: {avg_metrics['accuracy']:.4f}\n")
        f.write(f"Average Precision: {avg_metrics['precision']:.4f}\n")
        f.write(f"Average Recall: {avg_metrics['recall']:.4f}\n")
        f.write(f"Average F1 Score: {avg_metrics['f1_score']:.4f}\n")
        f.write(f"Average IoU: {avg_metrics['iou']:.4f} (calculated from {num_images_with_cracks} images with cracks out of {len(all_metrics)} total images)\n")
        
        # Calculate total confusion matrix values
        total_tp = sum(m['TP'] for m in all_metrics)
        total_fp = sum(m['FP'] for m in all_metrics)
        total_tn = sum(m['TN'] for m in all_metrics)
        total_fn = sum(m['FN'] for m in all_metrics)
        
        f.write(f"\nTotal Confusion Matrix:\n")
        f.write(f"  True Positives: {total_tp}\n")
        f.write(f"  False Positives: {total_fp}\n")
        f.write(f"  True Negatives: {total_tn}\n")
        f.write(f"  False Negatives: {total_fn}\n")
        
        # Calculate global metrics from totals
        total_pixels = total_tp + total_fp + total_tn + total_fn
        global_accuracy = (total_tp + total_tn) / total_pixels
        global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
        global_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        
        f.write(f"\nGlobal Metrics (calculated from total pixel counts):\n")
        f.write(f"  Global Accuracy: {global_accuracy:.4f}\n")
        f.write(f"  Global Precision: {global_precision:.4f}\n")
        f.write(f"  Global Recall: {global_recall:.4f}\n")
        f.write(f"  Global F1 Score: {global_f1:.4f}\n")
        f.write(f"  Global IoU: {global_iou:.4f}\n")
          # Add benchmark metrics section (remove AUC section)
        f.write("\nBoundary Detection Benchmark Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"ODS (Optimal Dataset Scale): {benchmark_metrics['ODS']:.4f} at threshold {benchmark_metrics['ODS_threshold']:.2f}\n")
        f.write(f"OIS (Optimal Image Scale): {benchmark_metrics['OIS']:.4f}\n")
    
    print(f"Evaluation results saved to {eval_file}")
    print(f"Images saved to {timestamped_save_path}")


if __name__ == '__main__':
    test()