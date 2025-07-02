from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from model.hnet import HNet  # Need to import HNet model
from model.unet import UNet  # Need to import UNet model
from trainer import DeepCrackTrainer, UNetTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
import datetime
from config import Config as cfg

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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
    if true_positive + false_positive + false_negative > 0:
        metrics['iou'] = true_positive / (true_positive + false_positive + false_negative)
    else:
        metrics['iou'] = 0.0
    
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
    
    # Calculate AP
    # Sort by increasing recall
    sorted_indices = np.argsort(recalls)
    sorted_recalls = recalls[sorted_indices]
    sorted_precisions = precisions[sorted_indices]
    
    # Interpolate precision to handle zigzag effect
    for i in range(len(sorted_precisions) - 2, -1, -1):
        sorted_precisions[i] = max(sorted_precisions[i], sorted_precisions[i + 1])
    
    # AP calculation with trapezoid rule
    ap = 0
    for i in range(1, len(sorted_recalls)):
        ap += (sorted_recalls[i] - sorted_recalls[i-1]) * sorted_precisions[i]
    
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
        'AP': ap
    }


def test(test_data_path='data/test_example.txt',
         save_path='deepcrack_results/images',
         eval_path='deepcrack_results/eval',
         pretrained_model='checkpoints/testtesttest.pth',
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
    
    test_pipline = dataReadPip(transforms=None, crop=True) # Set crop=True if you want to perform random cropping in the test

    test_list = readIndex(test_data_path)

    test_dataset = loadedDataset(test_list, preprocess=test_pipline)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=1, drop_last=False)

    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()    # Select model based on config
    if cfg.model_type == 'hnet':
        model = HNet()
        print("Using HNet architecture")
    elif cfg.model_type == 'unet':
        model = UNet()
        print("Using UNet architecture")
    else:
        model = DeepCrack()
        print("Using DeepCrack architecture")
        
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    # Select trainer based on config
    if cfg.model_type in ['hnet', 'unet']:
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
    
    with torch.no_grad():
        for idx, (img, lab) in enumerate(tqdm(test_loader)):
            test_data, test_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
            test_pred = trainer.val_op(test_data, test_target)
            
            # Handle different model types
            if cfg.model_type in ['hnet', 'unet']:
                # Single output model
                test_pred = torch.sigmoid(test_pred.cpu().squeeze())
            else:
                # Multi-output model (DeepCrack) - use main output
                test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
            
            # Convert prediction and ground truth to numpy arrays
            pred_np = test_pred.numpy()
            gt_np = lab.cpu().squeeze().numpy()
            
            # Store predictions and ground truths for benchmark calculation
            all_predictions.append(pred_np)
            all_groundtruths.append(gt_np)
            
            # Calculate metrics with specified threshold
            metrics = calculate_metrics(pred_np, gt_np, threshold=threshold)
            all_metrics.append(metrics)
            
            # Add accuracy to the saved image
            pred_img = (pred_np * 255).astype(np.uint8)
            gt_img = (gt_np * 255).astype(np.uint8)
            
            # Convert input RGB image from tensor (C,H,W) to numpy (H,W,C)
            rgb_np = img.cpu().squeeze().permute(1, 2, 0).numpy()
            # Scale from [0,1] to [0,255] and convert to uint8
            rgb_np = (rgb_np * 255).astype(np.uint8)
            # Convert from RGB to BGR for OpenCV
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            
            # Create a 3-panel display image with prediction, ground truth, and original
            h, w = pred_img.shape
            
            # Create 3 horizontal panels
            display_img = np.zeros((h * 3, w, 3), dtype=np.uint8)
            
            # Top panel: prediction (convert to BGR for color visualization)
            display_img[:h, :, 0] = pred_img  # All channels get same value for grayscale
            display_img[:h, :, 1] = pred_img
            display_img[:h, :, 2] = pred_img
            
            # Middle panel: ground truth
            display_img[h:2*h, :, 0] = gt_img
            display_img[h:2*h, :, 1] = gt_img
            display_img[h:2*h, :, 2] = gt_img
            
            # Bottom panel: original RGB image
            display_img[2*h:, :, :] = rgb_np
            
            # Add labels with black outline to identify each panel
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            text_color = (255, 255, 255)  # White text
            outline_color = (0, 0, 0)      # Black outline
            thickness = 2
            
            # Draw text with outline for better visibility
            draw_text_with_outline(display_img, "Prediction", (10, 30), 
                                  font, font_scale, text_color, outline_color, thickness)
            
            # Add accuracy metrics to prediction panel
            acc_text = f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, IoU: {metrics['iou']:.4f}"
            draw_text_with_outline(display_img, acc_text, (10, 70), 
                                  font, font_scale * 0.8, text_color, outline_color, thickness)
            
            draw_text_with_outline(display_img, "Ground Truth", (10, h + 30), 
                                  font, font_scale, text_color, outline_color, thickness)
            
            draw_text_with_outline(display_img, "Original Image", (10, 2*h + 30), 
                                  font, font_scale, text_color, outline_color, thickness)
            
            # Generate a unique filename for each crop in the timestamped folder
            save_name = os.path.join(timestamped_save_path, f"crop_{idx:04d}.png")
            cv2.imwrite(save_name, display_img)
    
    # Calculate benchmark metrics (ODS, OIS, AP)
    print(f"Collected {len(all_predictions)} images for benchmark evaluation")
    thresholds = np.linspace(0.01, 0.99, 99)  # 99 thresholds from 0.01 to 0.99
    benchmark_metrics = calculate_pr_metrics_at_thresholds(all_predictions, all_groundtruths, thresholds)
    
    # Calculate and save average metrics
    with open(eval_file, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write(f"DeepCrack Evaluation Results - {timestamp}\n")
        f.write(f"Model: {pretrained_model}\n")
        f.write(f"Images saved in: {timestamped_save_path}\n")  # Add image path info
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
        
        avg_metrics = {
            'accuracy': sum(m['accuracy'] for m in all_metrics) / len(all_metrics),
            'precision': sum(m['precision'] for m in all_metrics) / len(all_metrics),
            'recall': sum(m['recall'] for m in all_metrics) / len(all_metrics),
            'f1_score': sum(m['f1_score'] for m in all_metrics) / len(all_metrics),
            'iou': sum(m['iou'] for m in all_metrics) / len(all_metrics),
            # Skip crack/non-crack accuracy as some might be NaN
        }
        
        f.write(f"Average Accuracy: {avg_metrics['accuracy']:.4f}\n")
        f.write(f"Average Precision: {avg_metrics['precision']:.4f}\n")
        f.write(f"Average Recall: {avg_metrics['recall']:.4f}\n")
        f.write(f"Average F1 Score: {avg_metrics['f1_score']:.4f}\n")
        f.write(f"Average IoU: {avg_metrics['iou']:.4f}\n")
        
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
        
        # Add benchmark metrics section
        f.write("\nBoundary Detection Benchmark Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"ODS (Optimal Dataset Scale): {benchmark_metrics['ODS']:.4f} at threshold {benchmark_metrics['ODS_threshold']:.2f}\n")
        f.write(f"OIS (Optimal Image Scale): {benchmark_metrics['OIS']:.4f}\n")
        f.write(f"AP (Average Precision): {benchmark_metrics['AP']:.4f}\n")
    
    print(f"Evaluation results saved to {eval_file}")
    print(f"Images saved to {timestamped_save_path}")
    print(f"Average accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"Average F1-Score: {avg_metrics['f1_score']:.4f}")
    print(f"Average IoU: {avg_metrics['iou']:.4f}")
    print(f"Benchmark metrics:")
    print(f"ODS: {benchmark_metrics['ODS']:.4f}")
    print(f"OIS: {benchmark_metrics['OIS']:.4f}")
    print(f"AP: {benchmark_metrics['AP']:.4f}")


if __name__ == '__main__':
    test()
