from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
import datetime

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


def test(test_data_path='data/test_example.txt', # Path to the test data files
         save_path='deepcrack_results/images', # Path to the test results directory
         eval_path='deepcrack_results/eval', # Path to save evaluation metrics
         pretrained_model='checkpoints/test_good_12.pth', # Change this to the path of your pth file
         threshold=0.1):  # Add threshold parameter with reasonable default
    
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
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()

    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)

    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=True))

    model.eval()
    
    # Store metrics for all images
    all_metrics = []
    
    # Create evaluation file path with matching timestamp
    eval_file = os.path.join(eval_path, f"{result_folder_name}.txt")

    with torch.no_grad():
        for idx, (img, lab) in enumerate(tqdm(test_loader)):
            test_data, test_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
            test_pred = trainer.val_op(test_data, test_target)
            test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
            
            # Convert prediction and ground truth to numpy arrays
            pred_np = test_pred.numpy()
            gt_np = lab.cpu().squeeze().numpy()
            
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
    
    print(f"Evaluation results saved to {eval_file}")
    print(f"Images saved to {timestamped_save_path}")
    print(f"Average accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"Average F1-Score: {avg_metrics['f1_score']:.4f}")
    print(f"Average IoU: {avg_metrics['iou']:.4f}")


if __name__ == '__main__':
    test()
