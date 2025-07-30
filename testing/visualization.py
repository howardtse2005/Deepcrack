import cv2
import numpy as np
from benchmark import calculate_accuracy, calculate_f1, calculate_iou

#--------------------- Visualization and Evaluation Functions ---------------------

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
    Args:
        original_img: Original image array (H, W, C)
        ground_truth: Ground truth array (H, W) with values 0-1
        prediction: Prediction array (H, W) with values 0-1
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
    
    # Add metrics (adjust position for borders)
    accuracy = calculate_accuracy(prediction, ground_truth)
    f1_score = calculate_f1(prediction, ground_truth)
    iou = calculate_iou(prediction, ground_truth)
    acc_text = f"Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}, IoU: {iou:.4f}"
    draw_text_with_outline(display_combined, acc_text, (10, pred_start + 120),
                            font, font_scale * 0.8, text_color, outline_color, thickness)
    
    return display_combined

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
