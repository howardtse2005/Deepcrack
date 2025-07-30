import numpy as np
import torch

#--------------------- Benchmark Metric Calculation Functions ---------------------

def calculate_accuracy(prediction, ground_truth, threshold=0.5):
    """Calculate accuracy for a single image
    Args:
        prediction: Prediction array (H, W) with values 0-1
        ground_truth: Ground truth array (H, W) with values 0-1
    """
    pred_binary = (prediction > threshold).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    tn = np.sum((pred_binary == 0) & (gt_binary == 0))
    
    return (tp + tn) / pred_binary.size

def calculate_f1(prediction, ground_truth, threshold=0.5):
    """Calculate F1 score for a single image
    Args:
        prediction: Prediction array (H, W) with values 0-1
        ground_truth: Ground truth array (H, W) with values 0-1
    """
    pred_binary = (prediction > threshold).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0

def calculate_iou(prediction, ground_truth, threshold=0.5):
    """Calculate IoU for a single image
    Args:
        prediction: Prediction array (H, W) with values 0-1
        ground_truth: Ground truth array (H, W) with values 0-1
    """
    pred_binary = (prediction > threshold).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0

def calculate_precision(prediction, ground_truth, threshold=0.5):
    """Calculate precision for a single image
    Args:
        prediction: Prediction array (H, W) with values 0-1
        ground_truth: Ground truth array (H, W) with values 0-1
    """
    pred_binary = (prediction > threshold).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    
    return tp / (tp + fp) if (tp + fp) > 0 else 1.0

def calculate_recall(prediction, ground_truth, threshold=0.5):
    """Calculate recall for a single image
    Args:
        prediction: Prediction array (H, W) with values 0-1
        ground_truth: Ground truth array (H, W) with values 0-1
    """
    pred_binary = (prediction > threshold).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    
    return tp / (tp + fn) if (tp + fn) > 0 else 1.0

def calculate_confusion_matrix(prediction, ground_truth, threshold=0.5):
    """Calculate confusion matrix values for a single image
    Args:
        prediction: Prediction array (H, W) with values 0-1
        ground_truth: Ground truth array (H, W) with values 0-1
    """
    pred_binary = (prediction > threshold).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    tn = np.sum((pred_binary == 0) & (gt_binary == 0))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))

    return {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}

# Calculate thresold-based metrics
def calculate_metrics(predictions, groundtruths, threshold=0.5):
    """
    Calculate comprehensive metrics for all images in the dataset using individual metric functions
    Args:
        predictions: List of prediction arrays, each (H, W) with values 0-1
        groundtruths: List of ground truth arrays, each (H, W) with values 0-1
    """
    if not predictions or not groundtruths:
        print("Warning: No prediction or ground truth data for metric calculation")
        return {}
    
    # Store per-image metrics
    per_image_metrics = []
    
    # Global counters for overall metrics
    total_tp = total_fp = total_tn = total_fn = 0
    
    # Calculate metrics for each image using individual functions
    for idx, (pred, gt) in enumerate(zip(predictions, groundtruths)):
        # Convert tensors to numpy arrays if needed
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
        
        # Use individual metric functions
        accuracy = calculate_accuracy(pred, gt, threshold)
        precision = calculate_precision(pred, gt, threshold)
        recall = calculate_recall(pred, gt, threshold)
        f1_score = calculate_f1(pred, gt, threshold)
        iou = calculate_iou(pred, gt, threshold)
        confusion = calculate_confusion_matrix(pred, gt, threshold)
        
        # Accumulate for global metrics
        total_tp += confusion['TP']
        total_fp += confusion['FP']
        total_tn += confusion['TN']
        total_fn += confusion['FN']
        
        # Store per-image metrics
        image_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou,
            'TP': confusion['TP'],
            'TN': confusion['TN'],
            'FP': confusion['FP'],
            'FN': confusion['FN']
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

# Calculate ODS and OIS metrics for multiple thresholds
def calculate_pr_metrics_at_thresholds(predictions, groundtruths, thresholds):
    """
    Calculate ODS and OIS benchmark metrics for multiple thresholds
    
    Args:
        predictions: List of prediction arrays, each (H, W) with values 0-1
        groundtruths: List of ground truth arrays, each (H, W) with values 0-1
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
