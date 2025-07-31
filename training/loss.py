import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, outputs, targets):
        """
        Forward pass for the loss calculation.
        Implement this method in subclasses to define specific loss functions.
        """
        pass
    
class BCEWithLogitsLoss(Loss):
    def __init__(self, reduction: str = 'mean', weight: torch.Tensor = None, 
                 pos_weight: torch.Tensor = None):
        
        super().__init__(name="BCE Loss")
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction, weight=weight, pos_weight=pos_weight)
        
    def forward(self, outputs, targets):
        #input logits
        
        if outputs.dim() != 4 or (targets.dim() != 3 and targets.dim() != 4):
            raise ValueError(f"Outputs must be 4D (B, C, H, W) and targets must be 3D (B, H, W) or 4D (B, C, H, W), got outputs: {outputs.dim()} and targets: {targets.dim()} dimensions.")
       
        # makesure targets are 4D tensors
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # makesure outputs and targets follow (B, C, H, W) shape
        if outputs.shape != targets.shape:
            raise ValueError(f"Outputs and targets must have the same shape. got outputs: {outputs.shape} and targets: {targets.shape}.")
        
        loss = self.criterion(outputs, targets)
        return loss

class FocalWithLogitsLoss(Loss):
    def __init__(self, alpha=0.75, gamma=3.0,  reduction='mean'):
        super().__init__(name="Focal Loss")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        # makesure targets are 4D tensors
        if outputs.dim() != 4 or (targets.dim() != 3 and targets.dim() != 4):
            raise ValueError(f"Outputs must be 4D (B, C, H, W) and targets must be 3D (B, H, W) or 4D (B, C, H, W), got outputs: {outputs.dim()} and targets: {targets.dim()} dimensions.")
       
        # makesure targets are 4D tensors
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # makesure outputs and targets follow (B, C, H, W) shape
        if outputs.shape != targets.shape:
            raise ValueError(f"Outputs and targets must have the same shape. got outputs: {outputs.shape} and targets: {targets.shape}.")
        
        # For binary classification with logits
        if targets.shape[1] == 1:
            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)
            
            # Compute BCE loss manually to get proper focal weighting
            log_probs = torch.nn.functional.logsigmoid(outputs)
            log_probs_neg = torch.nn.functional.logsigmoid(-outputs)
            
            # pt is the probability of the correct class
            pt = torch.where(targets == 1, probs, 1 - probs)
            
            # Focal weight
            focal_weight = (1 - pt) ** self.gamma
            
            # Alpha weighting
            if 0 <= self.alpha <= 1:
                alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
                focal_weight = alpha_weight * focal_weight
            
            # Compute loss manually
            bce_loss = torch.where(targets == 1, -log_probs, -log_probs_neg)
            loss = focal_weight * bce_loss
            
        else:
            raise NotImplementedError("Multi-class focal loss not implemented yet")

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class DiceWithLogitsLoss(Loss):
    def __init__(self, smooth=1.0, reduction='mean'):
        super().__init__(name="Dice Loss")
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        # input logits
        
        if outputs.dim() != 4 or (targets.dim() != 3 and targets.dim() != 4):
            raise ValueError(f"Outputs must be 4D (B, C, H, W) and targets must be 3D (B, H, W) or 4D (B, C, H, W), got outputs: {outputs.dim()} and targets: {targets.dim()} dimensions.")
       
        # makesure targets are 4D tensors
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # makesure outputs and targets follow (B, C, H, W) shape
        if outputs.shape != targets.shape:
            raise ValueError(f"Outputs and targets must have the same shape. got outputs: {outputs.shape} and targets: {targets.shape}.")
        
        if targets.shape[1] == 1:  # Binary classification (single channel)
            outputs = torch.sigmoid(outputs)
            intersection = (outputs * targets).sum(dim=(2, 3))
            union = outputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        else:  # Multi-class segmentation (multiple channels)
            outputs = torch.softmax(outputs, dim=1)
            intersection = (outputs * targets).sum(dim=(2, 3))
            union = outputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
            
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == 'mean':
            return 1 - dice_score.mean()
        elif self.reduction == 'sum':
            return 1 - dice_score.sum()
        else:  # 'none'
            return 1 - dice_score