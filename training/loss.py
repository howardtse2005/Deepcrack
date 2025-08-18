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
        
        if targets.max() > 1 or targets.min() < 0:
            raise ValueError("Targets must be binary (0 or 1) .")
        
        targets = targets.float()  # Ensure targets are float for BCEWithLogitsLoss
        
        # makesure outputs and targets follow (B, C, H, W) shape
        if outputs.shape != targets.shape:
            raise ValueError(f"Outputs and targets must have the same shape. got outputs: {outputs.shape} and targets: {targets.shape}.")
        
        loss = self.criterion(outputs, targets)
        return loss

class FocalWithLogitsLoss(Loss):
    def __init__(self, alpha=0.75, gamma=3.0, reduction='mean'):
        super().__init__(name="Focal Loss")
        if alpha is not None and not (0 <= alpha <= 1):
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
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
            
        if targets.max() > 1 or targets.min() < 0:
            raise ValueError("Targets must be binary (0 or 1) .")
        
        # Ensure targets are float type
        targets = targets.float()
        
        # makesure outputs and targets follow (B, C, H, W) shape
        if outputs.shape != targets.shape:
            raise ValueError(f"Outputs and targets must have the same shape. got outputs: {outputs.shape} and targets: {targets.shape}.")
        
        # For binary classification with logits
        if targets.shape[1] == 1:
            probs = torch.sigmoid(outputs)

            # Binary cross entropy
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs, targets, reduction='none'
            )

            # Apply the focal term
            pt = torch.where(targets == 1, probs, 1 - probs)
            focal_weight = (1 - pt) ** self.gamma

            # Apply balanced alpha weighting:
            # For positive examples (targets==1): use alpha
            # For negative examples (targets==0): use (1-alpha)
            if 0 <= self.alpha <= 1:  # Ensure alpha is in valid range
                alpha_weight = torch.ones_like(targets) * (1 - self.alpha)  # Default for negative class
                alpha_weight = torch.where(targets == 1, torch.ones_like(targets) * self.alpha, alpha_weight)  # Set alpha for positive class
                focal_weight = alpha_weight * focal_weight

            loss = focal_weight * bce_loss

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
        
        if targets.max() > 1 or targets.min() < 0:
            raise ValueError("Targets must be binary (0 or 1) .")
        # Ensure targets are float type
        targets = targets.float()
        
        # makesure outputs and targets follow (B, C, H, W) shape
        if outputs.shape != targets.shape:
            raise ValueError(f"Outputs and targets must have the same shape. got outputs: {outputs.shape} and targets: {targets.shape}.")
        
        if targets.shape[1] == 1:  # Binary classification (single channel)
            outputs = torch.sigmoid(outputs)
            
            # Flatten for each sample in batch
            outputs_flat = outputs.view(outputs.size(0), -1)  # (B, H*W)
            targets_flat = targets.view(targets.size(0), -1)  # (B, H*W)
            
            intersection = (outputs_flat * targets_flat).sum(dim=1)  # (B,)
            union = outputs_flat.sum(dim=1) + targets_flat.sum(dim=1)  # (B,)
            
            dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
    
        else:  # Multi-class segmentation (multiple channels)
            outputs = torch.softmax(outputs, dim=1)
            
            # Calculate dice for each class and each sample
            outputs_flat = outputs.view(outputs.size(0), outputs.size(1), -1)  # (B, C, H*W)
            targets_flat = targets.view(targets.size(0), targets.size(1), -1)  # (B, C, H*W)
            
            intersection = (outputs_flat * targets_flat).sum(dim=2)  # (B, C)
            union = outputs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)
            
            dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_score = dice_score.mean(dim=1)  # Average across classes for each sample
            
        dice_loss = 1 - dice_score
    
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:  # 'none'
            return dice_loss