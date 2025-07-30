from training.trainer import Trainer
import torch
import numpy as np
from training.loss import Loss

class DeepCrackTrainer(Trainer):
    """
    Trainer class for training DeepCrack model.
    Inherits from the base Trainer class.
    """
    
    def __init__(self, model, optimizer, criterions, train_loader, val_loader, 
                 log_dir, chkp_dir, scheduler=None, device='cpu', epoch_goal=10):
        """
        Initialize DeepCrackTrainer with same parameters as parent Trainer class.
        
        Args:
            model: DeepCrack model to be trained
            config: Configuration object containing training parameters
            optimizer: Optimizer for training
            criterions: List of loss functions
            train_loader: Training data loader
            val_loader: Validation data loader
            log_dir: Directory for logging
            scheduler: Learning rate scheduler (optional)
            device: Device to run training on
            epoch_goal: Total number of epochs to train
            epoch_trained: Number of epochs already trained
        """
        super().__init__(
            model=model,
            name='deepcrack',
            optimizer=optimizer,
            criterions=criterions,
            train_loader=train_loader,
            val_loader=val_loader,
            log_dir=log_dir,
            chkp_dir=chkp_dir,
            scheduler=scheduler,
            device=device,
            epoch_goal=epoch_goal,
        )

    def _calculate_loss(self, output, target):
        """
        Calculate loss for DeepCrack model.
        """
        pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1 = output
        loss_output = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_fuse5 = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_fuse4 = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_fuse3 = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_fuse2 = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_fuse1 = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        log_loss = {}
        for criterion in self.criterions:
            if isinstance(criterion, Loss):
                loss_output = loss_output + criterion(pred_output, target)
                loss_fuse5 = loss_fuse5 + criterion(pred_fuse5, target)
                loss_fuse4 = loss_fuse4 + criterion(pred_fuse4, target)
                loss_fuse3 = loss_fuse3 + criterion(pred_fuse3, target)
                loss_fuse2 = loss_fuse2 + criterion(pred_fuse2, target)
                loss_fuse1 = loss_fuse1 + criterion(pred_fuse1, target)
               
                log_loss[criterion.name] = loss_fuse1.item() + loss_fuse2.item() + \
                                           loss_fuse3.item() + loss_fuse4.item() + \
                                           loss_fuse5.item() + loss_output.item()
        loss_output = loss_output 
        loss_fuse5 = loss_fuse5 
        loss_fuse4 = loss_fuse4 
        loss_fuse3 = loss_fuse3 
        loss_fuse2 = loss_fuse2 
        loss_fuse1 = loss_fuse1 
        total_loss = (loss_output + loss_fuse1 + loss_fuse2 +
                      loss_fuse3 + loss_fuse4 + loss_fuse5) 
        
        log_loss['total_loss'] = total_loss.item()
        log_loss['loss_output'] = loss_output.item()
        log_loss['loss_fuse5'] = loss_fuse5.item()
        log_loss['loss_fuse4'] = loss_fuse4.item()
        log_loss['loss_fuse3'] = loss_fuse3.item()
        log_loss['loss_fuse2'] = loss_fuse2.item()
        log_loss['loss_fuse1'] = loss_fuse1.item()
        
        return total_loss, log_loss