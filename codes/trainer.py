from torch import nn
from tools.tensorboard_logger import TensorBoardLogger
from tools.checkpointer import Checkpointer
from config import Config as cfg
import torch


def get_optimizer(model):
    if cfg.use_adam:
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum, )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, beta=1.0, reduction='mean'):
        """
        Focal Loss for dense pixel-wise binary classification

        Args:
            alpha: Weight factor for class balance. Must be in [0,1].
                  alpha is for positive class (crack pixels),
                  (1-alpha) is for negative class (background pixels)
            gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted
            beta: Global scaling factor for the focal term
            reduction: 'mean', 'sum' or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, inputs, targets):
        # Apply sigmoid to raw logits
        probs = torch.sigmoid(inputs)

        # Binary cross entropy
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Apply the focal term
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.beta * ((1 - pt) ** self.gamma)

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


class DeepCrackTrainer(nn.Module):
    def __init__(self, model):
        super(DeepCrackTrainer, self).__init__()
        self.vis = TensorBoardLogger(log_dir='runs/deepcrack', exp_name=cfg.name)
        self.model = model

        self.saver = Checkpointer(cfg.name, cfg.saver_path, overwrite=False, verbose=True, timestamp=True,
                                  max_queue=cfg.max_save)

        self.optimizer = get_optimizer(self.model)

        self.iter_counter = 0

        # -------------------- Loss --------------------- #

        # Use Focal Loss if enabled in config
        if cfg.use_focal_loss:
            self.mask_loss = FocalLoss(
                alpha=cfg.focal_alpha,
                gamma=cfg.focal_gamma,
                beta=cfg.focal_beta,
                reduction='mean'
            )
        else:
            # Original BCE loss with pos_weight
            self.mask_loss = nn.BCEWithLogitsLoss(
                reduction='mean',
                pos_weight=torch.cuda.FloatTensor([cfg.pos_pixel_weight])
            )

        self.log_loss = {}
        self.log_acc = {}

    def train_op(self, input, target):
        self.optimizer.zero_grad()

        pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1, = self.model(input)
    
        output_loss = self.mask_loss(pred_output.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse5_loss = self.mask_loss(pred_fuse5.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse4_loss = self.mask_loss(pred_fuse4.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse3_loss = self.mask_loss(pred_fuse3.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse2_loss = self.mask_loss(pred_fuse2.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse1_loss = self.mask_loss(pred_fuse1.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size

        total_loss = output_loss + fuse5_loss + fuse4_loss + fuse3_loss + fuse2_loss + fuse1_loss
        total_loss.backward()
        self.optimizer.step()

        self.iter_counter += 1
        self.vis.set_step(self.iter_counter)  # Update step for TensorBoard

        self.log_loss = {
            'total_loss': total_loss.item(),
            'output_loss': output_loss.item(),
            'fuse5_loss': fuse5_loss.item(),
            'fuse4_loss': fuse4_loss.item(),
            'fuse3_loss': fuse3_loss.item(),
            'fuse2_loss': fuse2_loss.item(),
            'fuse1_loss': fuse1_loss.item()
        }

        return pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1,

    def val_op(self, input, target):
        pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1, = self.model(input)

        output_loss = self.mask_loss(pred_output.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse5_loss = self.mask_loss(pred_fuse5.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse4_loss = self.mask_loss(pred_fuse4.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse3_loss = self.mask_loss(pred_fuse3.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse2_loss = self.mask_loss(pred_fuse2.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse1_loss = self.mask_loss(pred_fuse1.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size

        total_loss = output_loss + fuse5_loss + fuse4_loss + fuse3_loss + fuse2_loss + fuse1_loss

        self.log_loss = {
            'total_loss': total_loss.item(),
            'output_loss': output_loss.item(),
            'fuse5_loss': fuse5_loss.item(),
            'fuse4_loss': fuse4_loss.item(),
            'fuse3_loss': fuse3_loss.item(),
            'fuse2_loss': fuse2_loss.item(),
            'fuse1_loss': fuse1_loss.item()
        }

        return pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1,

    def acc_op(self, pred, target):
        mask = target

        pred = pred
        pred[pred > cfg.acc_sigmoid_th] = 1
        pred[pred <= cfg.acc_sigmoid_th] = 0

        pred_mask = pred[:, 0, :, :].contiguous()

        # Calculate overall accuracy
        mask_acc = pred_mask.eq(mask.view_as(pred_mask)).sum().item() / max(mask.numel(), 1)

        # Check if there are any positive pixels before division
        pos_pixels = mask[mask > 0].numel()
        if pos_pixels > 0:
            mask_pos_acc = pred_mask[mask > 0].eq(mask[mask > 0].view_as(pred_mask[mask > 0])).sum().item() / pos_pixels
        else:
            mask_pos_acc = float('nan')  # Use NaN to indicate no positive pixels

        # Check if there are any negative pixels before division
        neg_pixels = mask[mask < 1].numel()
        if neg_pixels > 0:
            mask_neg_acc = pred_mask[mask < 1].eq(mask[mask < 1].view_as(pred_mask[mask < 1])).sum().item() / neg_pixels
        else:
            mask_neg_acc = float('nan')  # Use NaN to indicate no negative pixels

        self.log_acc = {
            'mask_acc': mask_acc,
            'mask_pos_acc': mask_pos_acc,
            'mask_neg_acc': mask_neg_acc,
        }
    
    def export_loss_curves(self, epoch=None):
        """Export loss curves as JPG."""
        if epoch is not None:
            filename = f'loss_curves_epoch{epoch}.jpg'
        else:
            filename = None
        return self.vis.export_loss_curves(filename)