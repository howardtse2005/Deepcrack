from data.dataset import CrackDataset
import data.preprocess_pipeline as pp
from training.unet_trainer import UNetTrainer
from training.deepcrack_trainer import DeepCrackTrainer
from training.loss import FocalWithLogitsLoss, DiceWithLogitsLoss, BCEWithLogitsLoss
from model.unet import UNet
from model.deepcrack import DeepCrack
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def main():
    # Initialize dataset
    train_dataset = CrackDataset(
        dataset_img_path="data/img_debug_tr",
        dataset_mask_path="data/mask_debug_tr",
        augmentations=[
            pp.Resize(target_size=(448, 448))
        ]
    )
    val_dataset = CrackDataset(
        dataset_img_path="data/img_debug_val",
        dataset_mask_path="data/mask_debug_val",
        augmentations=[
            pp.Resize(target_size=(448, 448))
        ]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Initialize model, optimizer, and loss function
    model = DeepCrack().to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # Define loss functions
    criterions = [
        FocalWithLogitsLoss(alpha=.25, gamma=2.0),
    ]
    
    # Initialize trainer
    trainer = DeepCrackTrainer(
        model=model,
        criterions=criterions,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        log_dir='test',
        chkp_dir='checkpoints',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()