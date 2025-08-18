import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR

import data.preprocess_pipeline as pp
from data.dataset import CrackDataset
from config import Config as cfg
from model.attention_unet import AttentionUNet
from model.segformer import SegFormer
from model.hnet import HNet
from model.deepcrack import DeepCrack
from model.unet import UNet
from training.loss import FocalWithLogitsLoss, DiceWithLogitsLoss, BCEWithLogitsLoss
from training.trainer_deepcrack import DeepCrackTrainer
from training.trainer_unet import UNetTrainer
from training.trainer_hnet import HNetTrainer
from training.trainer_segformer import SegFormerTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#<---------Initialize model ------------>
if cfg.model_type == 'deepcrack':
    print("Using DeepCrack model")
    model = DeepCrack()
elif cfg.model_type == 'unet':
    print("Using UNet model")
    model = UNet()
elif cfg.model_type == 'attention_unet':
    print("Using Attention UNet model")
    model = AttentionUNet()
elif cfg.model_type == 'segformer':
    print("Using SegFormer model")
    model = SegFormer(variant=cfg.segformer_variant) 
elif cfg.model_type == 'hnet':
    print("Using HNet model")
    model = HNet()
else:
    raise ValueError(f"Unknown model type: {cfg.model_type}")

if cfg.use_checkpoint:
    print(f"Loading pretrained model from: {cfg.pretrained_model}")
    model.load_state_dict(torch.load(cfg.pretrained_model, map_location=device))
    
model.to(device)

#<---------Initialize optimizer ------------>
if cfg.optimizer == 'adam':
    print("Using Adam optimizer")
    optimizer = optim.Adam(model.parameters(), **cfg.adam_params)
elif cfg.optimizer == 'sgd':
    print("Using SGD optimizer")
    optimizer = optim.SGD(model.parameters(), **cfg.sgd_params)
elif cfg.optimizer == 'rmsprop':
    print("Using RMSprop optimizer")
    optimizer = optim.RMSprop(model.parameters(), **cfg.rmsprop_params)
else:
    raise ValueError(f"Unknown optimizer type: {cfg.optimizer}")
    
#<---------Initialize scheduler ------------>
scheduler = None
if cfg.scheduler == 'step':
    print("Using StepLR scheduler")
    scheduler = StepLR(optimizer, **cfg.step_params)
elif cfg.scheduler == 'plateau':
    print("Using ReduceLROnPlateau scheduler")
    scheduler = ReduceLROnPlateau(optimizer,**cfg.plateau_params)
elif cfg.scheduler == 'cosine':
    print("Using CosineAnnealingLR scheduler")
    scheduler = CosineAnnealingLR(optimizer,**cfg.cosine_params)
elif cfg.scheduler == 'none':
    print("No scheduler will be used")
    scheduler = None
else:
    raise ValueError(f"Unknown scheduler type: {cfg.scheduler}")

#<---------Initialize loss functions ------------>
criterions = []
if 'focal' in cfg.criterions:
    print("Using Focal loss")
    criterions.append(FocalWithLogitsLoss(**cfg.focal_params))
if 'dice' in cfg.criterions:
    print("Using Dice loss")
    criterions.append(DiceWithLogitsLoss(**cfg.dice_params))
if 'bce' in cfg.criterions:
    print("Using BCE loss")
    criterions.append(BCEWithLogitsLoss(**cfg.bce_params))

#<---------Initialize data loaders ------------>
augmentations_train = []
if cfg.train_random_crop:
    augmentations_train.append(pp.Crop(range_crop_len=cfg.crop_range,
                                 n_copy=cfg.num_crops,
                                 each_has_crack=cfg.p_hascrack,
                                 ))
    
if cfg.train_random_rotate:
    augmentations_train.append(pp.RandomRotate())
    
if cfg.train_random_jitter:
    augmentations_train.append(pp.RandomJitter())
    
if cfg.train_random_gaussian_noise:
    augmentations_train.append(pp.RandomGaussianNoise())
    
augmentations_train.append(pp.Resize(target_size=cfg.target_size))
train_dataset = CrackDataset(
    dataset_img_path=cfg.dir_img_tr,
    dataset_mask_path=cfg.dir_mask_tr,
    augmentations=augmentations_train,
    temp_dir=cfg.dir_temp_tr,
    keep_temp=cfg.keep_temp_tr
)
print(f"Dataset length: {len(train_dataset)}")
print(f"Number of images: {train_dataset.get_num_imgs()}")

augmentations_val = []
if cfg.val_random_crop:
    augmentations_val.append(pp.Crop(range_crop_len=cfg.crop_range,
                                 n_copy=cfg.num_crops,
                                 each_has_crack=cfg.p_hascrack,
                                 ))
augmentations_val.append(pp.Resize(target_size=cfg.target_size))
val_dataset = CrackDataset(
    dataset_img_path=cfg.dir_img_val,
    dataset_mask_path=cfg.dir_mask_val,
    augmentations=augmentations_val,
    temp_dir=cfg.dir_temp_val,
    keep_temp=cfg.keep_temp_val
)

print(f"Validation dataset length: {len(val_dataset)}")
print(f"Number of validation images: {val_dataset.get_num_imgs()}")

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=1)

#<---------Initialize trainer ------------>
trainer_params = {
    'model': model,
    'optimizer': optimizer,
    'criterions': criterions,
    'train_loader': train_loader,
    'val_loader': val_loader,
    'log_dir': cfg.log_path,
    'chkp_dir': cfg.checkpoint_path,
    'scheduler': scheduler,
    'device': device,
    'epoch_goal': cfg.epoch
}
if cfg.model_type == 'deepcrack':
    trainer = DeepCrackTrainer(**trainer_params)
elif cfg.model_type == 'unet':
    trainer = UNetTrainer(**trainer_params)
elif cfg.model_type == 'hnet':
    trainer = HNetTrainer(**trainer_params)
elif cfg.model_type == 'segformer':
    trainer = SegFormerTrainer(**trainer_params)
else:
    raise ValueError(f"Unknown model type: {cfg.model_type}")

trainer.train()