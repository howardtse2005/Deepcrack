from config import Config as cfg
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#<---------Initialize model ------------>
from model.attention_unet import AttentionUNet
from model.segformer import SegFormer
from model.hnet import HNet
from model.deepcrack import DeepCrack
from model.unet import UNet
if cfg.model_type == 'deepcrack':
    model = DeepCrack()
elif cfg.model_type == 'unet':
    model = UNet()
elif cfg.model_type == 'attention_unet':
    model = AttentionUNet()
elif cfg.model_type == 'segformer':
    model = SegFormer(variant=cfg.segformer_variant) 
elif cfg.model_type == 'hnet':
    model = HNet()
else:
    raise ValueError(f"Unknown model type: {cfg.model_type}")
model.to(device)

#<---------Initialize optimizer ------------>
import torch.optim as optim
if cfg.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), **cfg.adam_params)
elif cfg.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), **cfg.sgd_params)
elif cfg.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), **cfg.rmsprop_params)
else:
    raise ValueError(f"Unknown optimizer type: {cfg.optimizer}")
    
#<---------Initialize scheduler ------------>
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = None
if cfg.scheduler == 'step':
    scheduler = StepLR(optimizer, **cfg.step_params)
elif cfg.scheduler == 'plateau':
    scheduler = ReduceLROnPlateau(optimizer,**cfg.plateau_params)
elif cfg.scheduler == 'cosine':
    scheduler = CosineAnnealingLR(optimizer,**cfg.cosine_params)
elif cfg.scheduler is None:
    pass
else:
    raise ValueError(f"Unknown scheduler type: {cfg.scheduler}")

#<---------Initialize loss functions ------------>
from training.loss import FocalWithLogitsLoss, DiceWithLogitsLoss, BCEWithLogitsLoss
criterions = []
if 'focal' in cfg.criterions:
    criterions.append(FocalWithLogitsLoss(**cfg.focal_params))
if 'dice' in cfg.criterions:
    criterions.append(DiceWithLogitsLoss(**cfg.dice_params))
if 'bce' in cfg.criterions:
    criterions.append(BCEWithLogitsLoss(**cfg.bce_params))

#<---------Initialize data loaders ------------>
from data.dataset import CrackDataset
import data.preprocess_pipeline as pp

augmentations_train = []
if cfg.train_random_crop:
    augmentations_train.append(pp.Crop(range_crop_len=cfg.crop_range,
                                 n_copy=cfg.num_crops,
                                 each_has_crack=cfg.p_hascrack,
                                 use_raw=cfg.use_raw
                                 ))
augmentations_train.append(pp.Resize(target_size=cfg.target_size))
train_dataset = CrackDataset(
    dataset_img_path=cfg.dir_img_tr,
    dataset_mask_path=cfg.dir_mask_tr,
    augmentations=augmentations_train
)

augmentations_val = []
if cfg.val_random_crop:
    augmentations_val.append(pp.Crop(range_crop_len=cfg.crop_range,
                                 n_copy=cfg.num_crops,
                                 each_has_crack=cfg.p_hascrack,
                                 use_raw=cfg.use_raw
                                 ))
augmentations_val.append(pp.Resize(target_size=cfg.target_size))
val_dataset = CrackDataset(
    dataset_img_path=cfg.dir_img_val,
    dataset_mask_path=cfg.dir_mask_val,
    augmentations=augmentations_val
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=1)

#<---------Initialize trainer ------------>
from training.trainer_deepcrack import DeepCrackTrainer
from training.trainer_unet import UNetTrainer
from training.trainer_hnet import HNetTrainer
from training.trainer_segformer import SegFormerTrainer
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