from pprint import pprint
import os
import setproctitle

class Config:
    name = 'DeepCrack_CT260_FT1'
    model_type = 'hnet'  # Options: 'deepcrack', 'unet', 'attention_unet', 'segformer', 'hnet'
    gpu_id = '0,1,2,3'

    setproctitle.setproctitle("%s" % name)

    #<---------- Paths and Directories ----------->#
    dir_img_tr = 'data/july2025_imgs/crop_img_tr'
    dir_mask_tr = 'data/july2025_imgs/crop_msk_tr'
    dir_img_val = 'data/july2025_imgs/crop_img_val'
    dir_mask_val = 'data/july2025_imgs/crop_msk_val'
    dir_img_test = 'data/july2025_imgs/img_raw_ts'
    dir_mask_test = 'data/july2025_imgs/masks_raw_ts'
    checkpoint_path = 'checkpoints/' # Checkpoint path for training
    log_path = 'log'
    pretrained_model = 'checkpoints/hnet3_july.pth'  # Checkpoint path for testing
    #<-------------------------------------------->#
    
    
    #<---------- Preprocessing Settings ----------->#
    # Image and crop settings
    target_width = 448
    target_height = 448
    target_size = (target_width, target_height)
    
    #random crop settings
    crop_range = (300, 1000)
    num_crops = 10   # Total number of crops per image
    p_hascrack = 1
    use_raw = True
    kernel_size = 3  # Kernel size for convolutions
    min_size = 448   # Minimum size for images
    
    num_crops_with_cracks = 10  # Number of crops that should contain cracks
    train_random_crop = False # Bool to state wheteher we want to perform random cropping for training or not
    val_random_crop = False # Bool to state wheteher we want to perform random cropping for validation or not
    #<-------------------------------------------->#
    
    
    #<---------- Training Settings ----------->#
    epoch = 100
    lr = 1e-4
    train_batch_size = 4
    val_batch_size = 4
    test_batch_size = 4
    
    val_every = 1 # legacy parameter for old train.py
    use_focal_loss = True  # legacy parameter for old train.py to use focal loss or not
    
    
    weight_decay = 0.0000
    lr_decay = 0.1
    momentum = 0.9
    
    # optimizer settings:
    optimizer = 'adam'  # Options: 'adam', 'sgd', 'rmsprop'
    adam_params = {
        'lr': lr,
        'betas': (0.9, 0.999),
        'eps': 1e-08,
        'weight_decay': weight_decay
    }
    sgd_params = {
        'lr': lr,
        'momentum': momentum,
        'weight_decay': weight_decay
    }
    rmsprop_params = {
        'lr': lr,
        'alpha': 0.99,
        'eps': 1e-08,
        'weight_decay': weight_decay
    }
    
    # scheduler settings:
    scheduler = 'plateau'  # Options: 'step', 'plateau', 'cosine'
    step_params = {
        'step_size': 10,
        'gamma': lr_decay
    }
    plateau_params = {
        'mode': 'max',
        'factor': lr_decay,
        'patience': 5,
    }
    cosine_params = {
        'T_max': epoch,
        'eta_min': 0.0
    }
    #<-------------------------------------------->#

    #<------------- Loss Settings ---------------->#   
    criterions = [
        'focal'
    ] # Options: 'focal', 'dice', 'bce'
    
    focal_params = {
        'alpha': 0.25,
        'gamma': 2.0,
        'reduction': 'mean'
    }
    dice_params = {
        'smooth': 1e-6,
        'reduction': 'mean'
    }
    bce_params = {
        'reduction': 'mean'
    }
    #<-------------------------------------------->#
    
    
    # checkpointer
    save_format = ''
    save_acc = -1
    save_pos_acc = -1    # Model configuration
    
    
    # SegFormer-specific configuration
    segformer_variant = 'b5'  # Options: 'b0', 'b1', 'b2', 'b3', 'b4', 'b5'
    segformer_pretrained = False  # Whether to use ImageNet pretrained weights
    
    # TensorBoard settings
    tensorboard_dir = 'runs/deepcrack'
    export_loss_dir = 'results/loss'  # Directory to save loss curve JPGs
    
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

    def show(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')