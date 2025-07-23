from pprint import pprint
import os
import setproctitle

class Config:
    name = 'DeepCrack_CT260_FT1'

    gpu_id = '0,1,2,3'

    setproctitle.setproctitle("%s" % name)

    # path
    train_data_path = 'data/train_example.txt'
    val_data_path = 'data/val_example.txt'
    test_data_path = 'data/test_example.txt'
    checkpoint_path = 'checkpoints'
    log_path = 'log'
    saver_path = os.path.join(checkpoint_path, name)
    print(f"Checkpoint directory: {saver_path}") # Debug
    max_save = 20

    # visdom
    vis_env = 'DeepCrack'
    port = 8097
    vis_train_loss_every = 40
    vis_train_acc_every = 40
    vis_train_img_every = 120
    val_every = 200

    # Image and crop settings
    target_width = 448
    target_height = 448
    target_size = (target_width, target_height)
    kernel_size = 3  # Kernel size for convolutions
    min_size = 448   # Minimum size for images
    num_crops = 20   # Total number of crops per image
    num_crops_with_cracks = 10  # Number of crops that should contain cracks
    train_random_crop = False # Bool to state wheteher we want to perform random cropping for training or not
    val_random_crop = False # Bool to state wheteher we want to perform random cropping for validation or not

    # training
    epoch = 100
    pretrained_model = ''  # Path to the pretrained model
    weight_decay = 0.0000
    lr_decay = 0.1
    lr = 1e-3
    momentum = 0.9
    use_adam = True  # Use Adam optimizer
    train_batch_size = 4
    val_batch_size = 4
    test_batch_size = 4

    # Loss configuration
    use_focal_loss = True  # Whether to use Focal Loss
    focal_alpha = 0.75  # Weight for positive examples in Focal Loss
    focal_gamma = 3.0  # Focusing parameter (higher values focus more on hard examples)
    focal_beta = 1.0   # Global scaling factor for the focal term
    pos_pixel_weight = 1  # Legacy parameter, used when use_focal_loss = False

    acc_sigmoid_th = 0.5

    # checkpointer
    save_format = ''
    save_acc = -1
    save_pos_acc = -1    # Model configuration
    model_type = 'unet'  # Options: 'deepcrack', 'unet', 'attention_unet', 'segformer', 'hnet'
    
    # SegFormer-specific configuration
    segformer_variant = 'b5'  # Options: 'b0', 'b1', 'b2', 'b3', 'b4', 'b5'
    segformer_pretrained = False  # Whether to use ImageNet pretrained weights
    
    # TensorBoard settings
    tensorboard_dir = 'runs/deepcrack'
    export_loss_dir = 'deepcrack_results/loss'  # Directory to save loss curve JPGs
    
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