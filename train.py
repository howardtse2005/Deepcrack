from data.augmentation import augCompose, RandomBlur, RandomColorJitter
from data.dataset import readIndex, dataReadPip, loadedDataset
from tqdm import tqdm
from model.deepcrack import DeepCrack
from model.hnet import HNet
from model.unet import UNet
from model.attention_unet import AttentionUNet
from model.segformer import SegFormer
from trainer import DeepCrackTrainer, UNetTrainer, SegFormerTrainer
from config import Config as cfg
import numpy as np
import torch
import os
import cv2
import sys
import datetime
import time

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id


def save_config_to_file(model_path=None):
    """
    Save all configuration parameters to a text file
    Args:
        model_path: Path to the saved model, if available
    """
    # Create directory if it doesn't exist
    config_dir = 'deepcrack_results/config'
    os.makedirs(config_dir, exist_ok=True)

    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a descriptive filename
    if model_path:
        model_name = os.path.basename(model_path)
        filename = f"{timestamp}_{model_name}_config.txt"
    else:
        filename = f"{timestamp}_config.txt"

    filepath = os.path.join(config_dir, filename)

    # Write configuration to file
    with open(filepath, 'w') as f:
        # Write header
        f.write("=" * 50 + "\n")
        f.write(f"DeepCrack Training Configuration - {timestamp}\n")
        f.write("=" * 50 + "\n\n")

        # Write model configuration
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"model_type: {cfg.model_type}\n")
        f.write("\n")

        # Write image and crop configuration
        f.write("IMAGE AND CROP CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"target_width: {cfg.target_width}\n")
        f.write(f"target_height: {cfg.target_height}\n")
        f.write(f"target_size: {cfg.target_size}\n")
        f.write(f"kernel_size: {cfg.kernel_size}\n")
        f.write(f"min_size: {cfg.min_size}\n")
        f.write(f"num_crops: {cfg.num_crops}\n")
        f.write(f"num_crops_with_cracks: {cfg.num_crops_with_cracks}\n")
        f.write("\n")

        # Write loss configuration with emphasis
        f.write("LOSS CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"use_focal_loss: {cfg.use_focal_loss}\n")
        if cfg.use_focal_loss:
            f.write(f"focal_alpha: {cfg.focal_alpha} (Class balance weight for cracks)\n")
            f.write(f"focal_gamma: {cfg.focal_gamma} (Focus on hard examples)\n")
            f.write(f"focal_beta: {cfg.focal_beta} (Global scaling factor)\n")
        else:
            f.write(f"pos_pixel_weight: {cfg.pos_pixel_weight}\n")
        f.write("\n")

        # Write training configuration
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"learning_rate: {cfg.lr}\n")
        f.write(f"optimizer: {'Adam' if cfg.use_adam else 'SGD'}\n")
        f.write(f"momentum: {cfg.momentum}\n")
        f.write(f"weight_decay: {cfg.weight_decay}\n")
        f.write(f"lr_decay: {cfg.lr_decay}\n")
        f.write(f"batch_size: {cfg.train_batch_size}\n")
        f.write(f"epochs: {cfg.epoch}\n")
        if cfg.pretrained_model:
            f.write(f"pretrained_model: {cfg.pretrained_model}\n")
        f.write("\n")

        # Write dataset configuration
        f.write("DATASET CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"train_data_path: {cfg.train_data_path}\n")
        f.write(f"val_data_path: {cfg.val_data_path}\n")
        f.write(f"test_data_path: {cfg.test_data_path}\n")
        f.write("\n")

        # Write model saving information
        f.write("MODEL SAVING CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"checkpoint_path: {cfg.checkpoint_path}\n")
        f.write(f"saver_path: {cfg.saver_path}\n")
        f.write(f"max_save: {cfg.max_save}\n")
        if model_path:
            f.write(f"saved_model: {model_path}\n")
        f.write("\n")

        # Write other configuration
        f.write("OTHER CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"name: {cfg.name}\n")
        f.write(f"gpu_id: {cfg.gpu_id}\n")
        f.write(f"acc_sigmoid_th: {cfg.acc_sigmoid_th}\n")

    print(f"Configuration saved to {filepath}")
    return filepath


def main():
    # ----------------------- dataset ----------------------- #

    data_augment_op = augCompose(transforms=[[RandomColorJitter, 0.5], [RandomBlur, 0.2]])

    ## SELF-ADDED (Specify target size for resizing)
    train_pipline = dataReadPip(transforms=data_augment_op, crop=True)
    test_pipline = dataReadPip(transforms=None, crop=True)

    train_dataset = loadedDataset(readIndex(cfg.train_data_path, shuffle=True), preprocess=train_pipline)

    test_dataset = loadedDataset(readIndex(cfg.val_data_path), preprocess=test_pipline)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size,
                                               shuffle=True, num_workers=4, drop_last=False)

    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.val_batch_size,
                                             shuffle=False, num_workers=4, drop_last=False)

    # Debug
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of batches in train_loader: {len(train_loader)}")
    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()    # Select model based on config
    if cfg.model_type == 'hnet':
        model = HNet()
        print("Using HNet architecture")
    elif cfg.model_type == 'unet':
        model = UNet()        
        print("Using UNet architecture")
    elif cfg.model_type == 'attention_unet':
        model = AttentionUNet()
        print("Using Attention UNet architecture")
    elif cfg.model_type == 'segformer': 
        model = SegFormer(num_classes=1, phi=cfg.segformer_variant, pretrained=cfg.segformer_pretrained)
        print(f"Using SegFormer {cfg.segformer_variant} architecture")
    else:
        model = DeepCrack()
        print("Using DeepCrack architecture")

    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    # Select trainer based on model type
    if cfg.model_type in ['hnet', 'unet', 'attention_unet']:
        trainer = UNetTrainer(model).to(device)
        print("Using UNetTrainer (single output)")
    elif cfg.model_type == 'segformer':
        trainer = SegFormerTrainer(model).to(device)
        print("Using SegFormerTrainer (single output)")
    else:
        trainer = DeepCrackTrainer(model).to(device)
        print("Using DeepCrackTrainer (multi-output)")

    if cfg.pretrained_model:
        pretrained_dict = trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        trainer.vis.log('load checkpoint: %s' % cfg.pretrained_model, 'train info')

    try:

        for epoch in range(1, cfg.epoch):
            trainer.vis.log(f'Start Epoch {epoch} ...', 'train info')
            trainer.vis.set_epoch(epoch)  # Update epoch for TensorBoard
            model.train()

            # Track total loss for this epoch
            epoch_total_loss = 0.0
            epoch_samples = 0            # ---------------------  training ------------------- #
            bar = tqdm(enumerate(train_loader), total=len(train_loader))
            bar.set_description('Epoch %d --- Training --- :' % epoch)
            
            for idx, (img, lab) in bar:
                data, target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
                pred = trainer.train_op(data, target)
                
                # Accumulate total loss for this epoch
                epoch_total_loss += trainer.log_loss['total_loss']
                epoch_samples += 1
                
                if idx % cfg.vis_train_loss_every == 0:
                    trainer.vis.log_dict(trainer.log_loss, 'train_loss')
                    if cfg.model_type in ['hnet', 'unet', 'attention_unet', 'segformer']:
                        # Single output model
                        trainer.vis.plot_many({
                            'train_total_loss': trainer.log_loss['total_loss'],
                            'train_output_loss': trainer.log_loss['output_loss'],
                        })
                    else:
                        # Multi-output model (DeepCrack)
                        trainer.vis.plot_many({
                            'train_total_loss': trainer.log_loss['total_loss'],
                            'train_output_loss': trainer.log_loss['output_loss'],
                            'train_fuse5_loss': trainer.log_loss['fuse5_loss'],
                            'train_fuse4_loss': trainer.log_loss['fuse4_loss'],
                            'train_fuse3_loss': trainer.log_loss['fuse3_loss'],
                            'train_fuse2_loss': trainer.log_loss['fuse2_loss'],
                            'train_fuse1_loss': trainer.log_loss['fuse1_loss'],
                        })

                if idx % cfg.vis_train_acc_every == 0:
                    if cfg.model_type in ['hnet', 'unet', 'attention_unet', 'segformer']:
                        trainer.acc_op(pred, target)
                    else:
                        trainer.acc_op(pred[0], target)
                    trainer.vis.log_dict(trainer.log_acc, 'train_acc')
                    trainer.vis.plot_many({
                        'train_mask_acc': trainer.log_acc['mask_acc'],
                        'train_mask_pos_acc': trainer.log_acc['mask_pos_acc'],
                        'train_mask_neg_acc': trainer.log_acc['mask_neg_acc'],
                    })
                    
                if idx % cfg.vis_train_img_every == 0:
                    if cfg.model_type in ['hnet', 'unet', 'attention_unet', 'segformer']:
                        # Single output model
                        trainer.vis.img_many({
                            'train_img': data.cpu(),
                            'train_output': torch.sigmoid(pred.contiguous().cpu()),
                            'train_lab': target.unsqueeze(1).cpu(),
                        })
                    else:
                        # Multi-output model (DeepCrack)
                        trainer.vis.img_many({
                            'train_img': data.cpu(),
                            'train_output': torch.sigmoid(pred[0].contiguous().cpu()),
                            'train_lab': target.unsqueeze(1).cpu(),
                            'train_fuse5': torch.sigmoid(pred[1].contiguous().cpu()),
                            'train_fuse4': torch.sigmoid(pred[2].contiguous().cpu()),
                            'train_fuse3': torch.sigmoid(pred[3].contiguous().cpu()),
                            'train_fuse2': torch.sigmoid(pred[4].contiguous().cpu()),
                            'train_fuse1': torch.sigmoid(pred[5].contiguous().cpu()),
                        })

                if idx % cfg.val_every == 0:
                    # -------------------- val ------------------- #
                    trainer.vis.log('Start Val %d ....' % idx, 'train info') 
                    model.eval()
                    
                    if cfg.model_type in ['hnet', 'unet', 'attention_unet', 'segformer']:
                        # Single output model
                        val_loss = {
                            'eval_total_loss': 0,
                            'eval_output_loss': 0,
                        }
                    else:
                        # Multi-output model
                        val_loss = {
                            'eval_total_loss': 0,
                            'eval_output_loss': 0,
                            'eval_fuse5_loss': 0,
                            'eval_fuse4_loss': 0,
                            'eval_fuse3_loss': 0,
                            'eval_fuse2_loss': 0,
                            'eval_fuse1_loss': 0,
                        }
                        
                    val_acc = {
                        'mask_acc': 0,
                        'mask_pos_acc': 0,
                        'mask_neg_acc': 0,
                    }

                    bar.set_description('Epoch %d --- Evaluation --- :' % epoch)

                    with torch.no_grad():
                        for idx, (img, lab) in enumerate(val_loader, start=1):
                            val_data, val_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(
                                torch.cuda.FloatTensor).to(device)
                            val_pred = trainer.val_op(val_data, val_target)
                            
                            if cfg.model_type in ['hnet', 'unet', 'attention_unet', 'segformer']:
                                trainer.acc_op(val_pred, val_target)
                                val_loss['eval_total_loss'] += trainer.log_loss['total_loss']
                                val_loss['eval_output_loss'] += trainer.log_loss['output_loss']
                            else:
                                trainer.acc_op(val_pred[0], val_target)
                                val_loss['eval_total_loss'] += trainer.log_loss['total_loss']
                                val_loss['eval_output_loss'] += trainer.log_loss['output_loss']
                                val_loss['eval_fuse5_loss'] += trainer.log_loss['fuse5_loss']
                                val_loss['eval_fuse4_loss'] += trainer.log_loss['fuse4_loss']
                                val_loss['eval_fuse3_loss'] += trainer.log_loss['fuse3_loss']
                                val_loss['eval_fuse2_loss'] += trainer.log_loss['fuse2_loss']
                                val_loss['eval_fuse1_loss'] += trainer.log_loss['fuse1_loss']
                                
                            val_acc['mask_acc'] += trainer.log_acc['mask_acc']
                            val_acc['mask_pos_acc'] += trainer.log_acc['mask_pos_acc']
                            val_acc['mask_neg_acc'] += trainer.log_acc['mask_neg_acc']
                        else:
                            if cfg.model_type in ['hnet', 'unet', 'attention_unet', 'segformer']:
                                # Single output model
                                trainer.vis.img_many({
                                    'eval_img': val_data.cpu(),
                                    'eval_output': torch.sigmoid(val_pred.contiguous().cpu()),
                                    'eval_lab': val_target.unsqueeze(1).cpu(),
                                })
                                trainer.vis.plot_many({
                                    'eval_total_loss': val_loss['eval_total_loss'] / idx,
                                    'eval_output_loss': val_loss['eval_output_loss'] / idx,
                                })
                            else:
                                # Multi-output model
                                trainer.vis.img_many({
                                    'eval_img': val_data.cpu(),
                                    'eval_output': torch.sigmoid(val_pred[0].contiguous().cpu()),
                                    'eval_lab': val_target.unsqueeze(1).cpu(),
                                    'eval_fuse5': torch.sigmoid(val_pred[1].contiguous().cpu()),
                                    'eval_fuse4': torch.sigmoid(val_pred[2].contiguous().cpu()),
                                    'eval_fuse3': torch.sigmoid(val_pred[3].contiguous().cpu()),
                                    'eval_fuse2': torch.sigmoid(val_pred[4].contiguous().cpu()),
                                    'eval_fuse1': torch.sigmoid(val_pred[5].contiguous().cpu()),
                                })
                                trainer.vis.plot_many({
                                    'eval_total_loss': val_loss['eval_total_loss'] / idx,
                                    'eval_output_loss': val_loss['eval_output_loss'] / idx,
                                    'eval_fuse5_loss': val_loss['eval_fuse5_loss'] / idx,
                                    'eval_fuse4_loss': val_loss['eval_fuse4_loss'] / idx,
                                    'eval_fuse3_loss': val_loss['eval_fuse3_loss'] / idx,
                                    'eval_fuse2_loss': val_loss['eval_fuse2_loss'] / idx,
                                    'eval_fuse1_loss': val_loss['eval_fuse1_loss'] / idx,
                                })
                            
                            trainer.vis.plot_many({
                                'eval_mask_acc': val_acc['mask_acc'] / idx,
                                'eval_mask_neg_acc': val_acc['mask_neg_acc'] / idx,
                                'eval_mask_pos_acc': val_acc['mask_pos_acc'] / idx,
                            })
                            
                            # ----------------- save model ---------------- #
                            if cfg.save_pos_acc < (val_acc['mask_pos_acc'] / idx) and cfg.save_acc < (
                                    val_acc['mask_acc'] / idx):
                                cfg.save_pos_acc = (val_acc['mask_pos_acc'] / idx)
                                cfg.save_acc = (val_acc['mask_acc'] / idx)
                                trainer.saver.save(model, tag='%s_epoch(%d)_acc(%0.5f/%0.5f)' % (
                                    cfg.name, epoch, cfg.save_pos_acc, cfg.save_acc))
                                trainer.vis.log('Save Model %s_epoch(%d)_acc(%0.5f/%0.5f)' % (
                                    cfg.name, epoch, cfg.save_pos_acc, cfg.save_acc), 'train info')

                    bar.set_description('Epoch %d --- Training --- :' % epoch)
                    model.train()

            # Calculate and print average loss for this epoch
            epoch_avg_loss = epoch_total_loss / max(epoch_samples, 1)
            print(f"\nEpoch {epoch} completed - Average Loss: {epoch_avg_loss:.6f}")

            if epoch != 0:
                trainer.saver.save(model, tag='%s_epoch(%d)' % (
                    cfg.name, epoch))
                trainer.vis.log('Save Model -%s_epoch(%d)' % (
                    cfg.name, epoch), 'train info')

        # Save configuration at the end of training
        save_config_to_file(model_path=trainer.saver.show_save_pth_name)
        
        # Export a single loss curve only at the end of training
        loss_curve_path = trainer.export_loss_curves("training_complete")
        trainer.vis.log(f'Loss curves saved to {loss_curve_path}', 'train info')

    except KeyboardInterrupt:
        # Save model on interruption
        trainer.saver.save(model, tag='Auto_Save_Model')
        print('\n Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name)
        trainer.vis.log('Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name,
                      'train info')

        # Export loss curves only when interrupted
        loss_curve_path = trainer.export_loss_curves("training_interrupted") 
        print(f'Loss curves saved to {loss_curve_path}')
        
        # Also save configuration on interruption
        config_path = save_config_to_file(model_path=trainer.saver.show_save_pth_name)
        print(f'Configuration saved to {config_path}')

        trainer.vis.log('Training End!!')
        # Close TensorBoard writer
        trainer.vis.close()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    main()