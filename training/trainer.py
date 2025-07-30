import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import tqdm
from tools.tensorboard_logger import TensorBoardLogger
from tools.checkpointer import Checkpointer
from training.loss import Loss

class Trainer(nn.Module):
    """
    Trainer class for training models with various configurations.

    REMARK:
    optimizer steps every batch, scheduler steps every epoch.
    """

    def __init__(self, model, name:str, optimizer:Optimizer,  criterions:list[Loss],
                 train_loader:DataLoader, val_loader:DataLoader,  log_dir:str, chkp_dir:str,
                 epoch_goal=100,scheduler:LRScheduler=None, device='cpu',
                 save_chkp_every:int=1
                 ):
        super().__init__()
        self.model = model
        self.name = name
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterions = criterions
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch_goal = epoch_goal
        self.logger = TensorBoardLogger(log_dir=log_dir, exp_name=name)
        self.checkpointer = Checkpointer(name=name, directory=chkp_dir, verbose=True, timestamp=False)
        self.save_chkp_every = save_chkp_every
        
    def train(self):
        """
        Train the model using the provided data loaders.
        """
        try:
            for epoch in range(self.epoch_goal):
                self.logger.set_epoch(epoch+1)
                self.checkpointer.set_epoch(epoch+1)
                print(f"Epoch {epoch+1}/{self.epoch_goal}")
                
                # Training round
                with tqdm.tqdm(self.train_loader, desc='Training') as pbar:
                    epoch_loss_train = 0
                    log_epoch_loss_train = {}
                    for batch_idx, (data, target) in enumerate(self.train_loader):
                                            
                        # get the result from one training step
                        batch_loss, log_loss = self._train_batch(data, target)
                        log_epoch_loss_train = self._add_dict(log_epoch_loss_train, log_loss)
                        epoch_loss_train += batch_loss.item()
                        

                        # update progress bar
                        pbar.set_postfix({'loss (batch)': batch_loss.item()})
                        pbar.update(1)
                    pbar.set_postfix({'loss (epoch)': epoch_loss_train / len(self.train_loader)})
                    log_epoch_loss_train = self._avg_dict(log_epoch_loss_train, len(self.train_loader))
                    self.logger.log_dict(log_epoch_loss_train, 'train')
                    pbar.close()
                    
                # Validation round
                with tqdm.tqdm(self.val_loader, desc='Validation') as pbar:
                    epoch_loss_val = 0
                    log_epoch_loss_val = {}
                    for batch_idx, (data, target) in enumerate(self.val_loader):

                        batch_loss, log_loss = self._val_batch(data, target)
                        log_epoch_loss_val = self._add_dict(log_epoch_loss_val, log_loss)
                        epoch_loss_val += batch_loss.item()
                        
                        pbar.set_postfix({'loss (batch)': batch_loss.item()})
                        pbar.update(1)
                    self.logger.log_dict(log_epoch_loss_val, 'val')
                    if self.scheduler is not None:
                        self.scheduler.step(epoch_loss_val / len(self.val_loader))
                pbar.set_postfix({'loss (epoch)': epoch_loss_val / len(self.val_loader)})
                pbar.close()
                
                if epoch % self.save_chkp_every == 0:
                    self.checkpointer(self.model)
            print(f"Training complete.")
            
        except KeyboardInterrupt:
            print("Training stopped by user.")
            print("Saving model state...")
            self.checkpointer(self.model)
            
        finally:
            print(f"Training loss curves saved to: {self.logger.export_loss_curves()}")

    def _train_batch(self, data, target):
        self.model.train()
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        batch_loss, log_loss = self._calculate_loss(output, target)
        self.optimizer.zero_grad()
        batch_loss.backward()
        
         # Check if gradients are flowing
        total_grad_norm = 0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        if total_grad_norm < 1e-8:
            print(f"Warning: Very small gradients! Norm: {total_grad_norm}")
        
        self.optimizer.step()
        return batch_loss, log_loss

    def _val_batch(self, data, target):
        self.model.eval()
        with torch.no_grad():
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            batch_loss, log_loss = self._calculate_loss(output, target)
        return batch_loss, log_loss


    def _calculate_loss(self, output, target):
        '''
        Sample loss calculation function.
        To be implemented by trainer subclass.
        '''
        for criterion in self.criterions:
            if isinstance(criterion, Loss):
                loss = criterion(output, target)
            else:
                loss = criterion(output, target)
        log_loss = {
            'bce_loss': loss.item(),
            'loss_test1': 1,
            'loss_test2': 2
            }
        return loss, log_loss
        pass
    
    def _add_dict(self, dict1, dict2):
        return {k: dict1.get(k, 0) + dict2.get(k, 0) for k in set(dict1) | set(dict2)}

    def _avg_dict(self, dict1, n):
        return {k: v / n for k, v in dict1.items()}
if __name__ == "__main__":
    # Example usage
    from torch import nn, optim
    from torchvision import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Dummy data loaders
    train_loader = [(torch.randn(4, 3, 224, 224), torch.randint(0, 2, (4,))) for _ in range(100)]
    val_loader = [(torch.randn(4, 3, 224, 224), torch.randint(0, 2, (4,))) for _ in range(20)]

    trainer = Trainer(model, optimizer, scheduler, criterion, device, None, 
                      train_loader=train_loader, val_loader=val_loader, log_dir='test')
    
    trainer.train()
    print("Training complete.")