import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import tqdm
from tools.tensorboard_logger import TensorBoardLogger
from tools.checkpointer import Checkpointer
from training.loss import Loss
import inspect

class Trainer(nn.Module):
    """
    Trainer class for training models with various configurations.

    REMARK:
    optimizer steps every batch, scheduler steps every epoch.
    """

    def __init__(self, model, name:str, optimizer:Optimizer,  criterions:list[Loss],
                 train_loader:DataLoader, val_loader:DataLoader,  log_dir:str=None, chkp_dir:str=None,
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
        if log_dir is not None:
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
                self.model.train()
                with tqdm.tqdm(self.train_loader, desc='Training') as pbar:
                    epoch_loss_train = 0
                    log_epoch_loss_train = {}
                    num_train_batches = 0
                    
                    for batch_idx, (data, target) in enumerate(self.train_loader):
                        data, target = data.to(self.device), target.to(self.device)                    
                        batch_loss, log_loss = self._train_batch(data, target)
                        
                        # Accumulate losses properly
                        log_epoch_loss_train = self._add_dict(log_epoch_loss_train, log_loss)
                        epoch_loss_train += batch_loss.item()
                        num_train_batches += 1

                        pbar.set_postfix({'loss (batch)': batch_loss.item()})
                        pbar.update(1)
                    
                    # Calculate average losses
                    avg_train_loss = epoch_loss_train / num_train_batches
                    log_epoch_loss_train = self._avg_dict(log_epoch_loss_train, num_train_batches)
                    
                    pbar.set_postfix({'loss (epoch)': avg_train_loss})
                    self.logger.log_dict(log_epoch_loss_train, 'train')
                    pbar.close()
                    
                # Validation round
                self.model.eval()
                with tqdm.tqdm(self.val_loader, desc='Validation') as pbar:
                    with torch.no_grad():
                        epoch_loss_val = 0
                        log_epoch_loss_val = {}
                        num_val_batches = 0
                        
                        for batch_idx, (data, target) in enumerate(self.val_loader):
                            data, target = data.to(self.device), target.to(self.device)
                            batch_loss, log_loss = self._val_batch(data, target)
                            
                            log_epoch_loss_val = self._add_dict(log_epoch_loss_val, log_loss)
                            epoch_loss_val += batch_loss.item()
                            if batch_loss.item() > 1:
                                print(f"Warning: High validation loss {batch_loss.item()} at batch {batch_idx+1}")
                            num_val_batches += 1
                            
                            pbar.set_postfix({'loss (batch)': batch_loss.item()})
                            pbar.update(1)
                        
                        # Calculate average losses
                        avg_val_loss = epoch_loss_val / num_val_batches
                        log_epoch_loss_val = self._avg_dict(log_epoch_loss_val, num_val_batches)
                        
                        pbar.set_postfix({'loss (epoch)': avg_val_loss})
                        self.logger.log_dict(log_epoch_loss_val, 'val')
                        
                        if self.scheduler is not None:
                            sig = inspect.signature(self.scheduler.step)
                            if 'metrics' in sig.parameters:
                                self.scheduler.step(avg_val_loss)
                            else:
                                self.scheduler.step()
                    pbar.close()
                
                if epoch % self.save_chkp_every == 0:
                    self.checkpointer(self.model)
                
                print(f"Training: avg_loss={avg_train_loss:.6f}, batches={num_train_batches}")
                print(f"Validation: avg_loss={avg_val_loss:.6f}, batches={num_val_batches}")
                print(f"Training dataset size: {len(self.train_loader.dataset)}")
                print(f"Validation dataset size: {len(self.val_loader.dataset)}")
                
            print(f"Training complete.")
            
        except KeyboardInterrupt:
            print("Training stopped by user.")
            print("Saving model state...")
            self.checkpointer(self.model)
            
        finally:
            print(f"Training loss curves saved to: {self.logger.export_loss_curves()}")

    def _train_batch(self, data, target):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(data)
        batch_loss, log_loss = self._calculate_loss(output, target, requires_grad=True)
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss, log_loss

    def _val_batch(self, data, target):
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
            batch_loss, log_loss = self._calculate_loss(output, target, requires_grad=False)
        return batch_loss, log_loss


    def _calculate_loss(self, output, target, requires_grad):
        '''
        Sample loss calculation function.
        To be implemented by trainer subclass.
        '''
        raise NotImplementedError("This method should be implemented in the subclass.")
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