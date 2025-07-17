import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialModule(L.LightningModule):
    """Lightning module for sequential classification tasks like MNIST/CIFAR."""
    
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        
        # Training parameters
        self.lr_factor = args.optimizer_cfg.get('configs', {}).get('lr_factor', 4.5)
        self.ssm_lr_base = args.optimizer_cfg.get('configs', {}).get('ssm_lr_base', 0.001)
        self.weight_decay = args.optimizer_cfg.get('configs', {}).get('weight_decay', 0.0)
        self.epochs = args.trainer_cfg.get('max_epochs', 250)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, _ = self.model(inputs)

        if torch.isnan(outputs).any():
            raise ValueError("Outputs are nan")

        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        accuracy = predicted.eq(targets).float().mean()
        
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_accuracy', accuracy, prog_bar=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, _ = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        accuracy = predicted.eq(targets).float().mean()
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', accuracy, prog_bar=True, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, _ = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        accuracy = predicted.eq(targets).float().mean()
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_accuracy', accuracy, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        S5 requires a specific optimizer setup.
        The S5 layer (A, B, C, dt) parameters typically require a smaller learning rate
        with no weight decay. The rest of the model uses higher learning rate with weight decay.
        """
        # Parameter groups for different learning rates
        ssm_fn_list = ["Lambda_re", "Lambda_im", "log_step", "norm"]
        not_optim = []
        
        lr = self.lr_factor * self.ssm_lr_base
        ssm_lr = self.ssm_lr_base
        
        def ssm_fn(param):
            if any(keyword in param[0] for keyword in ssm_fn_list):
                return 'ssm'
            elif any(keyword in param[0] for keyword in not_optim):
                return 'none'
            else:
                return 'regular'
        
        # Separate parameter groups
        params = list(self.model.named_parameters())
        param_groups = {'none': [], 'ssm': [], 'regular': []}
        
        for param in params:
            group = ssm_fn(param)
            param_groups[group].append(param[1])
        
        # Create optimizer with different parameter groups
        optimizer = torch.optim.AdamW([
            {'params': param_groups['ssm'], 'lr': ssm_lr, 'weight_decay': 0.0},
            {'params': param_groups['regular'], 'lr': lr, 'weight_decay': self.weight_decay},
        ])
        
        # Get steps per epoch - use a simple approach for now
        steps_per_epoch = 100  # Reasonable default for MNIST
        
        # Simple learning rate scheduler for now
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }