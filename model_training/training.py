import numpy as np
import torch

class ModelTraining:

    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.losses = []
        self.val_losses = []
        self.total_epoches = 0
        self.train_test_fn = self._make_train_step_fn()
        self.val_test_fn = self._make_valid_step_fn()
        self.train_loader = None
        self.val_loader = None
    
    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def _make_train_step_fn(self):

        def perform_train_step(x, y):
            self.model.train()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return perform_train_step
    
    def _make_valid_step_fn(self):

        def perform_valid_step(x, y):
            self.model.eval()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)

            return loss.item
        
        return perform_valid_step
    
    def _mini_batch(self, validation=False):

        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn
        
        if data_loader is None:
            return None
        
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
        loss = np.mean(mini_batch_losses)

        return loss
    
    def set_seed(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def train(self, num_epoch, seed=42):
        self.set_seed(seed)

        for epoch in range(num_epoch):
            self.total_epoches += 1

            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.total_epoches,
            'model_state_dict': self.model.state_dict,
            'optimizer_state_dict': self.optimizer.state_dict,
            'losses': self.losses,
            'val_losses': self.val_losses
        }

        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimixer_state_dict'])
        self.total_epoches = checkpoint['epoch']
        self.losses = checkpoint['losses']
        self.val_losses = checkpoint['val_losses']

        self.model.train()
    
    def predict(self, x):
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        y_hat = self.model(x_tensor)
        self.model.train()

        return y_hat.numpy()
    
    def count_parameters(self):
        return sum(p.numel() for p in self.modle.parametrs() if p.requires_grad)
