import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam

class ABRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.main_net, self.temp_net, self.spat_net = self._build_networks()

    # Network building helper function
    # Returns the main, temporal, and spatial networks
    def _build_networks(self):
        return None, None, None # Temp

    def forward(self, x):
        # Forward pass through the network
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        # Training step logic
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step logic
        pass

    def test_step(self, batch, batch_idx):
        # Test step logic
        pass

    def configure_optimizers(self):
        # Configure optimizers and LR schedulers
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
