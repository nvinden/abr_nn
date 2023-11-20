from collections import defaultdict

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import wandb

from .config import Config
from .ids import COUNTY2ID, COLUMN_DRUGS

class ABRModel(pl.LightningModule):
    # Sizes from the data creation
    MAIN_FEATURE_LENGTH = 186
    SPAT_FEATURE_LENGTH = 3
    TEMP_FEATURE_LENGTH = 188
    TEMP_SEQLEN = 100
    
    def __init__(self, config : Config):
        super().__init__()
        self.config = config
        
        self.main_net, self.temp_net, self.spat_net = self._build_networks()
        
        # Loss logging for end of epoch
        self.all_gt_classifications = defaultdict(list)
        self.all_pred_classifications = defaultdict(list)

    # Network building helper function
    # Returns the main, temporal, and spatial networks
    def _build_networks(self):
        main_net = self._build_main_net()
        temp_net = self._build_temp_net()
        spat_net = self._build_spat_net()
        
        return main_net, temp_net, spat_net # Temp
    
    def _build_main_net(self):
        # Main network
        out_net = nn.Sequential(
            nn.Linear(self.MAIN_FEATURE_LENGTH + self.config.SPAT_OUTPUT_LENGTH + self.config.TEMP_OUTPUT_LENGTH, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(self.config.DROPOUT_RATE),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(self.config.DROPOUT_RATE),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(self.config.DROPOUT_RATE),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(self.config.DROPOUT_RATE),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(self.config.DROPOUT_RATE),

            nn.Linear(64, len(COLUMN_DRUGS) + 1)
        )
        
        return out_net
    
    def _build_temp_net(self):
        lstm = nn.LSTM(input_size = self.TEMP_FEATURE_LENGTH, 
                       hidden_size=self.config.LSTM_HIDDEN_SIZE,
                       num_layers = self.config.LSTM_NUM_LAYERS, 
                       dropout=self.config.DROPOUT_RATE,
                       batch_first=True)
        
        
        out_net = nn.Linear(self.config.LSTM_HIDDEN_SIZE, self.config.TEMP_OUTPUT_LENGTH)
        
        # Define the temporal network as a custom Sequential module
        # Cannot use sequential because LSTM returns a tuple
        class TemporalNet(nn.Module):
            def __init__(self, lstm, linear):
                super().__init__()
                self.lstm = lstm
                self.linear = linear

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_out = lstm_out[:, -1, :]
                return self.linear(last_out)

        # Instantiate the custom network
        temp_net = TemporalNet(lstm, out_net)
        return temp_net
    
    def _build_spat_net(self):
        out_net = nn.Sequential(
            nn.Linear(self.SPAT_FEATURE_LENGTH * (len(COUNTY2ID) + 1), 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(self.config.DROPOUT_RATE),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(self.config.DROPOUT_RATE),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(self.config.DROPOUT_RATE),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(self.config.DROPOUT_RATE),

            nn.Linear(64, self.config.TEMP_OUTPUT_LENGTH)
        )
        
        return out_net

    def forward(self, main_x : torch.Tensor, temp_x : torch.Tensor, spat_x : torch.Tensor):
        assert int(main_x.shape[1]) == self.MAIN_FEATURE_LENGTH
        assert int(temp_x.shape[1]) == self.TEMP_SEQLEN and int(temp_x.shape[2]) == self.TEMP_FEATURE_LENGTH
        assert int(spat_x.shape[1]) == len(COUNTY2ID) + 1 and int(spat_x.shape[2]) == self.SPAT_FEATURE_LENGTH
        
        assert len(main_x.shape) == 2
        assert len(temp_x.shape) == 3
        assert len(spat_x.shape) == 3
        
        # Flatten spatial data
        spat_x = spat_x.reshape(spat_x.shape[0], -1)
        
        # Forward pass through the network
        temp_feat = self.temp_net(temp_x)
        spat_feat = self.spat_net(spat_x)
        
        x = torch.cat([main_x, temp_feat, spat_feat], dim=1)
        x = self.main_net(x)
        
        total_abr_number = torch.tanh(x[:, 0])
        per_ab_class = torch.sigmoid(x[:, 1:])
        
        return total_abr_number, per_ab_class

    def training_step(self, batch, batch_idx):
        # Training step logic
        gt = batch['gt']
        
        gt_abr = gt[:, 0]
        gt_per = gt[:, 1:]
        
        main_x = batch['main']
        temp_x = batch['temp']
        spat_x = batch['spat']
        
        pred_abr, pred_per = self(main_x, temp_x, spat_x)
        
        abr_loss = nn.functional.mse_loss(pred_abr, gt_abr)
        per_loss = nn.functional.mse_loss(pred_per, gt_per)
        
        total_loss = abr_loss + per_loss
        
        self.log("Train Loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ABR Loss", abr_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Per Loss", per_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss
    
    def on_validation_epoch_end(self):
        all_gt = np.concatenate(self.all_gt_classifications["abr"]) - 1
        all_preds = np.concatenate(self.all_pred_classifications["abr"]) - 1

        # Compute the confusion matrix
        cm = confusion_matrix(all_gt, all_preds)

        # Log the confusion matrix using wandb
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, 
                                                                y_true=all_gt, 
                                                                preds=all_preds, 
                                                                class_names=[
                                                                    "1) Very Susceptible",
                                                                    "2) Moderate Susceptible",
                                                                    "3) Intermediate",
                                                                    "4) Moderate Resistant",
                                                                    "5) Very Resistant"
                                                                ])})

        # Reset for the next epoch
        self.all_gt_classifications = defaultdict(list)
        self.all_pred_classifications = defaultdict(list)

    def validation_step(self, batch, batch_idx):
        # Validation step logic
        gt = batch['gt']
        
        gt_abr = gt[:, 0]
        gt_per = gt[:, 1:]
        
        main_x = batch['main']
        temp_x = batch['temp']
        spat_x = batch['spat']
        
        pred_abr, pred_per = self(main_x, temp_x, spat_x)
        
        pred_abr = pred_abr.detach().cpu().numpy()
        pred_per = pred_per.detach().cpu().numpy()
        gt_abr = gt_abr.detach().cpu().numpy()
        gt_per = gt_per.detach().cpu().numpy()
        
        abr_accuracy, (abr_pred_buckets, abr_gt_buckets) = self._get_abr_accuracy(pred = pred_abr, gt = gt_abr)
        per_accuracy, (per_pred_buckets, per_gt_buckets) = self._get_per_accuracy(pred = pred_per, gt = gt_per)
        
        # Log the metrics
        self.log('val_abr_accuracy', abr_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_per_accuracy', per_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.all_gt_classifications['abr'].append(abr_gt_buckets.astype(int))
        self.all_pred_classifications['abr'].append(abr_pred_buckets.astype(int))
        
        for i, drug in enumerate(COLUMN_DRUGS):
            self.all_gt_classifications[drug].append(per_gt_buckets[:, i].astype(int))
            self.all_pred_classifications[drug].append(per_pred_buckets[:, i].astype(int))

    # Pred: (batch_size, 1) each_containing a float [-1, and 1]
    # GT: (batch_size, 1) each_containing a number normally distrobuted with mean 0 and std 1
    def _get_abr_accuracy(self, pred, gt):        
        # Define 5 buckets for the range [-1, 1]
        pred_buckets = np.linspace(-1, 1, 6)  # Creates 6 points, resulting in 5 intervals
        pred_bunches = np.digitize(pred, pred_buckets, right=False)
        
        gt_percentiles = norm.ppf([0.2, 0.4, 0.6, 0.8])
        gt_bunches = np.digitize(gt, gt_percentiles, right=False) + 1
        
        abr_accuracy = np.mean(pred_bunches == gt_bunches)
        
        return abr_accuracy, (pred_bunches, gt_bunches)
    
    def _get_per_accuracy(self, pred, gt):
        pred_buckets = np.linspace(0, 1, 4)  # Creates 4 points, resulting in 3 intervals
        pred_bunches = np.digitize(pred, pred_buckets, right=False)
        
        gt_bunches = gt * 2 + 1
        
        # Remove all of the -1s
        num_removed = np.sum(gt_bunches < -0.5)
        pred_bunches[gt_bunches < -0.5] = -100
        
        if num_removed == len(gt_bunches):
            return 0.0, (pred_bunches, gt_bunches)
        
        per_accuracy = np.sum(pred_bunches == gt_bunches) / (len(gt_bunches) - num_removed)
        
        return per_accuracy, (pred_bunches, gt_bunches)

    def test_step(self, batch, batch_idx):
        # Test step logic
        pass

    def configure_optimizers(self):
        # Configure optimizers and LR schedulers
        optimizer = Adam(self.parameters(), lr=self.config.LEARNING_RATE)
        return optimizer
