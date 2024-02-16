from collections import defaultdict

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import wandb

import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from .config import Config
from .ids import COUNTY2ID, COLUMN_DRUGS

import torch
import torch.nn as nn
import torch.nn.functional as F

class MainModel(nn.Module):
    def __init__(self, config : Config, main_feature_length, num_drugs, dropout_rate):
        super(MainModel, self).__init__()
        
        self.config = config

        self.dropout_rate = dropout_rate
        self.init_linear = nn.Linear(main_feature_length + self.config.SPAT_OUTPUT_LENGTH + self.config.TEMP_OUTPUT_LENGTH, 512)

        # Increasing the depth
        self.layer1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate)
        )
        # Skip connection 1
        self.skip1 = nn.Linear(512, 1024)

        self.layer4 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate)
        )
        # Skip connection 2
        self.skip2 = nn.Linear(1024, 1024)

        self.final_linear = nn.Linear(512, num_drugs + 1)

    def forward(self, x):
        x = F.relu(self.init_linear(x))

        # Applying layers with skip connections
        identity = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x + self.skip1(identity))  # Skip connection

        identity = x
        x = self.layer4(x)
        x = self.layer5(x + self.skip2(identity))  # Skip connection

        x = self.final_linear(x)
        return x


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
        
        self.all_gt_classifications_test = defaultdict(list)
        self.all_pred_classifications_test = defaultdict(list)

        self.epoch_number = 0

    # Network building helper function
    # Returns the main, temporal, and spatial networks
    def _build_networks(self):
        main_net = self._build_main_net()
        temp_net = self._build_temp_net()
        spat_net = self._build_spat_net()
        
        return main_net, temp_net, spat_net # Temp
    
    def _build_main_net(self):
        # Main network
        out_net = MainModel(
            config = self.config,
            main_feature_length = self.MAIN_FEATURE_LENGTH,
            num_drugs = len(COLUMN_DRUGS),
            dropout_rate = self.config.DROPOUT_RATE
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
        #assert int(main_x.shape[1]) == self.MAIN_FEATURE_LENGTH
        #assert int(temp_x.shape[1]) == self.TEMP_SEQLEN and int(temp_x.shape[2]) == self.TEMP_FEATURE_LENGTH
        #assert int(spat_x.shape[1]) == len(COUNTY2ID) + 1 and int(spat_x.shape[2]) == self.SPAT_FEATURE_LENGTH
        
        #assert len(main_x.shape) == 2
        #assert len(temp_x.shape) == 3
        #assert len(spat_x.shape) == 3
        
        # Flatten spatial data
        spat_x = spat_x.reshape(spat_x.shape[0], -1)
        
        # Temporal section
        if self.config.USE_TEMPORAL_MODEL:
            temp_feat = self.temp_net(temp_x)
        else:
            temp_feat = torch.zeros((temp_x.shape[0], self.config.TEMP_OUTPUT_LENGTH), device=self.device)
           
        # Spatial section 
        if self.config.USE_SPATIAL_MODEL:
            spat_feat = self.spat_net(spat_x)
        else:
            spat_feat = torch.zeros((spat_x.shape[0], self.config.TEMP_OUTPUT_LENGTH), device=self.device)
        
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
        
        # Ignore the tests that were never conducted
        mask = gt_per != -1
        filtered_pred_per = pred_per[mask]
        filtered_gt_per = gt_per[mask]
        if len(filtered_pred_per) == 0:
            per_loss = torch.tensor(0.0)
        else:
            per_loss = nn.functional.mse_loss(filtered_pred_per, filtered_gt_per)
        
        total_loss = abr_loss + per_loss
        
        self.log("Train Total Loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Train ABR Loss", abr_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Train Per Loss", per_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss

    def plot_confusion_matrix_as_image(self, cm, epoch, class_names):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title=f'Epoch {epoch}',
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        # Convert the plot to a PIL Image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)

        # Create wandb Image
        wandb_image = wandb.Image(image, caption=f"Confusion Matrix at Epoch {epoch}")

        buf.close()
        plt.close(fig)

        return wandb_image

    def validation_step(self, batch, batch_idx):
        # Validation step logic
        gt = batch['gt']
        
        gt_abr = gt[:, 0]
        gt_per = gt[:, 1:]
        
        main_x = batch['main']
        temp_x = batch['temp']
        spat_x = batch['spat']
        
        pred_abr, pred_per = self(main_x, temp_x, spat_x)

        # Loggin losses
        abr_loss = nn.functional.mse_loss(pred_abr, gt_abr)
        
        # Ignore the tests that were never conducted
        mask = gt_per != -1
        filtered_pred_per = pred_per[mask]
        filtered_gt_per = gt_per[mask]
        
        if len(filtered_pred_per) == 0:
            per_loss = torch.tensor(0.0)
        else:
            per_loss = nn.functional.mse_loss(filtered_pred_per, filtered_gt_per)
        
        total_loss = abr_loss + per_loss
        
        self.log("Val Total Loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Val ABR Loss", abr_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Val Per Loss", per_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Loggin Val Accuracy
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
        
        self.log('naive_model_accuracy', 0.333333333333333, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        mask = per_gt_buckets != -1
        filtered_pred_per_bucket = per_pred_buckets[mask]
        filtered_gt_per_bucket = per_gt_buckets[mask]

        self.all_gt_classifications["per"].append(filtered_gt_per_bucket)
        self.all_pred_classifications["per"].append(filtered_pred_per_bucket)
            
    def on_validation_epoch_end(self):
        all_gt = np.concatenate(self.all_gt_classifications["abr"]) - 1
        all_preds = np.concatenate(self.all_pred_classifications["abr"]) - 1
        
        # print all the unique values
        #print("UNIQUE VALUES FOR GT AND PRED")
        #print(np.unique(all_gt), all_gt.shape)
        #print(np.unique(all_preds), all_preds.shape)

        # Compute the confusion matrix
        cm = confusion_matrix(all_gt, all_preds)
        # Convert the confusion matrix to an image
        cm_image = self.plot_confusion_matrix_as_image(cm, self.epoch_number, [
                                                        "1) Susceptible",
                                                        "2) Intermediate",
                                                        "3) Resistant",
        ])
        
        wandb.log({"confusion_matrix_img": cm_image})

        per_gt = np.concatenate(self.all_gt_classifications["per"])
        per_preds = np.concatenate(self.all_pred_classifications["per"])
        
        # Compute the confusion matrix
        cm = confusion_matrix(per_gt, per_preds)
        
        # Convert the confusion matrix to an image
        cm_image = self.plot_confusion_matrix_as_image(cm, self.epoch_number, [
                                                        "1) Susceptible",
                                                        "2) Intermediate",
                                                        "3) Resistant",
        ])
        
        wandb.log({"per_confusion_matrix_img": cm_image})
        
        # Reset for the next epoch
        self.all_gt_classifications = defaultdict(list)
        self.all_pred_classifications = defaultdict(list)

        self.epoch_number += 1

    # Pred: (batch_size, 1) each_containing a float [-1, and 1]
    # GT: (batch_size, 1) each_containing a number normally distrobuted with mean 0 and std 1
    def _get_abr_accuracy(self, pred, gt):        
        # Define 5 buckets for the range [-1, 1]
        pred_buckets = np.linspace(-1.000001, 1.000001, 4)  # Creates 4 points, resulting in 3 intervals
        pred_bunches = np.digitize(pred, pred_buckets, right=False)
        
        gt_percentiles = norm.ppf([0.33, 0.66])
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
        
        mask = gt_bunches != -1
        filtered_pred_per_bucket = pred_bunches[mask]
        filtered_gt_per_bucket = gt_bunches[mask]
        
        per_accuracy = np.mean(filtered_pred_per_bucket == filtered_gt_per_bucket)
        
        return per_accuracy, (pred_bunches, gt_bunches)

    def test_step(self, batch, batch_idx):
        # Test step logic
        # Validation step logic
        gt = batch['gt']
        
        gt_abr = gt[:, 0]
        gt_per = gt[:, 1:]
        
        main_x = batch['main']
        temp_x = batch['temp']
        spat_x = batch['spat']
        
        pred_abr, pred_per = self(main_x, temp_x, spat_x)

        # Loggin losses
        abr_loss = nn.functional.mse_loss(pred_abr, gt_abr)
        per_loss = nn.functional.mse_loss(pred_per, gt_per)
        
        total_loss = abr_loss + per_loss
        
        self.log("Test Total Loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Test ABR Loss", abr_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Test Per Loss", per_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Loggin Val Accuracy
        pred_abr = pred_abr.detach().cpu().numpy()
        pred_per = pred_per.detach().cpu().numpy()
        gt_abr = gt_abr.detach().cpu().numpy()
        gt_per = gt_per.detach().cpu().numpy()
        
        abr_accuracy, (abr_pred_buckets, abr_gt_buckets) = self._get_abr_accuracy(pred = pred_abr, gt = gt_abr)
        per_accuracy, (per_pred_buckets, per_gt_buckets) = self._get_per_accuracy(pred = pred_per, gt = gt_per)
        
        # Log the metrics
        self.log('test_abr_accuracy', abr_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_per_accuracy', per_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.all_gt_classifications_test['abr'].append(abr_gt_buckets.astype(int))
        self.all_pred_classifications_test['abr'].append(abr_pred_buckets.astype(int))
        
        self.log('naive_model_accuracy', 0.333333333333333, on_step=False, on_epoch=True, prog_bar=True, logger=True)
         
        mask = per_gt_buckets != -1
        filtered_pred_per_bucket = per_pred_buckets[mask]
        filtered_gt_per_bucket = per_gt_buckets[mask]

        self.all_gt_classifications_test["per"].append(filtered_gt_per_bucket)
        self.all_pred_classifications_test["per"].append(filtered_pred_per_bucket)
        
    def on_test_epoch_end(self):
        all_gt = np.concatenate(self.all_gt_classifications_test["abr"]) - 1
        all_preds = np.concatenate(self.all_pred_classifications_test["abr"]) - 1

        # Compute the confusion matrix
        cm = confusion_matrix(all_gt, all_preds)
        # Convert the confusion matrix to an image
        cm_image = self.plot_confusion_matrix_as_image(cm, self.config.EPOCHS, [
                                                        "1) Susceptible",
                                                        "2) Intermediate",
                                                        "3) Resistant",
        ])
        
        wandb.log({"test_confusion_matrix_img": cm_image})
        
        per_gt = np.concatenate(self.all_gt_classifications_test["per"])
        per_preds = np.concatenate(self.all_pred_classifications_test["per"])
        
        # Compute the confusion matrix
        cm = confusion_matrix(per_gt, per_preds)
        
        # Convert the confusion matrix to an image
        cm_image = self.plot_confusion_matrix_as_image(cm, self.epoch_number, [
                                                        "1) Susceptible",
                                                        "2) Intermediate",
                                                        "3) Resistant",
        ])
        
        wandb.log({"test_per_confusion_matrix_img": cm_image})

        # Reset for the next epoch
        self.all_gt_classifications = defaultdict(list)
        self.all_pred_classifications = defaultdict(list)

    def configure_optimizers(self):
        # Configure optimizers and LR schedulers
        optimizer = Adam(self.parameters(), lr=self.config.LEARNING_RATE)
        return optimizer
