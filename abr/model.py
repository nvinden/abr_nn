from collections import defaultdict

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

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

        self.final_linear = nn.Linear(512, num_drugs)

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
    MAIN_FEATURE_LENGTH = 923
    SPAT_FEATURE_LENGTH = 58
    TEMP_FEATURE_LENGTH = 981
    TEMP_SEQLEN = 100
    
    def __init__(self, config : Config):
        super().__init__()
        self.config = config
        
        self.main_net, self.temp_net, self.spat_net = self._build_networks()
        
        # Loss logging for end of epoch
        self.all_gt_class_val = list()
        self.all_pred_class_val = list()
        
        self.all_gt_class_test = list()
        self.all_pred_class_test = list()

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
            nn.Linear(self.SPAT_FEATURE_LENGTH * (len(COUNTY2ID)), 128),
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
        x = torch.sigmoid(x)
        
        return x

    def training_step(self, batch, batch_idx):
        # Training step logic
        gt = batch['gt']
        
        main_x = batch['main']
        temp_x = batch['temp']
        spat_x = batch['spat']
        
        pred = self(main_x, temp_x, spat_x)
        
        abr_loss = self.__get_loss(pred, gt)
        
        self.log("train_loss", abr_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return abr_loss

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
        
        main_x = batch['main']
        temp_x = batch['temp']
        spat_x = batch['spat']
        
        pred = self(main_x, temp_x, spat_x)
        
        abr_loss = self.__get_loss(pred, gt)
        
        self.log("val_loss", abr_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Loggin Val Accuracy
        (accuracy, precision, recall, f1), (gt_class, pred_class) = self.__calculate_accuracy_metrics(pred = pred, gt = gt)
        
        # Log the metrics
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('naive_model_accuracy', 0.333333333333333, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # For each row add the predicted drug values, and the gt drug values, along with each drugs tp, fp, fn, tn
        self.all_gt_class_val.extend(gt_class.tolist())
        self.all_pred_class_val.extend(pred_class.tolist())
            
    def on_validation_epoch_end(self):
        all_gt = self.all_gt_class_val
        all_preds = self.all_pred_class_val
        
        self.all_gt_class_val = list()
        self.all_pred_class_val = list()
        
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
    
    def __get_loss(self, pred, gt):
        # Create a mask for values where gt is not equal to -1.
        # drugs with -1 values were not tested in reality and should be ignored
        mask = gt != -1

        # Apply the mask to filter out ignored gt and corresponding predictions
        gt_filtered = torch.masked_select(gt, mask)
        pred_filtered = torch.masked_select(pred, mask)

        # Logging losses
        abr_loss = nn.functional.mse_loss(pred_filtered, gt_filtered)
        
        return abr_loss
    
    # Returns metrics:
    # - Accuracy
    # - Precision
    # - Recall
    # - F1
    # GT, and pred can fit into 3 buckets, S = 0.0, I = 0.5, R = 1.0
    def __calculate_accuracy_metrics(self, pred, gt):
        # Create a mask for values where gt is not equal to -1.
        # drugs with -1 values were not tested in reality and should be ignored
        mask = gt != -1

        # Apply the mask to filter out ignored gt and corresponding predictions
        gt_filtered = torch.masked_select(gt, mask)
        pred_filtered = torch.masked_select(pred, mask)
        
        # Turn the matricies into their corresponding categories: S, I, R
        convert_func = np.vectorize(lambda x: 'S' if x < 0.3333 else ('I' if x < 0.66666 else 'R'))
        gt_cat = convert_func(gt_filtered)
        pred_cat = convert_func(pred_filtered)
        
        # 1) Accuracy, Precision, Recall, f1
        # TODO: change macro to use global data
        accuracy = accuracy_score(gt_cat, pred_cat)
        precision = precision_score(gt_cat, pred_cat, average='macro')
        recall = recall_score(gt_cat, pred_cat, average='macro')
        f1 = f1_score(gt_cat, pred_cat, average='macro')
        
        # 2) AUC scores with the probabilities:
        #pred_np = np.array(pred_filtered)
        #auc = roc_auc_score(gt_cat, pred_np, average='macro', multi_class='ovr')
        
        return (accuracy, precision, recall, f1), (gt_cat, pred_cat)
    
    # Returns a list of dicts. One list for each drug.
    # [
    #    "druga_pred": S, "druga_gt": S, "druga_cat": TP, "drugb_pred:"....  
    # ]
    # If a drug is not tested, pred, gt and cat are all set as "N/A"
    # TODO: implement this
    def __get_result_dicts(self, pred: torch.Tensor, gt: torch.Tensor):
        out_list = []
        
        for i in range(pred.shape[1]):
            curr_pred = pred[i]
            curr_gt = gt[i]
            
            curr_dict = []
            
            # For each drug
            for j in range(len(curr_pred)):
                current_drug = COLUMN_DRUGS[j]
                
                # If result is a null
                if curr_gt[j] < 0.0:
                    pass
        
    def test_step(self, batch, batch_idx):
        # test step logic
        gt = batch['gt']
        
        main_x = batch['main']
        temp_x = batch['temp']
        spat_x = batch['spat']
        
        pred = self(main_x, temp_x, spat_x)
        
        abr_loss = self.__get_loss(pred, gt)
        
        self.log("test_loss", abr_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Loggin Val Accuracy
        (accuracy, precision, recall, f1), (gt_class, pred_class) = self.__calculate_accuracy_metrics(pred = pred, gt = gt)
        
        # Log the metrics
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # For each row add the predicted drug values, and the gt drug values, along with each drugs tp, fp, fn, tn
        self.all_gt_class_test.extend(gt_class.tolist())
        self.all_pred_class_test.extend(pred_class.tolist())
        
    def on_test_epoch_end(self):
        all_gt = self.all_gt_class_test
        all_preds = self.all_pred_class_val
        
        self.all_gt_class_test = list()
        self.all_pred_class_test = list()
        
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

        self.epoch_number += 1

    def configure_optimizers(self):
        # Configure optimizers and LR schedulers
        optimizer = Adam(self.parameters(), lr=self.config.LEARNING_RATE)
        return optimizer
