from pytorch_lightning import Trainer
from abr.data import ABRDataModule
from abr.model import ABRModel
from abr.config import Config

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import datetime as dt

RUN_NAME = "new_ny_data"

def main():
    run_name = RUN_NAME + dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    wandb_logger = WandbLogger(
        name = run_name,
        project = 'abr',
        config = Config.to_dict(),
    )
    
    # Set up the checkpoint callback to save your model every epoch
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"saves/{run_name}",  # Specify the directory to save checkpoints
        filename="{epoch}",  # Filename includes epoch and validation loss
        save_top_k=-1,  # Set to -1 to save all epochs
        every_n_epochs=1,  # Save every epoch
    )
    
    config = Config()
    
    # Instantiate the data module and model
    data_module = ABRDataModule(config = config, data_path = config.DATA_DIR)    
    model = ABRModel(config = config)

    # Initialize a trainer
    trainer = Trainer(
        max_epochs=config.EPOCHS, 
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        #limit_train_batches=0.05 # TEMP
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Optionally, test the model
    trainer.test(datamodule=data_module)

if __name__ == '__main__':
    main()
