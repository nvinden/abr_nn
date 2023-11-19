from pytorch_lightning import Trainer
from abr.data import ABRDataModule
from abr.model import ABRModel
from abr.config import Config

def main():
    config = Config()
    # Instantiate the data module and model
    data_module = ABRDataModule(config = config)
    data_module.setup()
    train_dataset = data_module.custom_train
    train_dataset[0]
    
    model = ABRModel()

    # Initialize a trainer
    trainer = Trainer(max_epochs=10)

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Optionally, test the model
    trainer.test(datamodule=data_module)

if __name__ == '__main__':
    main()
