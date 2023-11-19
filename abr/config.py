
class Config():
    #######################
    # Training Parameters #
    #######################
    
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32
    EPOCHS = 10

    #################
    # Data Creation #
    #################

    # Train / Val / Test split ratios
    DATA_SPLIT = [0.8, 0.1, 0.1]

    # Forces a data preparation
    PREPARE_DATA = True
    
    # Forces a new train/val/test split randomization
    USE_NEW_TTV_SPLIT = False