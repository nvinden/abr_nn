
class Config():
    #######################
    # Training Parameters #
    #######################
    
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32
    EPOCHS = 10
    DROPOUT_RATE = 0.2
    
    # Model output sizes
    TEMP_OUTPUT_LENGTH = 100
    SPAT_OUTPUT_LENGTH = 100
    
    # LSTM
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 3

    #################
    # Data Creation #
    #################

    # Train / Val / Test split ratios
    DATA_SPLIT = [0.8, 0.1, 0.1]

    # Forces a data preparation
    PREPARE_DATA = False
    
    # Forces a new train/val/test split randomization
    USE_NEW_TTV_SPLIT = False
    
    # Total values to add to the temporal LSTM
    MAX_TEMPORAL_DATA = 100
    
    @staticmethod
    def to_dict():
        return {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and not callable(v)}