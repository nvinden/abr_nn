
class Config():
    #######################
    # Training Parameters #
    #######################
    
    LEARNING_RATE = 0.000006
    BATCH_SIZE = 32
    EPOCHS = 50
    DROPOUT_RATE = 0.2
    
    # Model output sizes
    TEMP_OUTPUT_LENGTH = 30
    SPAT_OUTPUT_LENGTH = 30
    
    # LSTM
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 6
    LSTM_MAX_TEMPORAL_DATA = 30 # Total values to add to the temporal LSTM
    
    # Turning on and off temporal and spatial models
    USE_TEMPORAL_MODEL = False
    USE_SPATIAL_MODEL = False

    #################
    # Data Creation #
    #################

    DATA_DIR = "data/full_data_NY"
    DRUG_IDS_FILE = "data/full_data_NY/drug_key.csv"
    ORG_IDS_FILE = "data/full_data_NY/org_standard_key.csv"
    SOURCE_IDS_FILE = "data/full_data_NY/source_key.csv"
    ASSAY_IDS_FILE = "data/full_data_NY/assay_key.csv"

    # Train / Val / Test split ratios
    DATA_SPLIT = [0.8, 0.1, 0.1]

    # Forces a data preparation
    PREPARE_DATA = False
    
    # Forces a new train/val/test split randomization
    USE_NEW_TTV_SPLIT = False
    
    @staticmethod
    def to_dict():
        return {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and not callable(v)}