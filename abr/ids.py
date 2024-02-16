from abr.config import *
import pandas as pd

COUNTY2ID = {}
COLUMN_DRUGS = {}

SPECIES2ID = {
        'CANINE': 0,
        'FELINE': 1
}

################
# Data Loading #
################

SOURCE_IDS = None
DRUG_IDS = None
ORG_IDS = None

def get_source_value(val):
    global SOURCE_IDS

    # If source IDS has not been initialized, do so
    if SOURCE_IDS is None:
        df = pd.read_csv(Config.SOURCE_IDS_FILE)

        # Unique categories with id associated
        unique_source_categories = df['source_cat1'].unique()
        unique_source_categories.sort()
        source_cat_to_ids = {k: v for v, k in enumerate(unique_source_categories)}

        SOURCE_IDS = {}
        for i, row in df.iterrows():
            SOURCE_IDS[row['source']] = source_cat_to_ids[row['source_cat1']]

    if val not in SOURCE_IDS:
        id_val = SOURCE_IDS['UNKNOWN']
        print(f"Unknown source: {val}")
    else:
        id_val = SOURCE_IDS[val]

    return id_val

def get_drug_value(val):
    global DRUG_IDS

    # If drug IDS has not been initialized, do so
    if DRUG_IDS is None:
        df = pd.read_csv(Config.DRUG_IDS_FILE)

        # Unique categories with id associated
        unique_drug_categories = df['class'].unique()
        unique_drug_categories.sort()
        drug_cat_to_ids = {k: v for v, k in enumerate(unique_drug_categories)}

        DRUG_IDS = {}
        for i, row in df.iterrows():
            DRUG_IDS[row['drug']] = drug_cat_to_ids[row['class']]

    if val not in DRUG_IDS:
        id_val = DRUG_IDS['UNKNOWN']
        print(f"Unknown drug: {val}")
    else:
        id_val = DRUG_IDS[val]

    return id_val

        
def get_org_value(val):
    global ORG_IDS

    # If org IDS has not been initialized, do so
    if ORG_IDS is None:
        df = pd.read_csv(Config.ORG_IDS_FILE)
        df.fillna('UNKNOWN', inplace=True)

        # Unique categories with id associated
        unique_org_categories = df['org_order'].unique()
        unique_org_categories.sort()
        org_cat_to_ids = {k: v for v, k in enumerate(unique_org_categories)}

        ORG_IDS = {}
        for i, row in df.iterrows():
            ORG_IDS[row['org_standard']] = org_cat_to_ids[row['org_order']]

    if val not in ORG_IDS:
        id_val = ORG_IDS['UNKNOWN']
        print(f"Unknown org: {val}")
    else:
        id_val = ORG_IDS[val]

    return id_val