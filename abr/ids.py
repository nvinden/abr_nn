from abr.config import *
import pandas as pd

COUNTY2ID = {}
COLUMN_DRUGS = {}

SPECIES2ID = {
        'CANINE': 0,
        'FELINE': 1
}

COUNTY2ID = {
   'New York': 0,
   'Richmond': 1,
   'Bronx': 2,
   'Westchester': 3,
   'Putnam': 4,
   'Rockland': 5,
   'Orange': 6,
   'Nassau': 7,
   'Queens': 8,
   'Kings': 9,
   'Suffolk': 10,
   'Saratoga': 11,
   'Albany': 12,
   'Montgomery': 13,
   'Schoharie': 14,
   'Rensselaer': 15,
   'Columbia': 16,
   'Schenectady': 17,
   'Ulster': 18,
   'Greene': 19,
   'Dutchess': 20,
   'Sullivan': 21,
   'Washington': 22,
   'Warren': 23,
   'Clinton': 24,
   'Essex': 25,
   'Onondaga': 26,
   'Madison': 27,
   'Cayuga': 28,
   'Oswego': 29,
   'Tompkins': 30,
   'Seneca': 31,
   'Otsego': 32,
   'Herkimer': 33,
   'Oneida': 34,
   'Broome': 35,
   'Chenango': 36,
   'Erie': 37,
   'Niagara': 38,
   'Chautauqua': 39,
   'Genesee': 40,
   'Monroe': 41,
   'Orleans': 42,
   'Ontario': 43,
   'Livingston': 44,
   'Wayne': 45,
   'Allegany': 46,
   'Steuben': 47,
   'Chemung': 48,
   'Yates': 49,
   'Delaware': 50,
   'Franklin': 51,
   'Saint Lawrence': 52,
   'Jefferson': 53,
   'Tioga': 54,
   'Wyoming': 55,
   'Schuyler': 56,
   'Fulton': 57,
   'Lewis': 58,
   'Cattaraugus': 59,
}

COLUMN_DRUGS = (
    "AMOXICILLINCLAVULANIC ACID",
    "AMOXICILLIN",
    "AMIKACIN",
    "CEFTAZIDIME",
    "CEFOVECIN",
    "CEPHALEXIN",
    "CHLORAMPHENICOL",
    "CIPROFLOXACIN",
    "CEFPODOXIME",
    "CEFTIOFUR",
    "CEFOTAXIME",
    "DOXYCYCLINE",
    "ENROFLOXACIN",
    "GENTAMICIN",
    "MARBOFLOXACIN",
    "NITROFURANTOIN",
    "TRIMETHOPRIMSULPHATE|TRIMETHOPRIMSULFAMETHOXAZOLE",
    "IMIPENEM",
    "POLYMYXIN B",
    "AZITHROMYCIN",
    "CLINDAMYCIN",
    "ERYTHROMYCIN",
    "FLORFENICOL",
    "MINOCYCLINE",
    "MUPIROCIN",
    "MOXIFLOXACIN",
    "NEOMYCIN",
    "OXACILLINMETHICILLIN",
    "PENICILLIN",
    "RIFAMPICIN|RIFAMPIN",
    "MEROPENEM",
    "PRADOFLOXACIN",
    "BACITRACIN",
    "ORBIFLOXACIN",
    "TOBRAMYCIN",
    "FOSFOMYCIN",
    "OFLOXACIN",
    "SULFISOXAZOLE",
    "CEFAZOLIN",
    "LINEZOLID",
    "VANCOMYCIN",
    "PIPERACILLINTAZOBACTAM",
    "FUSIDIC ACID",
    "TETRACYCLINE",
    "TULATHROMYCIN",
    "AMPICILLIN",
    "CEFOPERAZONE",
    "CEFIXIME",
    "CEFOXITIN",
    "ANIDULAFUNGIN",
    "CEFIPIME",
    "CLARITHROMYCIN",
    "KANAMYCIN",
    "NOVOBIOCIN",
    "PIPERACILLIN",
    "SULFAMETHOXAZOLE",
    "TICARCILLINCLAVULANIC ACID"
)


################
# Data Loading #
################

SOURCE_IDS = None
DRUG_IDS = None
ORG_IDS = None
ASSAY_IDS = None

# Initializing values for source, drug, org, and assay
def init_source_ids():
    global SOURCE_IDS
    
    df = pd.read_csv(Config.SOURCE_IDS_FILE)

    # Unique categories with id associated
    unique_source_categories = df['source_cat1'].unique()
    unique_source_categories.sort()
    source_cat_to_ids = {k: v for v, k in enumerate(unique_source_categories)}

    SOURCE_IDS = {}
    for i, row in df.iterrows():
        SOURCE_IDS[row['source']] = source_cat_to_ids[row['source_cat1']]
        
def init_drug_ids():
    global DRUG_IDS
    
    df = pd.read_csv(Config.DRUG_IDS_FILE)

    # Unique categories with id associated
    unique_drug_categories = df['class'].unique()
    unique_drug_categories.sort()
    drug_cat_to_ids = {k: v for v, k in enumerate(unique_drug_categories)}

    DRUG_IDS = {}
    for i, row in df.iterrows():
        DRUG_IDS[row['drug']] = drug_cat_to_ids[row['class']]

def init_org_ids():
    global ORG_IDS
    
    df = pd.read_csv(Config.ORG_IDS_FILE)
    df.fillna('UNKNOWN', inplace=True)

    # Unique categories with id associated
    unique_org_categories = df['org_order'].unique()
    unique_org_categories.sort()
    org_cat_to_ids = {k: v for v, k in enumerate(unique_org_categories)}

    ORG_IDS = {}
    for i, row in df.iterrows():
        ORG_IDS[row['org_standard']] = org_cat_to_ids[row['org_order']]

# Returns the id of the assay, source, drug, or org

def init_assay_ids():
    global ASSAY_IDS
    
    df = pd.read_csv(Config.ASSAY_IDS_FILE)

    # Unique categories with id associated
    unique_assay_categories = df['assay_cat'].unique()
    unique_assay_categories.sort()
    assay_cat_to_ids = {k: v for v, k in enumerate(unique_assay_categories)}
    
    ASSAY_IDS = {}
    for i, row in df.iterrows():
        ASSAY_IDS[row['assay_name']] = assay_cat_to_ids[row['assay_cat']]

def get_source_value(val):
    # If source IDS has not been initialized, do so
    if SOURCE_IDS is None: init_source_ids()

    if val not in SOURCE_IDS:
        id_val = SOURCE_IDS['UNKNOWN']
        print(f"Unknown source: {val}")
    else:
        id_val = SOURCE_IDS[val]

    return id_val

def get_drug_value(val):
    # If drug IDS has not been initialized, do so
    if DRUG_IDS is None:
        init_drug_ids()

    if val not in DRUG_IDS:
        id_val = DRUG_IDS['UNKNOWN']
        print(f"Unknown drug: {val}")
    else:
        id_val = DRUG_IDS[val]

    return id_val

def get_org_value(val):
    # If org IDS has not been initialized, do so
    if ORG_IDS is None:
        init_org_ids()

    if val not in ORG_IDS:
        id_val = ORG_IDS['UNKNOWN']
        print(f"Unknown org: {val}")
    else:
        id_val = ORG_IDS[val]

    return id_val

def get_assay_value(val):
    # If assay IDS has not been initialized, do so
    if ASSAY_IDS is None:
        init_assay_ids()
            
    if val not in ASSAY_IDS:
        id_val = ASSAY_IDS['UNKNOWN']
        print(f"Unknown assay: {val}")
    else:
        id_val = ASSAY_IDS[val]
        
    return id_val

def get_assay_length():
    # If assay IDS has not been initialized, do so
    if ASSAY_IDS is None:
        init_assay_ids()
            
    return len(ASSAY_IDS)

def get_source_length():
    # If source IDS has not been initialized, do so
    if SOURCE_IDS is None:
        init_source_ids()
            
    return len(SOURCE_IDS)

def get_drug_length():
    # If drug IDS has not been initialized, do so
    if DRUG_IDS is None:
        init_drug_ids()
            
    return len(DRUG_IDS)

def get_org_length():
    # If org IDS has not been initialized, do so
    if ORG_IDS is None:
        init_org_ids()
            
    return len(ORG_IDS)

# Returning the list

def get_org_list():
    if ORG_IDS is None:
        init_org_ids()
        
    return ORG_IDS