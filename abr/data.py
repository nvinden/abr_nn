import os
import pandas as pd
import numpy as np
import json
import math

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import QuantileTransformer

from .config import Config
from .ids import *

SUSCEPTIBILITY_SCORE_MATRIX = {
    #      T1,   T2,    T3
    "S": {1: -4, 2: -2, 3: -1},
    "I": {1: -1, 2:  0, 3:  1},
    "R": {1:  1, 2:  2, 3:  3}
}

class ABRDataset(Dataset):
    def __init__(self, mode : str, config : Config, data : dict, indicies : list):
        # Initialization code goes here: Load data, preprocess, etc.
        assert mode in ['train', 'val', 'test']
        
        self.config = config
        self.data = data
        self.indicies = indicies
        self.mode = mode

        super().__init__()

    def __len__(self):
        # Return the size of your dataset
        return len(self.indicies)

    def __getitem__(self, idx):
        # Logic to get a single item at index `idx`
        row_number = self.indicies[idx]
        
        player_data = self.data['player_data'][row_number]
        main_row = self.data['main'][row_number]
        spatial_data = self.data['spat'][player_data['spatial_data_key']]
        disease_data = self.data['disease_data'][row_number]
        res_score = self.data['res_score'][row_number]
        county_dist = self.data['county_dist'][player_data['county_id']]
        
        gt = self._prep_gt(res_score, disease_data)
        main = self._prep_main(main_row, disease_data)
        temp = self._prep_temporal(player_data)
        spat = self._prep_spatial(spatial_data, county_dist)
        
        gt = torch.from_numpy(gt).to(torch.float32)
        main = torch.from_numpy(main).to(torch.float32)
        temp = torch.from_numpy(temp).to(torch.float32)
        spat = torch.from_numpy(spat).to(torch.float32)
        
        return {
            'gt': gt,
            'main': main,
            'temp': temp,
            'spat': spat,
        }
    
    def _prep_gt(self, res_score, disease_data):
        gt = np.concatenate([[res_score], disease_data])
        return gt

    def _prep_main(self, main_row, disease_data):
        abr_tested = np.array(disease_data > -1, dtype=np.float32)
        main = np.concatenate([main_row, abr_tested])
        return main
    
    def _prep_temporal(self, player_data):
        temporal_feature_size = self.data["main"].shape[1] + len(COLUMN_DRUGS) + 2
        temporal_data = np.zeros(shape = (self.config.LSTM_MAX_TEMPORAL_DATA, temporal_feature_size), dtype = np.float32)
        
        n_empty_rows = self.config.LSTM_MAX_TEMPORAL_DATA - (len(player_data['curr_visits_row_nums']) + len(player_data['past_visits_row_nums']))
        n_empty_rows = max(0, n_empty_rows)
        
        past_row_rows = np.array([np.concatenate(([past_dist], self.data["main"][past_idx], [self.data["res_score"][past_idx]], self.data["disease_data"][past_idx])) for (past_idx, past_dist) in player_data["past_visits_row_nums"]], dtype=np.float32)
        curr_row_rows = np.array([np.concatenate(([0.0], self.data["main"][curr_idx], [-2.0], np.zeros(shape = (len(COLUMN_DRUGS))) - 1)) for curr_idx in player_data["curr_visits_row_nums"]], dtype=np.float32)

        if past_row_rows.shape[0] + curr_row_rows.shape[0] > self.config.LSTM_MAX_TEMPORAL_DATA:
            # First, ensure curr_row_rows does not exceed the limit
            curr_row_rows = curr_row_rows[-min(self.config.LSTM_MAX_TEMPORAL_DATA, curr_row_rows.shape[0]):]  # Get the last 'N' rows, where 'N' <= 100

            # Then, fill the remaining space with past_row_rows, prioritizing the end of the array
            rows_remaining = self.config.LSTM_MAX_TEMPORAL_DATA - curr_row_rows.shape[0]
            
            if rows_remaining == 0:
                past_row_rows = np.array([])
            else:
                past_row_rows = past_row_rows[-rows_remaining:]  # Get the last 'rows_remaining' rows
            
        # Fill with -1.0 for the empty rows
        temporal_data[:n_empty_rows, :] = -1.0

        # Fill with past data
        if len(past_row_rows) > 0:
            temporal_data[n_empty_rows:n_empty_rows + len(past_row_rows)] = past_row_rows

        # Fill with current data
        if len(curr_row_rows) > 0:
            temporal_data[n_empty_rows + len(past_row_rows):n_empty_rows + len(past_row_rows) + len(curr_row_rows)] = curr_row_rows
            
        return temporal_data
        
    def _prep_spatial(self, spatial_data, county_dist):
        county_dist = np.concatenate([county_dist, [0.0]], axis=0)
        county_dist = np.reshape(county_dist, (-1, 1))
        spatial_data = np.concatenate([county_dist, spatial_data], axis = 1)
        
        return spatial_data
        

class ABRDataModule(LightningDataModule):
    # Data paths

    # Data preparation organization:
    # For each patient ID, there is a dictionary of data
    # split into main, temporal, and spatial data.
    # self.data = {
    #    'main': {numpy.array (n_rows, n_main_features)},
    #    'res_score': {numpy.array (n_rows)},
    #    'county_dist': np.array(n_counties, n_counties),
    #    'spat': {
    #       'year_month1': {numpy.array (n_counties + 1, n_spat_features)},
    #       'year_month2': {numpy.array (n_counties + 1, n_spat_features)},
    #     },
    #    'player_data: {
    #        'id1': {
    #            'temp': list(ints), # Contains the row numbers of the temporal data
    #            'spat': list(ints), # Contains the row numbers of the spatial data
    #            'drugs': dict{str : str}, # Contains the list of drugs that the disease was tested on and the result
    #         },      
    #           ... 
    #    }
    # }
    # 

    # If running from a data save load that.
    # If not running from a data save, do all of the data preparation
    # for the entire dataset and then split it into train/val/test.
    def __init__(self, config : Config, data_path : str = None):
        super().__init__()
        
        self.config = config
        self.data_path = data_path

        # Data paths
        self.AMR_DATA_PATH = os.path.join(self.data_path, "AMR_Data.csv")
        self.DRUG_TIERS_PATH = os.path.join(self.data_path, "Drug_Tiers.csv")
        self.COUNTY_LOCATIONS_PATH = os.path.join(self.data_path, "county_locations_normalized.json")
        self.TTV_SPLIT_PATH = os.path.join(self.data_path, "ttv_split.json")
        self.PREPARE_DATA_SAVE = os.path.join(self.data_path, "prepared_data.npz")
        
        self.prepare_data()

        if config.USE_NEW_TTV_SPLIT or os.path.isfile(self.TTV_SPLIT_PATH) == False:
            # Create a new train/val/test split
            ttv_splits = self._create_ttv_split_file()
            
            # Save ttv_splits
            with open(self.TTV_SPLIT_PATH, 'w') as f:
                json.dump(ttv_splits, f, indent=4)
        else:
            with open(self.TTV_SPLIT_PATH) as f:
                ttv_splits = json.load(f)
        
        self.train_indicies = ttv_splits['train']
        self.test_indicies = ttv_splits['test']
        self.val_indicies = ttv_splits['val']    

    def _create_ttv_split_file(self):
        # Create a new train/val/test split
        df = pd.read_csv(self.AMR_DATA_PATH)

        indexes = df["id"].unique().tolist()
        ttv_ratio = self.config.DATA_SPLIT
        
        # Shuffle the indexes randomly
        np.random.shuffle(indexes)

        # Calculate the number of elements in each set
        train_size = int(ttv_ratio[0] * len(indexes))
        test_size = int(ttv_ratio[1] * len(indexes))
        val_size = len(indexes) - train_size - test_size

        # Split the indexes into train, test, and validation sets
        train_indexes = indexes[:train_size]
        test_indexes = indexes[train_size:train_size + test_size]
        val_indexes = indexes[train_size + test_size:]
        
        train_rows = df[df["id"].isin(train_indexes)]["Unnamed: 0"].to_list()
        test_rows = df[df["id"].isin(test_indexes)]["Unnamed: 0"].to_list()
        val_rows = df[df["id"].isin(val_indexes)]["Unnamed: 0"].to_list()
        
        return {
            'train': train_rows,
            'test': test_rows,
            'val': val_rows,
        }

    # Takes data from AMR_DATA_PATH and DRUG_TIERS_PATH and prepares it
    def prepare_data(self):
        if self.config.PREPARE_DATA or os.path.isfile(self.PREPARE_DATA_SAVE) == False:
            data = self.tokenize_data()
            
            # Save the prepared data
            self.save_data(data)
        else:
            data = self.load_data() 
        self.data = data
    
    def tokenize_data(self):

        # Loading the data
        amr_df = self._get_amr_df()
        
        #self._print_unique_columns(amr_df, 'source')
        
        main_data = self._get_main_data(amr_df)
        res_score_data = self._get_res_score_data(amr_df)
        spat_data = self._get_spat_data(amr_df)
        county_dist_data = self._get_county_dist_data()
        player_org_data = self._get_player_org_data(amr_df)
        disease_data = self._get_disease_data(amr_df)
        
        data = {
            'main': main_data,
            'res_score': res_score_data,
            'county_dist': county_dist_data,
            'spat': spat_data,
            'player_data': player_org_data,
            'disease_data': disease_data,
        }

        return data


    # Helper functions for prepare_data
    def _get_amr_df(self):
        amr_df = pd.read_csv(self.AMR_DATA_PATH, low_memory=False)
        
        for col in ["panel_name", "assay_name", "source", "species"]:
            amr_df[col].fillna("OTHER", inplace=True)
            
        amr_df["age_year"] = amr_df["age_year"].replace("< 1 year", "0")
            
        return amr_df

    # main data structure:
    # [county_id (x 50), year (x4), month (x 12), normalized_year_month (1), species (1), normalized_age (1),
    # panel_name_id (x __), assay_name_id (x __), soruce_id (x __), drug_id (x __),]
    def _get_main_data(self, amr_df : pd.DataFrame):
        data = amr_df.to_dict("records")
        
        data_list = []
        
        for i, row in enumerate(data):
            panel_name_id = get_source_value(row['source'])
            org_name_id = get_org_value(row['org_standard'])

            county_id = COUNTY2ID[row['county']]
            order_year = row['order_year']
            order_month = row['order_month']
            normalized_year_month = ((order_year - 2019) * 12 + order_month) / (4 * 12)
            species = SPECIES2ID[row['species']]
            age = np.array(row['age_year']).astype(np.float32)
            panel_name_id = get_source_value[row['panel_name']]
            assay_name_id = ASSAYNAME2ID[row['assay_name']]
            source_id = SOURCE2ID[row['source']]
            
            county_id = np.eye(len(COUNTY2ID))[county_id]
            order_year = np.eye(4)[order_year - 2019]
            order_month = np.eye(12)[order_month - 1]
            normalized_year_month = np.array([normalized_year_month], dtype = np.float32)
            species = np.array([species], dtype = np.float32)
            age = np.array([-1 if np.isnan(age) else age / 18], dtype = np.float32)
            panel_name_id = np.eye(N_PANELNAMES)[panel_name_id]
            assay_name_id = np.eye(N_ASSAYNAMES)[assay_name_id]
            source_id = np.eye(N_SOURCES)[source_id]
            
            row_data = np.concatenate( [county_id, order_year, order_month, normalized_year_month,
                                        species, age, panel_name_id, assay_name_id, source_id], 
                                        dtype = np.float32)
            data_list.append(row_data)
            
        data_list = np.stack(data_list)
        
        return data_list

    def _get_res_score_data(self, amr_df : pd.DataFrame):
        data = amr_df.to_dict("records")
        
        data_list = []
        
        #abr_drug_base_score = self._get_abr_drug_base_score(amr_df)
        
        for i, row in enumerate(data):
            result_dict = dict()
            
            for ab in COLUMN_DRUGS:
                result = row[ab]
                
                # result is nan
                if result not in ["S", "R", "I"]:
                    continue
                
                tier = DRUG_TIERS[ab] if ab in DRUG_TIERS else 1
                
                result_dict[ab] = (result, tier)
                
            resistency_raw_score = self.calculate_resistency_score(result_dict)
            data_list.append(resistency_raw_score)
            
        resistency_mean = np.mean(data_list)
        resistency_std = np.std(data_list)
        
        resistency_score = (np.array(data_list) - resistency_mean) / resistency_std
        
        #print(f"Resistency mean: {resistency_mean}")
        #print(f"Resistency std: {resistency_std}")
        #print(f"Resistency max: {resistency_score.max()}")
        #print(f"Resistency min: {resistency_score.min()}")
        
        # Quantile Transformation
        data = np.array([resistency_score]).reshape(-1, 1)  # Replace with your list of values
        transformer = QuantileTransformer(output_distribution='normal', random_state=0)
        transformed_data = transformer.fit_transform(data)
        transformed_resistency = transformed_data.flatten()
        
        # Plot the resistency score histogram
        #import matplotlib.pyplot as plt
        #plt.hist(transformed_resistency, bins=100)
        #plt.show()
        
        return transformed_resistency

    def _get_spat_data(self, amr_df : pd.DataFrame):
        data = amr_df.to_dict("records")
        
        resistency_scores = self._get_res_score_data(amr_df)
        
        months_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        years_list = [2019, 2020, 2021, 2022]
        
        month_year = [(month, year) for month in months_list for year in years_list]
        
        res_results = dict()
        disease_count_results = dict()
        
        all_resistencies = []
        state_resistencies = []
        
        disease_count_monthly_average = dict()
        for county in COUNTY2ID.keys():
            county_df = len(amr_df[amr_df["county"] == county]) / len(month_year)
            disease_count_monthly_average[county] = county_df
        
        final_results = dict()
        
        for (month, year) in month_year:
            month_year_key = f"{month}_{year}"
            res_results[month_year_key] = {}
            disease_count_results[month_year_key] = {}
            
            for county, county_id in COUNTY2ID.items():
                current_df = amr_df[(amr_df["county"] == county) & (amr_df["order_month"] == month) & (amr_df["order_year"] == year)]
                #current_df = current_df.to_dict("records")
                
                cur_ids = current_df["Unnamed: 0"].to_list()
                cur_resistencies = resistency_scores[cur_ids]
                resistency = np.sum(cur_resistencies) if len(cur_resistencies) > 0 else 0.0
                
                res_results[month_year_key][county_id] = resistency
                all_resistencies.append(resistency)
                
                # Disease count
                disease_count = len(current_df)
                disease_count_results[month_year_key][county_id] = disease_count / disease_count_monthly_average[county]
                
            current_df = amr_df[(amr_df["order_month"] == month) & (amr_df["order_year"] == year)]
            cur_ids = current_df["Unnamed: 0"].to_list()
            cur_resistencies = resistency_scores[cur_ids]
            resistency = np.sum(cur_resistencies)
            
            res_results[month_year_key][len(COUNTY2ID)] = resistency
            state_resistencies.append(resistency)
            
            # Disease count
            disease_count = len(current_df)
            disease_count_results[month_year_key][len(COUNTY2ID)] = disease_count / (disease_count_monthly_average[county] * 12)
            
            #print(f"{month}/{year}: {resistency}")
                    
        mean_resistency = np.mean(all_resistencies)
        std_resistency = np.std(all_resistencies)
        
        state_mean_resistency = np.mean(state_resistencies)
        state_std_resistency = np.std(state_resistencies)
        
        for ym_key in res_results.keys():
            for county_key in res_results[ym_key].keys():
                if county_key == len(COUNTY2ID): # Total_county
                    res_results[ym_key][county_key] = (res_results[ym_key][county_key] - state_mean_resistency) / state_std_resistency
                else:
                    res_results[ym_key][county_key] = (res_results[ym_key][county_key] - mean_resistency) / std_resistency 
                    
            res_results_np = np.array([res_results[ym_key][county_key] for county_key in sorted(res_results[ym_key].keys())])
            disease_count_results_np = np.array([disease_count_results[ym_key][county_key] for county_key in sorted(disease_count_results[ym_key].keys())])
            combined = np.stack([res_results_np, disease_count_results_np], axis=1)
            final_results[ym_key] = combined
                
        return final_results
    
    def _get_county_dist_data(self):
        # Open the county locations file
        with open(self.COUNTY_LOCATIONS_PATH) as f:
            county_locations = json.load(f)
            
        county_distances = np.zeros((len(COUNTY2ID), len(COUNTY2ID)))
            
        for county_1, c1_loc in county_locations.items():
            for county_2, c2_loc in county_locations.items():
                distance = math.sqrt((c1_loc[0] - c2_loc[0]) ** 2 + (c1_loc[1] - c2_loc[1]) ** 2)
                
                county_id_1 = COUNTY2ID[county_1]
                county_id_2 = COUNTY2ID[county_2]
                
                county_distances[county_id_1][county_id_2] = distance
                
        return county_distances
            
    def _get_player_org_data(self, amr_df : pd.DataFrame):
        amr_rows = amr_df.to_dict("records")
        df = amr_df.copy()
        
        player_org_data = dict()
        
        for row in amr_rows:
            player_id = row["id"]
            county_id = COUNTY2ID[row["county"]]
            spatial_data_key = f"{row['order_month']}_{row['order_year']}"
            row_number = row["Unnamed: 0"]
            
            # Converting the DataFrame columns and the current year and month to integers for comparison
            df['order_year'] = df['order_year'].astype(int)
            df['order_month'] = df['order_month'].astype(int)
            curr_year = int(row["order_year"])
            curr_month = int(row["order_month"])

            # Getting all past visits from each row
            past_tests_df = df[((df['order_year'] < curr_year) | ((df['order_year'] == curr_year) & (df['order_month'] < curr_month))) &
                            (df["id"] == player_id) &
                            (df["Unnamed: 0"] != row_number)]
            
            # Getting the current visits from each row
            curr_tests_df = df[(df["id"] == player_id) &
                            (df["Unnamed: 0"] != row_number) &
                            (df["order_year"] == curr_year) &
                            (df["order_month"] == curr_month)]
            
            if len(past_tests_df) != 0:
                past_visits = past_tests_df["Unnamed: 0"].to_list()
            else:
                past_visits = []
                
            if len(curr_tests_df) != 0:
                curr_visits = curr_tests_df["Unnamed: 0"].to_list()
            else:
                curr_visits = []
                
            # Ordering the past visits and saving the normalized time between visits
            if len(past_visits) > 0:
                past_years = np.array(past_tests_df["order_year"].to_list())
                past_months = np.array(past_tests_df["order_month"].to_list())
                normalized_time = ((curr_year - past_years) * 12 + (curr_month - past_months)) / (12 * 4)
                past_visits = list(zip(past_visits, normalized_time))
                past_visits = sorted(past_visits, key=lambda x: x[1], reverse=True)
            
            player_data = {
                "player_id": player_id,
                "county_id": county_id,
                "spatial_data_key": spatial_data_key,
                "past_visits_row_nums": past_visits,
                "curr_visits_row_nums": curr_visits,
            }
            
            player_org_data[row_number] = player_data
            
        return player_org_data
    
    def _get_disease_data(self, amr_df : pd.DataFrame):
        data = amr_df.to_dict("records")
        
        data_list = []
        
        for row in data:
            tested_drugs = []
            
            for ab in COLUMN_DRUGS:
                result = row[ab]
                
                # result is nan
                if result in ["S", "R", "I"]:
                    if result == "S":
                        tested_drugs.append(0)
                    elif result == "R":
                        tested_drugs.append(1)
                    else:
                        tested_drugs.append(0.5)
                else:
                    tested_drugs.append(-1)
                    
            tested_drugs = np.array(tested_drugs, dtype=np.float32)
            data_list.append(tested_drugs)
            
        data_list = np.stack(data_list)
        
        return data_list       

    # Returns a dictionary of the form:
    # {
    #   "Drug1": # of resistencies / # of samples,
    #   "Drug2": # of resistencies / # of samples,
    #   "Drug3": # of resistencies / # of samples,
    #    ...
    # }
    def _get_abr_drug_base_score(self, amr_df):
        abr_base_score = dict()
        
        for drug_name in COLUMN_DRUGS:
            counts = amr_df[drug_name].value_counts().to_dict()
            count_R = counts["R"] if "R" in counts else 0
            count_S = counts["S"] if "S" in counts else 0
            count_I = counts["I"] if "I" in counts else 0
            
            assert count_R + count_S + count_I > 30
            
            score = (count_R + 0.5 * count_I) / (count_R + count_S + count_I)
            abr_base_score[drug_name] = score
            
            #print(f"{drug_name}: {count_R + count_S + count_I}")
            
        for (drug, tier) in list(DRUG_TIERS.items()):
            if drug in list(abr_base_score.keys()):
                #print(f"{drug}: {abr_base_score[drug]}: {tier}")  
                pass
        
        return abr_base_score

    def _print_unique_columns(self, df : pd.DataFrame, col_name : str):
        print(col_name + ":")
        unique_vals = df[col_name].unique()

        print("{")
        for id, val in enumerate(unique_vals):
            print(f"   '{val}': {id},")
        print("}")

        print()
        
    # Takes in data in the form:
    # {
    #   "Drug1": ("S", 1),
    #   "Drug2": ("R", 2),
    #   "Drug3": ("I", 3),
    #    ...
    # }
    # Returns a resistency score [0, 1]
    # 0 = no resistency
    # 1 = full resistency
    def calculate_resistency_score(self, result_dict):
        score = 0
        
        if len(result_dict) == 0:
            return 0.0
        
        for _, result in result_dict.items():
            result, tier = result
            score += SUSCEPTIBILITY_SCORE_MATRIX[result][tier]
            
        return float(score) / len(result_dict)

    # Loads the prepared data from PREPARE_DATA_SAVE
    def load_data(self):
        data = np.load(self.PREPARE_DATA_SAVE, allow_pickle=True)['data'][()]
        return data
    
    def save_data(self, data):
        np.savez(self.PREPARE_DATA_SAVE, data=data)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.custom_train = ABRDataset(mode='train', config=self.config, data=self.data, indicies=self.train_indicies)
            self.custom_val = ABRDataset(mode='val', config=self.config, data=self.data, indicies=self.val_indicies)
        if stage == 'test' or stage is None:
            self.custom_test = ABRDataset(mode='test', config=self.config, data=self.data, indicies=self.test_indicies)

    def train_dataloader(self):
        return DataLoader(self.custom_train, batch_size = self.config.BATCH_SIZE, shuffle = True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.custom_val, batch_size = self.config.BATCH_SIZE, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.custom_test, batch_size = self.config.BATCH_SIZE, drop_last=True)