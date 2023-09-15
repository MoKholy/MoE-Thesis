import torch
import numpy as np
import os 
import pandas as pd
import torch.nn.functional as F
import gc
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

LABEL2INT = {'Benign': 0,
 'Bot': 1,
 'Brute Force -Web': 2,
 'Brute Force -XSS': 3,
 'DDOS attack-HOIC': 4,
 'DDOS attack-LOIC-UDP': 5,
 'DDoS attacks-LOIC-HTTP': 6,
 'DoS attacks-GoldenEye': 7,
 'DoS attacks-Hulk': 8,
 'DoS attacks-SlowHTTPTest': 9,
 'DoS attacks-Slowloris': 10,
 'FTP-BruteForce': 11,
 'Infilteration': 12,
 'SQL Injection': 13,
 'SSH-Bruteforce': 14
}

INT2LABEL = {0: 'Benign',
 1: 'Bot',
 2: 'Brute Force -Web',
 3: 'Brute Force -XSS',
 4: 'DDOS attack-HOIC',
 5: 'DDOS attack-LOIC-UDP',
 6: 'DDoS attacks-LOIC-HTTP',
 7: 'DoS attacks-GoldenEye',
 8: 'DoS attacks-Hulk',
 9: 'DoS attacks-SlowHTTPTest',
 10: 'DoS attacks-Slowloris',
 11: 'FTP-BruteForce',
 12: 'Infilteration',
 13: 'SQL Injection',
 14: 'SSH-Bruteforce'
}


COLS_TO_DROP = ["DNS_QUERY_ID", "FTP_COMMAND_RET_CODE","IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "Label"]
### HELPER FUNCTONS TO LOAD LARGE CSV FILES##

# function to reduce memory usage
# link https://stackoverflow.com/questions/57531266/how-do-i-read-a-large-csv-file-in-google-colab

def process(df):
    # loop over dataframe cols
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            # get min
            col_min = df[col].min()
            col_max = df[col].max()

            # check if col type is int
            if str(col_type)[:3] == "int":
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.uint8).min and col_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.uint16).min and col_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif col_min > np.iinfo(np.uint32).min and col_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif col_min > np.iinfo(np.uint64).min and col_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            # Convert "L7_PROTO" column to uint16
            elif col == "L7_PROTO":
                df[col] = df[col].astype(np.uint16)

            elif str(col_type)[:5] =="float":
                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    # Replace zero or negative values with a small positive value before taking the logarithm
                    small_value = 1e-10
                    df[col] = np.where(df[col] <= 0, small_value, df[col])
                    df[col] = np.log(df[col]).astype(np.float32) 
            
    return df


def reduce_mem_usage(file_name, drop_cols = True, map_labels=True):
    gc.collect()
    # create an iterator over df
    df_iter = pd.read_csv(file_name, chunksize=10000)
    df_list = []

    for df in df_iter:
        if drop_cols:
            df = df.drop(columns=COLS_TO_DROP)
        if map_labels:
            df["Attack"] = df["Attack"].map(lambda x: LABEL2INT[x]).astype(np.int8)
        df_list.append(process(df))

    df_list = pd.concat(df_list)
    gc.collect()
    mem = df_list.memory_usage().sum()/1024**2
    print(f"Memory usage after processing: {mem}")
    return df_list


class GatingDataset(Dataset):
    def __init__(self, dataset_name, dataset_path = "data/processed/", mapping = INT2LABEL, inv_mapping = LABEL2INT, split = "train", seed=10, transform=True):
        
        self.file_path = os.path.join(dataset_path, f"{seed}" ,f"{split}_" + dataset_name)
        self.transform = transform
        self.data = reduce_mem_usage(self.file_path, drop_cols=False, map_labels=False)
        
        if mapping != INT2LABEL:
            self.mapping = mapping
            self.inv_mapping = inv_mapping
        else:
            self.mapping = INT2LABEL
            self.inv_mapping = LABEL2INT
        if transform:
            scaler = StandardScaler()
            cols = self.data.columns[:-1]
            self.data[cols] = scaler.fit_transform(self.data[cols])
        self.num_features = len(self.data.columns[:-1])
        # get number of unique values in attack column
        self.num_classes = np.unique(self.data["Attack"]).shape[0]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.data.iloc[idx, :-1]
        
        if type(idx) is int:
            label = torch.tensor([self.data.iloc[idx, -1]], dtype=torch.int64)
        else:
            label = torch.tensor(self.data.iloc[idx, -1].to_list(), dtype=torch.int64)
        
        
        gate_label = F.one_hot(label, num_classes=15).float()
        
        sample = torch.tensor(sample.values).float()
        # debugging
        # print(f"sample shape: {sample.shape}")
        # print(f"label shape: {label.shape}")
        # print(f"gate label shape: {gate_label.shape}")
        
        return sample, gate_label, label
    
    def get_mapping(self):
        return self.mapping
    
    def get_inv_mapping(self):
        return self.inv_mapping
    
    def get_input_dim(self):
        return self.num_features
    
    def get_num_classes(self):
        return self.num_classes
    
