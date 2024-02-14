# !pip install comtrade
import comtrade
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os

from tqdm.notebook import tqdm
from tqdm.notebook import trange

''' Code below based on Tutorial ["DATASETS & DATALOADERS"](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#datasets-dataloaders) from Pytorch.org

#### Creating a Custom Dataset

A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
'''


# Osc stands for Oscillogram
class CustomOscDataset(Dataset):
   
    def __init__(self, osc_dir):
        if not os.path.exists(osc_dir):
            raise FileNotFoundError(f"Path '{osc_dir}' does not exist.")
        self.osc_dir = osc_dir
        self.osc_files = sorted([file for file in os.listdir(self.osc_dir)
                          if "cfg" in file], reverse=False)                
    
    def __len__(self):        
        return len(self.osc_files)
    
    def __getitem__(self, idx):        
        max_idx = len(self.osc_files)
        if idx not in [*range(0, max_idx)]:
            raise IndexError(f"Index is out of range [0, {max_idx - 1}]")                     
        self.path = os.path.join(self.osc_dir, self.osc_files[idx])
        return comtrade.load(self.path)

def get_unique_labels(columns: np.array):
    ML_cols = [col for col in columns if 'signal' in col]
    labels = sorted(set([int(col.split('_')[2]) for col in ML_cols]))
    return ML_cols, labels
    
def logical_or(row):
    return int(np.any(row.values))

def combine_columns(ML_cols, labels, df: pd.DataFrame):
    for label in labels:
        col_combine = [col for col in ML_cols if int(col.split('_')[2]) == label]
        df['MLsignal_' + str(label)] = df[col_combine].apply(logical_or, axis=1)
    return df

def correct_columns(columns: list):
    '''
    Function is used to change 'Mlsignal' to 'MLsignal'  
    '''
    
    prefix = 'MLsignal_'
    mistyped_columns = [column for column in columns if 'Mlsignal' in column]
    correct_cols = {column: prefix + column.split('_', maxsplit=1)[1]
                       for column in mistyped_columns}
    if mistyped_columns:
        print('\nmistyped_columns:', mistyped_columns)
        print('correct_cols:', correct_cols)
    return correct_cols
    

def get_dataframe(osc_folder, unique_features, columns_to_rename, skip_normal=False):
    
    '''
    Input:
    osc_folder - path to oscillograms
    
    Output: pandas dataframe obtained from oscillograms
    '''
    
    # declare an object of the CustomOscDataset class
    dataset = CustomOscDataset(osc_folder)    
    
    # Initial columns may contain 'BB', 'BB' needs to be dropped
    cols_to_rename_init_osc = {'IA 1ВВ': 'IA1',
                               'IC 1ВВ': 'IC1',
                               'IA 2ВВ': 'IA2',
                               'IC 2ВВ': 'IC2'}
    
    
    # Columns with index '1'
    columns_idx_1 = ['IA1', 'IC1',
                     'UA1СШ', 'UB1СШ',
                     'UC1СШ', 'UN1СШ',
                     'UN2СШ','Пуск осциллографа'] 

    # Columns with index '2'
    columns_idx_2 = ['IA2', 'IC2',
                     'UA2СШ','UB2СШ',
                     'UC2СШ', 'UN2СШ',
                     'UN1СШ', 'Пуск осциллографа']
        
    assert len(unique_features) == 10, f'Something went wrong: number of features {len(unique_features)}, must be 10'
    
    # dataframe to add data from oscillograms
    to_del= {key: 'delete_me' for key in unique_features}
    dataframe = pd.DataFrame(to_del, index=['delete_me']) # 'delete_me' line here is only to avoid warning
    for i in trange(len(dataset.osc_files), desc=f'Working on oscillograms'):        
        file_name = dataset.osc_files[i]
        print(f'{i}\t{file_name} is in progress', end='...\t')
        
        # create pandas DataFrame from comtrade object
        df_1 = dataset[i].to_dataframe()
        
        # if there are not any MLsignal columns in a file, we will not use it
        if skip_normal:
            if not 'signal' in ' '.join(df_1.columns):    
                print(f'There is not any disturbancies in the {file_name}.\tSkipped...')
                continue
        
        # create a column 'time' from the index of the DataFrame
        df_1.reset_index(drop=False, inplace=True)
     
        # to use for checking the correctness
        init_first_line_1 = df_1.iloc[0][1:6].values    # initial values of IA1, IC1, UA1СШ, UB1СШ, UC1СШ
        init_last_line_2  = df_1.iloc[-1][7:12].values  # initial values of IA2, IC2, UA2СШ, UB2СШ, UC2СШ
            
        # rename columns:
        # 'IA 1ВВ'->'IA1', 'IC 1ВВ'->'IC1' etc
        if 'IA 1ВВ' in df_1.columns:
            df_1.rename(columns=cols_to_rename_init_osc, inplace=True)
        
        # substitute: 'Mlsignal' -> 'MLsignal'
        if 'Mlsignal' in ' '.join(df_1.columns):
            correct_cols = correct_columns(df_1.columns)
            df_1.rename(columns=correct_cols, inplace=True)
        
        # create a deep copy of modified DataFrame
        df_2 = df_1.copy(deep=True)
        
        # add column with the oscillogram file name
        df_1.insert(0, 'file_name',  [file_name + '_1'] * df_1.shape[0])
        df_2.insert(0, 'file_name',  [file_name + '_2'] * df_2.shape[0])
        
        # these columns will be dropped from df_1
        columns_to_drop_1 = [column for column in df_1.columns
                             if column in columns_idx_2
                             or 'MLsignal_2_' in column] 
        
        # these columns will be dropped from df_2
        columns_to_drop_2 = [column for column in df_2.columns
                             if column in columns_idx_1
                             or 'MLsignal_1_' in column]
        
        # drop columns
        df_1.drop(columns=columns_to_drop_1, inplace=True)
        df_2.drop(columns=columns_to_drop_2, inplace=True)
        
        # combine columns with equal disturbances label into one column, using logical OR
        ML_cols, labels = get_unique_labels(columns=df_1.columns.values)
        df_1 = combine_columns(ML_cols, labels, df_1)
        df_1.drop(columns=ML_cols, inplace=True)
                
        ML_cols, labels = get_unique_labels(columns=df_2.columns.values)
        df_2 = combine_columns(ML_cols, labels, df_2)
        df_2.drop(columns=ML_cols, inplace=True)

        
       
        # rename columns before concatenation
        # 'IA1' -> 'IA', 'IC1' -> 'IC', 'IA2' -> 'IA' - drop indexes
        # 'UA2СШ' -> 'UA', 'UB2СШ' -> 'UB'            - drop indexes
        # 'MLsignal_12_1_1' -> 'MLsignal_12_1_1'      - do not drop section index '12' if MLsignal is in both sections
        # 'MLsignal_2_3': 'MLsignal_3'                - drop section index '2'
         
            
        df_1.rename(columns=columns_to_rename, inplace=True)
        df_2.rename(columns=columns_to_rename, inplace=True)
        
        
      
        # to use for checking the correctness
        modified_first_line_1 = df_1.iloc[0][2:7].values  # values of IA, IC, UA, UB, UC
        modified_last_line_2  = df_2.iloc[-1][2:7].values # values of IA, IC, UA, UB, UC

        # check the correctness
        assert np.all(init_first_line_1 == modified_first_line_1)  and \
               np.all(init_last_line_2  == modified_last_line_2)

         
        # concatenate df_1 'vertically' (axis=0)
        dataframe = pd.concat([dataframe, df_1], axis=0, ignore_index=False)
        # concatenate df_2 'vertically' (axis=0)
        dataframe = pd.concat([dataframe, df_2], axis=0, ignore_index=False)
        print('Completed!')
    dataframe.drop('delete_me', inplace=True)
    print('\nDataframe is ready!')
    # all NaN values = 0
    return dataframe.fillna(value=0)


def get_target(dataframe, all_ML, window_size=32, step=1):
    '''
    Function to get targets for dataframe by sliding window usage
    '''
    
    target = []
    files = dataframe['file_name'].unique()
    ML_1, ML_2, ML_3 = all_ML[0], all_ML[1], all_ML[2]

    for file in tqdm(files):
        sub_df = dataframe[dataframe['file_name'] == file]
        for i in range(window_size, sub_df.shape[0], step):
            window = sub_df.iloc[i - window_size: i]            
            label_1, label_2, label_3 = 0, 0, 0 
            if window[ML_1].values.sum() == window_size:
                label_1 = 1.0                    
            if window[ML_2].values.sum() == window_size:                    
                label_2 = 1.1
            if window[ML_3].values.sum() > 0:    
                label_3 = 2.2 * window[ML_3].mean()                
            labels = [0.0, label_1, label_2, label_3]
            target.append(np.argmax(labels))
    print('Completed!')
    return pd.Series(target)

class MyDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 target: pd.Series,
                 window_size: int):
        
        self.df = df
        self.target = target
        self.window_size = window_size

        window_end_indices = []
        run_ids = df.index.get_level_values(0).unique()
        for run_id in tqdm(run_ids, desc='Creating sequence of samples'):
            indices = np.array(df.index.get_locs([run_id]))
            indices = indices[self.window_size:]
            window_end_indices.extend(indices)
        self.window_end_indices = np.array(window_end_indices)

    def __len__(self):
        return len(self.window_end_indices)
    
    def __getitem__(self, idx):
        window_index = self.window_end_indices[idx]
        sample = self.df.values[window_index - self.window_size:window_index]
        target = self.target.values[idx]
        return sample.astype(np.float32), target

