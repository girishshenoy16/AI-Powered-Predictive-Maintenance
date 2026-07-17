import pandas as pd
import numpy as np
import os
import glob

def get_columns():
    return ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor_{i}' for i in range(1, 22)]

def process_train_data(input_dir="data/raw", output_path="data/processed/train_processed.csv"):
    print(f"\nScanning for train datasets in {input_dir}...")
    all_files = glob.glob(os.path.join(input_dir, "train_FD*.txt"))
    if not all_files:
        print(f"Error: No train_FD*.txt files found in {input_dir}")
        return None
        
    columns = get_columns()
    df_list = []
    
    for file in all_files:
        dataset_name = os.path.basename(file).split('.')[0]
        print(f"Processing {dataset_name}...")
        try:
            df = pd.read_csv(file, sep='\s+', header=None, names=columns)
            df['engine_id'] = f"{dataset_name}_" + df['engine_id'].astype(str)
            df_list.append(df)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            
    if not df_list: return None
    full_df = pd.concat(df_list, ignore_index=True)
    
    max_cycles = pd.DataFrame(full_df.groupby('engine_id')['cycle'].max()).reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    full_df = full_df.merge(max_cycles, on=['engine_id'], how='left')
    full_df['RUL'] = full_df['max_cycle'] - full_df['cycle']
    full_df.drop('max_cycle', axis=1, inplace=True)
    full_df['label'] = np.where(full_df['RUL'] <= 30, 1, 0)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_csv(output_path, index=False)
    print(f"Processed Train data saved to {output_path}")
    return full_df

def process_test_data(input_dir="data/raw", output_path="data/processed/test_processed.csv"):
    print(f"\nScanning for test and RUL datasets in {input_dir}...")
    all_test_files = glob.glob(os.path.join(input_dir, "test_FD*.txt"))
    if not all_test_files:
        print(f"Error: No test_FD*.txt files found in {input_dir}")
        return None
        
    columns = get_columns()
    df_list = []
    
    for file in all_test_files:
        dataset_name = os.path.basename(file).split('.')[0]
        rul_file_name = dataset_name.replace("test", "RUL") + ".txt"
        rul_path = os.path.join(input_dir, rul_file_name)
        
        if not os.path.exists(rul_path):
            print(f"Warning: RUL file {rul_path} not found. Skipping {dataset_name}.")
            continue
            
        print(f"Processing {dataset_name} and mapping true RUL...")
        try:
            # Read Test Data
            test_df = pd.read_csv(file, sep='\s+', header=None, names=columns)
            
            # Read RUL Data
            rul_df = pd.read_csv(rul_path, sep='\s+', header=None, names=['true_RUL'])
            rul_df['engine_num'] = rul_df.index + 1  # 1-indexed engine ID
            
            # Max cycles in test set
            max_cycles = pd.DataFrame(test_df.groupby('engine_id')['cycle'].max()).reset_index()
            max_cycles.columns = ['engine_id', 'max_cycle']
            
            test_df = test_df.merge(max_cycles, on=['engine_id'], how='left')
            test_df = test_df.merge(rul_df, left_on='engine_id', right_on='engine_num', how='left')
            
            # True RUL calculation: (max_cycle - current_cycle) + true_RUL_at_end
            test_df['RUL'] = (test_df['max_cycle'] - test_df['cycle']) + test_df['true_RUL']
            test_df['label'] = np.where(test_df['RUL'] <= 30, 1, 0)
            
            test_df.drop(['max_cycle', 'true_RUL', 'engine_num'], axis=1, inplace=True)
            
            # Global engine ID
            test_df['engine_id'] = f"{dataset_name}_" + test_df['engine_id'].astype(str)
            df_list.append(test_df)
        except Exception as e:
            print(f"Failed to process {file}: {e}")
            
    if not df_list: return None
    full_test_df = pd.concat(df_list, ignore_index=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_test_df.to_csv(output_path, index=False)
    print(f"Processed Test data saved to {output_path}")
    return full_test_df
