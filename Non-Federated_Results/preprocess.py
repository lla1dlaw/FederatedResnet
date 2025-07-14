import pretty_errors
import pandas as pd
from pandas import DataFrame
from process import find_csv_files
import os
from pathlib import Path
import numpy as np


def process_dataframe(df: DataFrame):
    trial_data = {}
    base_metrics = [col.strip().replace('epoch_1_', '') for col in df.columns[:4]]

    for i, row in df.iterrows():
        trial_values = row.values
        reshaped_data = trial_values.reshape(-1, 4) 
        epochs = np.arange(1, len(reshaped_data) + 1)
        final_df = pd.DataFrame(reshaped_data, index=epochs, columns=base_metrics)
        final_df.index.name = 'epoch'
        trial_data[f'trial_{i}'] = final_df
    return trial_data


def main():
    unprocessed_dataframes = {}
    files = find_csv_files("./")
    for file in files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        unprocessed_dataframes[Path(file).stem] = df
    
    for dir, df in unprocessed_dataframes.items():
        os.makedirs(dir, exist_ok=True)
        processed_frames = process_dataframe(df)
        for name, trial_data in processed_frames.items():
            path = os.path.join(dir, f"{name}.csv")
            trial_data.to_csv(path, index=False)
            print(f"Saved to {path}")


if __name__ == "__main__":
    main()
