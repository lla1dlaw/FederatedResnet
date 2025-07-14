import pretty_errors
import pandas as pd
import argparse
import os
from rich.console import Console
from pathlib import Path


def find_root_dirs(directory: str):
    all_contents = os.listdir(directory)
    dirs = [item for item in all_contents if os.path.isdir(item)]
    return dirs

def find_csv_files(folder_path):
    """
    Finds all files with the '.csv' extension in a specified folder.

    Args:
        folder_path (str): The path to the folder to search.

    Returns:
        list: A list of filenames ending with '.csv'.
              Returns an empty list if the folder does not exist or contains no CSV files.
    """
    # Check if the provided path is a valid directory
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return []

    try:
        # List all files and directories in the given folder
        all_files = os.listdir(folder_path)
        
        # Filter the list to include only files that end with '.csv'
        csv_files = [file for file in all_files if file.endswith('.csv') and os.path.isfile(os.path.join(folder_path, file))]
        
        return csv_files
    except OSError as e:
        print(f"Error accessing the folder: {e}")
        return []


def load_dataframes(roots: str) -> dict[str, dict[str, pd.DataFrame]]:
    all_dataframes = {}

    for root in roots:
        files = find_csv_files(root)
        print("CSV Files Found:")
        for file in files:
            print(f"- {file}")
        print()
        dataframes = {}

        console = Console()
        with console.status("Loading CSV Files", spinner='dots') as status:
            try:
                for file in files:
                    filename_no_ext = Path(file).stem
                    dataframes[filename_no_ext] = pd.read_csv(os.path.join(root, file))
                    status.update("[green]Done")
            except Exception as e:
                status.update("[red]Failed")
                print(f"{e}")
                exit()
        all_dataframes[root] = dataframes

    return all_dataframes


def save_summary_data(name: str, data: dict[str, pd.DataFrame]) -> None:
    path='' # for scope
    try:
        for root, df in data.items():
            path = os.path.join(root, 'summary.csv')
            df.to_csv(path, index=False)
            print(f"Successfully saved to {path}")
    except Exception as e:
        print(f"Error saving to file {path}:\n{e}")
        exit()


def get_summary_stats(data_dict: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    averaged_dfs: dict[str, pd.DataFrame] = {}
    for name, data in data_dict.items():
        console = Console()

        with console.status("Calculating Summary Statistics", spinner='dots') as status:
            sum_df = sum(data.values())
            averaged_df = sum_df/len(data)
            status.update("[green]Done.")
        
        averaged_dfs[name] = averaged_df
    
    return averaged_dfs

def main(**kwargs):
    load_dir = kwargs['load_path']
    save_name = kwargs['save_name']
    load_paths = find_root_dirs(load_dir)

    for load_path in load_paths:
        try:
            assert(os.path.exists(load_path))
        except AssertionError:
            print(f"File not found at: {load_path}")
            exit()

    dataframes = load_dataframes(load_paths)
    summary_dfs = get_summary_stats(dataframes)
    for summary in summary_dfs:
        print(f"Summary Data:\n{summary}")

    save_summary_data(save_name, summary_dfs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", type=str, default='./', help="Path to directory containing subdirectories for each model. Subdirectories contain trials as csv files.")
    parser.add_argument("--save-name", type=str, default='summary.csv', help="Name to assign summary files. Devaults to summary.csv")
    args = vars(parser.parse_args())
    main(**args)
