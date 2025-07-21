import pretty_errors
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import argparse
from process import find_root_dirs
import os
import traceback
from matplotlib.ticker import MultipleLocator


def get_metrics_from_user(summaries: dict[str, DataFrame]) -> list[str]:
    sample_key = list(summaries.keys())[0]
    sample_df = summaries[sample_key]
    headers = sample_df.columns.tolist()
    choices = [header for header in headers if header != 'epoch']
    usr_choice = "h"
    
    while usr_choice == "h":
        print("\nChoose Which Metrics You Want To Plot (h for help):")
        for i, choice in enumerate(choices, 1):
            print(f"{i}) {choice}")
        

        usr_choice = input(">> ")
        if usr_choice.strip() == 'h':
            print("-- Type the numbers to the left of each choice that you want to plot separated by spaces. --")
            continue
        
        try:
            usr_choice = usr_choice.split(' ')
            idxs = [int(num) for num in usr_choice]
            for idx in idxs:
                if idx not in range(1, len(choices)+1):
                    raise ValueError()
        except ValueError:
            print("Please enter a number from the menu.")
            usr_choice = "h"
        except Exception:
            print("Bad input. Type h for help.")
            usr_choice = "h"
    idxs.sort()
    return [choices[idx-1] for idx in idxs]


def gather_all_summaries(parent: str='./', summary_filename: str='summary.csv'):
    roots = find_root_dirs(parent)
    roots = [root for root in roots if root not in ('__pycache__', 'plots')]
    summaries = {}
    for root_dir in roots:
        try: 
            summaries[root_dir] = pd.read_csv(os.path.join(parent, root_dir, summary_filename))
        except Exception as e:
            print(f"Error Collecting Summaries:")
            traceback.print_exc()
            exit()

    return summaries


def plot_dataframes(dataframe_dict, metrics, plot_title, marker_frequency=20, y_limit=(0.0, 1.02), y_tick_interval=0.05):
    """
    Plots corresponding values from a dictionary of pandas DataFrames.

    Args:
        dataframe_dict (dict): A dictionary of pandas DataFrames, indexed by strings.
        metrics (list): A list of column names (metrics) to plot from each DataFrame.
        marker_frequency (int): The frequency of markers on the plot. For example,
                                a value of 2 will place a marker on every other point.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    marker_styles = ['o', 's', 'D', '^', 'v', 'p', '*', 'X']

    for i, (df_name, df) in enumerate(dataframe_dict.items()):
        color = colors[i % len(colors)]
        
        if not all(metric in df.columns for metric in metrics):
            print(f"Warning: DataFrame '{df_name}' is missing one or more metrics. Skipping.")
            continue

        for j, metric_name in enumerate(metrics):
            marker = marker_styles[j % len(marker_styles)]
            ax.plot(df.index+1, df[metric_name], marker=marker, linestyle='-', 
                    color=color, label=f'{df_name} - {metric_name}', 
                    markevery=marker_frequency)

    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.yaxis.set_major_locator(MultipleLocator(base=y_tick_interval))
    ax.set_ylim(y_limit)
    plt.xticks(rotation=45, ha='right')
    ax.legend(loc='lower right')
    plt.tight_layout()
    return plt



def save_plot(path: str, filename: str, plot):
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    plot.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")



def main(**kwargs):
    summary_filename = kwargs.get('filename', 'summary.csv')
    plots_directory = kwargs.get('plt_dir', 'plots')
    plot_name = 'WS-Arithmetic-Acc_Comparison-Learn_Imag.png'

    summaries = gather_all_summaries(summary_filename=summary_filename)
    metrics_to_plot = get_metrics_from_user(summaries)
    print(f"\nGenerating Plot For Metrics: {metrics_to_plot}\n")
    plot = plot_dataframes(summaries, metrics_to_plot, kwargs['title'])
    save_plot(plots_directory, plot_name, plot)
    plot.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="Figure", help="Title of the generated Plot")
    parser.add_argument("--filename", type=str, default='summary.csv', 
                           help="Each model's directory contains a file with summary stats calculated as an average for each metric over all trials.\nIf the name of this summary file was not changed when running process.py, leave this blank.\nOtherwise put the name of the summary file here. Defaults to 'summary.csv'.")
    parser.add_argument("--plt-dir", type=str, default='./plots/federated_plots', help="Subdirectory to save plots to.")
    args = vars(parser.parse_args())
    main(**args)
