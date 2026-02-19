import os
import re
import pandas as pd
from glob import glob

def get_mean_acc_df(csv_list):
    mean_ter_list = []
    col_list = []
    for csv in csv_list:
        # 正規表現を使用して "js" の後に続く数字を検索
        match = re.search(r'js(\d+)', csv)
        if match:
            js = int(match.group(1))

        df = pd.read_csv(csv, index_col=0)
        # print(js, len(df))
        mean_ter_list.append(df[['mean_acc']])
        col_list.append(f"js{js:02d}")

    mean_ter_df = pd.concat(mean_ter_list, axis=1)
    mean_ter_df.columns = col_list
    return mean_ter_df

def _calculate_mean_accuracy(path):
    """Calculate the mean accuracy for each channel and save it to a new TSV file."""
    df = pd.read_csv(path, sep='\t')
    js_name = path.split('/')[-1].split('_')[0]
    
    # Calculate the mean accuracy for each channel
    mean_accuracy_series = df.mean()
    # mean_accuracy.columns = [subject_id]
    # mean_accuracy.index = [subject_id]

    return js_name, mean_accuracy_series

def get_mean_acc_df_from_tsv(clean_tsv_list):
    """Calculate mean accuracy for all subjects from clean TSV files.
    
    Args:
        clean_tsv_list (list): List of paths to clean TSV files
        
    Returns:
        pd.DataFrame: DataFrame containing mean accuracy for all subjects
    """
    all_subjects_acc = pd.DataFrame()
    for clean_tsv in clean_tsv_list:
        js_name, mean_accuracy = _calculate_mean_accuracy(clean_tsv)
        df = pd.DataFrame({js_name: mean_accuracy})
        all_subjects_acc = pd.concat([all_subjects_acc, df], axis=1)
        all_subjects_acc = all_subjects_acc.sort_index(axis=0)
    
    return all_subjects_acc

def get_melt_df(all_subject_df):
    """Melt the DataFrame to long format for easier plotting.
    
    Args:
        all_subject_df (pd.DataFrame): DataFrame containing mean accuracy for all subjects
        
    Returns:
        pd.DataFrame: Melted DataFrame in long format
    """
    # melted_df = all_subject_df.reset_index().melt(id_vars='index', var_name='js', value_name='mean_acc')
    # melted_df.rename(columns={'index': 'subject'}, inplace=True)

    melted_df = pd.melt(all_subject_df)
    melted_df.columns = ["Participant", "acc"]

    return melted_df.dropna()

def get_melt_df_from_csv(all_subject_file):
    """Load a CSV file and melt it to long format.
    
    Args:
        all_subject_file (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Melted DataFrame in long format
    """
    if all_subject_file.endswith('.csv'):
        all_subject_df = pd.read_csv(all_subject_file, index_col=0)
    elif all_subject_file.endswith('.tsv'):
        all_subject_df = pd.read_csv(all_subject_file, sep='\t', index_col=0)
    
    return get_melt_df(all_subject_df)

def save_all_subjects_acc_to_tsv(in_tsv_path, out_dir):
    """
    Save all subjects' accuracy from TSV files to a single TSV file.
    Args:
        in_tsv_path (str): Path to the input TSV files.
        out_dir (str): Directory to save the output TSV file.
    """
    # Extract the experiment directory name from the input path
    exp_dir_name = os.path.dirname(in_tsv_path).split('/')[-1]
    tsv_list = sorted(glob(in_tsv_path))

    all_subjects_acc = get_mean_acc_df_from_tsv(tsv_list)
    # Create output directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Check if the file already exists    
    if os.path.exists(f"{out_dir}/{exp_dir_name}.tsv"):
        print(f"File {out_dir}/{exp_dir_name}.tsv already exists. Skipping save.")
    else:
        all_subjects_acc.to_csv(f"{out_dir}/{exp_dir_name}.tsv", sep='\t', index=True)
        print(f"Saved {exp_dir_name} to {out_dir}/{exp_dir_name}.tsv")


if __name__ == "__main__":
    out_dir = "to_plot_tsv_sp_eog_2600"

    for dir in sorted(glob("results_sp_eog_2600/*")):
        if os.path.isdir(dir):
            print(f"Processing directory: {dir}")
            in_tsv_path = os.path.join(dir, "*_accuracy.tsv")
            print(f"Found TSV files: {in_tsv_path}")

            save_all_subjects_acc_to_tsv(in_tsv_path, out_dir)