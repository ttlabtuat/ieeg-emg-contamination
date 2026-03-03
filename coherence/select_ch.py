import yaml
import pandas as pd
import numpy as np

def set_nan_unusable_ch(df, js_name, usable_ch_list):
    # column=js_name=のみインデックスがusable_ch_listに含まれないものを削除
    df.loc[~df.index.isin(usable_ch_list), js_name] = np.nan
    return df

if __name__ == "__main__":
    eog_ch_name = "right"
    task = "overt"
    dir = f"coh_0_600_ep2600_{eog_ch_name}_{task}"
    l_f = 70
    h_f = 140

    csv_path = f"{dir}/coh_0_600_ep2600_{eog_ch_name}_{task}_{l_f}_{h_f}_all_mean.csv"
    df = pd.read_csv(csv_path, index_col=0)

    js_list = ("js01", "js02", "js04", "js05", "js07", "js08", "js11", "js13", "js14", "js15", "js16")
    for js_name in js_list:
        yaml_path = f"js_yamls/{js_name}.yaml"
        
        
        with open(yaml_path, 'r') as file:
            js_yaml = yaml.load(file, Loader=yaml.FullLoader)
            usable_ch_list = [f"ch{ch:02d}" for ch in js_yaml['usable_ch']]

        df = set_nan_unusable_ch(df, js_name, usable_ch_list)

    df.to_csv(f"{dir}/{dir}_{l_f}_{h_f}_all_mean_usable_ch.csv")