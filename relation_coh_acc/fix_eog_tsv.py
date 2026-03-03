from glob import glob
from operator import index
import pandas as pd

for path in glob("to_plot_tsv_results_sp_eog_2600/*.tsv"):
    df = pd.read_csv(path, sep='\t', index_col=0)
    df = df.rename(index={'ch01': 'overt_eog_h', 'ch02': 'overt_eog_v'})
    df.to_csv(path.replace(".tsv", ".csv"))