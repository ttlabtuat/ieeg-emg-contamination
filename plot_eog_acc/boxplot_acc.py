import pandas as pd
from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def concat_all_acc_tsv(acc_tsv_list, task):
    all_sub_mean_acc_df = pd.DataFrame()
    
    for acc_path in acc_tsv_list:  # 各被験者のaccuracy.tsvを読み込み
        df_overt = pd.read_csv(acc_path, sep='\t')
        js_name = re.search(r'js\d{2}', acc_path).group()
        # 各chごとに平均を計算
        mean_acc = df_overt.mean(axis=0)
        # DataFrame化して列名をsubject名に
        mean_acc_df = pd.DataFrame(mean_acc)
        mean_acc_df.columns = [js_name]
        # all_sub_mean_acc_dfに追加（chがindex, subjectが列になるように）
        all_sub_mean_acc_df = pd.concat([all_sub_mean_acc_df, mean_acc_df], axis=1)

    new_index = [f"{task}_{index.replace('ch01', 'eog_h').replace('ch02', 'eog_v')}" for index in all_sub_mean_acc_df.index]
    all_sub_mean_acc_df.index = new_index
    
    return all_sub_mean_acc_df

# overt_eog_acc, covert_eog_accが既にDataFrameとして存在している前提
# データをlong形式に変換
def melt_eog(df, condition):
    df_long = df.reset_index().melt(id_vars='index', var_name='subject', value_name='accuracy')
    df_long = df_long.rename(columns={'index': 'type'})
    df_long['condition'] = condition
    return df_long


def plot_acc_boxplot(overt_df, covert_df, save_path='eog_acc_boxplot.svg'):
    """
    overt/covertのEOG精度データをボックスプロットで可視化する関数
    
    Parameters:
    -----------
    overt_df : pandas.DataFrame
        overtタスクのEOG精度データ
    covert_df : pandas.DataFrame
        covertタスクのEOG精度データ
    save_path : str
        保存するファイルパス（デフォルト: 'eog_acc_boxplot.svg'）
    """
    # データをlong形式に変換
    overt_long = melt_eog(overt_df, 'overt')
    overt_long['eog_type'] = overt_long['type'].apply(lambda x: 'HEOG' if 'eog_h' in x else 'VEOG')
    
    covert_long = melt_eog(covert_df, 'covert')
    covert_long['eog_type'] = covert_long['type'].apply(lambda x: 'HEOG' if 'eog_h' in x else 'VEOG')

    # データを結合
    plot_df = pd.concat([overt_long, covert_long], ignore_index=True)

    # プロット作成
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=plot_df, x='condition', y='accuracy', hue='eog_type', palette='Set2', showfliers=False)
    sns.stripplot(data=plot_df, x='condition', y='accuracy', hue='eog_type', 
                dodge=True, color='black', alpha=0.7, size=7, jitter=0.15, linewidth=0.5, edgecolor='gray')

    # 凡例を図の外に出す
    handles, labels = plt.gca().get_legend_handles_labels()
    n_types = plot_df['eog_type'].nunique()
    plt.legend(handles[:n_types], labels[:n_types], title='EOG type', fontsize=12, title_fontsize=13,
            bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # 軸ラベルとタイトル設定
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Task', fontsize=16)
    # plt.title('Accuracy by Overt/Covert and EOG Type', fontsize=18, pad=30)
    plt.ylim(0, 1.18)  # アスタリスク用に上下に余白を追加
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 上下に余白を追加
    
    # 表示と保存
    plt.savefig(save_path)
    # plt.show()


if __name__ == '__main__':
    overt_tsv_dir = 'results_sp_eog_2600'
    # acc_paths = sorted(glob(f'{overt_tsv_dir}/spec0_350_eog/*_accuracy.tsv'))
    acc_paths = sorted(glob(f'{overt_tsv_dir}/spec70_140_eog/*_accuracy.tsv'))
    overt_eog_acc = concat_all_acc_tsv(acc_paths, task='overt')

    covert_tsv_dir = 'results_sp_eog_2600_covert'
    acc_paths = sorted(glob(f'{covert_tsv_dir}/spec70_140_eog_covert/*_accuracy.tsv'))
    covert_eog_acc = concat_all_acc_tsv(acc_paths, task='covert')

    js_list = ['js01', 'js02', 'js04', 'js05', 'js07', 'js08', 'js11', 'js14', 'js15', 'js16']
    overt_eog_acc = overt_eog_acc.loc[:, js_list]
    covert_eog_acc = covert_eog_acc.loc[:,js_list]

    plot_acc_boxplot(overt_eog_acc, covert_eog_acc, save_path='eog_acc_boxplot_70140_notin13.svg')