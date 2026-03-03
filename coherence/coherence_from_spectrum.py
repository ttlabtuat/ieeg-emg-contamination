import os
from glob import glob
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
import mne
from mne_connectivity import spectral_connectivity_epochs, seed_target_indices

import sys
import os
sys.path.append('junten-ibids-preprocess')
from common import create_all_run_epochs, get_all_run_bidspath

def calc_coherence(epochsspectrum, eog_ch_name, method, fmin=None, fmax=None):

    # スペクトル接続性（コヒーレンス）を計算
    picks = [eog_ch_name] + [ch for ch in epochsspectrum.ch_names if ch.startswith('ch')]   # rightがEOG, chXXがiEEG
    # epochs_data = epochsspectrum.get_data(picks=picks)  # EOGとEEGのデータを取得
    epochsspectrum = epochsspectrum.copy().pick(picks)

    seeds = [0]
    targets = list(range(1, len(picks)))  # EOGを除くEEGチャンネルのインデックス
    indices = seed_target_indices(seeds=seeds, targets=targets)
    print(f"Indices for coherence calculation: {indices}")

    con = spectral_connectivity_epochs(
        epochsspectrum,    # (n_epochs, n_channels, n_times)
        # method='coh',  # コヒーレンス
        method=method,  # 位相遅れインデックス
        indices=indices,  # EOG(0)と各EEG(1,2,...)の組み合わせ
        fmin=fmin,
        fmax=fmax,
        sfreq=epochsspectrum.info['sfreq'],
    )
    return con  # 
    
# def get_mask_power_line_freqs(con, powerline_freq=50):
#     """
#     コヒーレンスの結果からパワーライン周波数のマスクを取得する関数
#     Get a mask for power line frequencies from coherence results.
    
#     Args:
#         con (Connectivity): コヒーレンスの結果 / Coherence results
#         power_line_freqs (list): パワーライン周波数のリスト / List of power line frequencies
#     Returns:
#         np.ndarray: パワーライン周波数のマスク / Mask for power line frequencies
#     """
#     freqs = con.freqs  # 周波数の配列
#     resolution = freqs[1] - freqs[0]  # 周波数の分解能
#     offset_1hz = int(2 / resolution)  # 2Hzのオフセット（分解能に基づく）
#     # 電源ノイズ周波数のインデックスを特定
#     powerline_harmonics = np.arange(powerline_freq, freqs[-1]+1, powerline_freq)
#     exclude_idx = []
#     for harm in powerline_harmonics:    
#         idx = np.argmin(np.abs(freqs - harm))   # 最も近い周波数のインデックスを取得
#         # offset_1hzに基づいて前後±1Hz分を除外
#         exclude_idx.extend(list(range(idx - offset_1hz, idx + offset_1hz + 1)))
#     exclude_idx = list(set([i for i in exclude_idx if 0 <= i < len(freqs)]))    # 重複を削除し、範囲外のインデックスを除外（ex.600Hzの場合，: 601など）を除外

#     # 除外する周波数にマスクを適用
#     mask = np.ones(len(freqs), dtype=bool)
#     mask[exclude_idx] = False
    
#     return mask, np.array(freqs)

# # プロット用の補助関数（MNE 1.9対応）
# def plot_coherence_with_exclusion(con, freqs, mask, eeg_labels):
#     """除外された周波数を視覚化（MNE 1.9対応）"""
    
#     # コヒーレンスデータを取得
#     coherence_data = con.get_data()  # shape: (n_connections, n_freqs)
    
#     plt.figure(figsize=(12, 8))
    
#     for i, ch_name in enumerate(eeg_labels):
#         # 全周波数をプロット（除外部分は赤）
#         plt.subplot(len(eeg_labels), 1, i+1)
        
#         # i番目の接続のコヒーレンス
#         coh_values = coherence_data[i, :]
        
#         plt.plot(freqs[mask], coh_values[mask], 'b-', label='Valid frequencies')
#         plt.plot(freqs[~mask], coh_values[~mask], 'r.', label='Excluded (power line)')
#         plt.ylabel(f'iEEG-{ch_name}')
#         plt.grid(True, alpha=0.3)
#         if i == 0:
#             plt.legend()
    
#     plt.xlabel('Frequency (Hz)')
#     plt.suptitle('EOG-EEG Coherence with Power Line Noise Exclusion')
#     plt.tight_layout()
#     plt.show()

# def plot_box(csv_path, plot_ch_names=False):
#     """
#     CSVファイルからコヒーレンス値の箱ひげ図と各点をプロットする関数
#     """
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     df = pd.read_csv(csv_path, index_col=0)
#     plt.figure(figsize=(12, 6))
#     # 箱ひげ図
#     sns.boxplot(data=df, whis=[0, 100], width=0.5, fliersize=0) # 外れ値を表示しないようにfliersizeを0に設定
#     # 各点を重ねてプロット（swarmplotで重ならないようにする）
#     for i, col in enumerate(df.columns):
#         y = df[col].values
#         x = np.random.normal(i, 0.08, size=len(y))  # x位置を少しランダムにずらす
#         plt.plot(x, y, 'o', color='black', alpha=0.5, markersize=5)
#         # index（行名）を各点の近くに小さい字で表示
#         if plot_ch_names:
#             for xi, yi, idx in zip(x, y, df.index):
#                 plt.text(xi, yi, str(idx), fontsize=7, color='gray', ha='left', va='bottom', alpha=0.7)
#                 save_path = csv_path.replace('.csv', '_boxplot_ch_names.png')
#         else:
#             save_path = csv_path.replace('.csv', '_boxplot.png')

#     plt.title('Coherence Mean Boxplot')
#     plt.xticks(ticks=range(len(df.columns)), labels=df.columns, rotation=45)
#     plt.ylim(0, 0.3)
#     plt.grid(True)  # グリッドを表示
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()

def main(save_name, eog_ch_name, method, js_list, low, high, task):
    all_coh_df = pd.DataFrame()  # 全ての被験者のコヒーレンス平均を保存するデータフレーム
    root='/Users/murakamishoya/git/tt-lab/tt_github/junten-ibids-preprocess/junten_bids'
    tmin = -0.1
    tmax = 2.5
    bandwidth = 4  # マルチターパー法の帯域幅
    fmin = 0
    fmax = 600

    for js_name in js_list:

        all_bids_path = get_all_run_bidspath(js_name, task='Speech8sen', root=root)
        
        overt_epochs = create_all_run_epochs(all_bids_path, event_names=[task], tmin=tmin, tmax=tmax, baseline=None)
        ieeg_names = [ch for ch in overt_epochs.ch_names if ch.startswith('ch')]
        sfreq = int(overt_epochs.info['sfreq'])

        # spectrum_welch = overt_epochs.compute_psd(
        #     picks='all', method='welch', fmin=fmin, fmax=fmax, n_fft=sfreq, 
        #     output='complex', n_overlap=(sfreq//4)*3,
        # )
        spectrum_multitap = overt_epochs.compute_psd(
            picks='all', method='multitaper', fmin=fmin, fmax=fmax, bandwidth=bandwidth, 
            output='complex'
        )

        # print("================================")
        # print(f"Welch spectrum for {js_name} computed with shape: {spectrum_multitap.get_data(picks='all').shape}")
        # print("================================")

        # all_epochs.plot(picks='all', block=True)
        con = calc_coherence(spectrum_multitap, eog_ch_name, method, fmin=low, fmax=high)
        # コヒーレンスの結果を保存
        save_path = f"{save_name}/{save_name}_{js_name}.npz"
        np.savez(save_path, con=con.get_data(), freqs=con.freqs, ch_names=ieeg_names)

        # # 平均の計算（電源ノイズの影響を考慮）
        # mask, freqs = get_mask_power_line_freqs(con, powerline_freq=overt_epochs.info['line_freq'])
        # con_avg = con.get_data()[:, mask].mean(axis=1)
        # con_avg_series = pd.Series(con_avg, index=ieeg_names, name=f"{js_name}_mean_coh")
        # # データフレームに保存
        # all_coh_df[f"{js_name}_mean_coh"] = con_avg_series
        # print(all_coh_df)
    
    # 保存
    all_coh_df.to_csv(f"{save_name}/{save_name}_{eog_ch_name}_{method}_all_mean.csv")
        
        # print(np.array(freqs)[~mask])

        # plot_coherence_with_exclusion(
        #     con, freqs, mask, eeg_labels=ieeg_names)
        # コヒーレンスの結果を保存



if __name__ == "__main__":
    js_list = ("js01", "js02", "js04", "js05", "js07", "js08", "js11", "js13", "js14", "js15", "js16")
    # js_list = ("js01", )
    low = 0
    high = 600
    method = "coh"
    eog_ch_name = "right"
    task = "overt"
    save_name = f"{method}_{low}_{high}_ep2600_{eog_ch_name}_{task}"
    os.makedirs(save_name, exist_ok=True,)

    main(save_name, eog_ch_name, method, js_list, low, high, task)
    # plot_box(f'{save_name}/{save_name}_all_mean.csv', plot_ch_names=True)
    
