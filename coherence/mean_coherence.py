import os
from glob import glob
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_epochs, seed_target_indices
import os

def get_mask_power_line_freqs(freqs, offset_hz=4, powerline_freq=50):
    """
    コヒーレンスの結果からパワーライン周波数のマスクを取得する関数

    Args:
        freqs (np.ndarray): 周波数の配列
        offset_hz (float): 除外する周波数幅（Hz単位, 例: 2Hzなら±1Hz）
        powerline_freq (float): パワーライン周波数（既定値: 50Hz）

    Returns:
        mask (np.ndarray): パワーライン周波数を除外したマスク（True: 使用, False: 除外）
        freqs (np.ndarray): 周波数の配列（そのまま返す）
    """
    resolution = freqs[1] - freqs[0]  # 周波数の分解能
    offset = int(offset_hz / resolution)  # 2Hzのオフセット（分解能に基づく）
    # 電源ノイズ周波数のインデックスを特定
    powerline_harmonics = np.arange(powerline_freq, freqs[-1]+1, powerline_freq)
    exclude_idx = []
    for harm in powerline_harmonics:    
        idx = np.argmin(np.abs(freqs - harm))   # 最も近い周波数のインデックスを取得
        # offsetに基づいて前後±1Hz分を除外
        exclude_idx.extend(list(range(idx - offset, idx + offset + 1)))
    exclude_idx = list(set([i for i in exclude_idx if 0 <= i < len(freqs)]))    # 重複を削除し、範囲外のインデックスを除外（ex.600Hzの場合，: 601など）を除外

    # 除外する周波数にマスクを適用
    mask = np.ones(len(freqs), dtype=bool)
    mask[exclude_idx] = False
    masked_freqs = freqs[mask]
    
    return mask, masked_freqs

def calc_mean_coherence(
    path,
    low_freq=150,
    high_freq=350,
    bandwidth=4,
    powerline_freq=50
):
    """
    指定したnpzファイルからコヒーレンスデータを読み込み、
    指定周波数範囲・電源ノイズ除去後の平均コヒーレンスを計算する関数

    Args:
        path (str): npzファイルのパス
        low_freq (float): 下限周波数
        high_freq (float): 上限周波数
        bandwidth (float): 電源ノイズ除去の帯域幅（Hz）
        powerline_freq (float): 電源周波数（Hz）

    Returns:
        mean_avg (np.ndarray): 各チャンネルの平均コヒーレンス
        masked_freqs (np.ndarray): 除外後の周波数配列
        con_avg (np.ndarray): 除外後のコヒーレンスデータ
    """
    data = np.load(path)
    cropped_idx = (data['freqs'] >= low_freq) & (data['freqs'] <= high_freq)
    cropped_freqs = data['freqs'][cropped_idx]
    croped_data = data['con'][:, cropped_idx]

    mask, masked_freqs = get_mask_power_line_freqs(
        cropped_freqs, offset_hz=bandwidth, powerline_freq=powerline_freq
    )

    con_avg = croped_data[:, mask]
    
    return con_avg, masked_freqs

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

if __name__ == "__main__":
    js_list = ("js01", "js02", "js04", "js05", "js07", "js08", "js11", "js13", "js14", "js15", "js16")
    coh_dir = 'coh_0_600_ep2600_right_overt'
    bandwidth=4
    powerline_freq=50
    freq_list = [0, 70, 140, 210, 280, 350]
    
    for l_f, h_f in zip(freq_list[:-1], freq_list[1:]):
        all_coh_df = pd.DataFrame() # 新しいDataFrameを初期化

        for js_name in js_list:
            # path = os.path.join(coh_dir, f'{coh_dir}_{js_name}.npz')
            path = os.path.join(coh_dir, f'{coh_dir}_{js_name}.npz')

            con_avg, masked_freqs = calc_mean_coherence(
                path,
                low_freq=l_f,
                high_freq=h_f,
                bandwidth=bandwidth,
                powerline_freq=powerline_freq
            )
            mean_avg = np.mean(con_avg, axis=1)
            
            # 電極名を取得
            ch_names = np.load(path)['ch_names']
            
            # Seriesに変換し、DataFrameに追加
            mean_coh_series = pd.Series(mean_avg, index=ch_names, name=js_name)
            all_coh_df = pd.concat([all_coh_df, mean_coh_series], axis=1)

        # 結果をCSVファイルとして保存
        output_filename = os.path.join(coh_dir, f'{coh_dir}_{l_f}_{h_f}_all_mean.csv')
        all_coh_df.to_csv(output_filename)
        print(f"すべての参加者の平均コヒーレンスが {output_filename} に保存されました。")
