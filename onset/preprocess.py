from scipy import signal, stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple, Optional, List

class Preprocess:
    @staticmethod
    def downsampling(input_data, original_sf: int, re_sf: int, lowpass_order=6):
        downsampled_data = signal.resample(input_data, int(len(input_data) * re_sf / original_sf), axis=-1)
        return downsampled_data
    
    @staticmethod
    def highpass_filter(input_data, re_sf: int, cutoff=0.5, order=4):
        nyquist = 0.5 * re_sf
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high')
        filtered_data = signal.filtfilt(b, a, input_data, axis=-1)
        return filtered_data

    @staticmethod
    def notch_filter(input_data, re_sf: int, notch_freq=50):
        nyquist = 0.5 * re_sf
        normal_notch = notch_freq / nyquist
        b, a = signal.iirnotch(normal_notch, Q=5)
        filtered_data = signal.filtfilt(b, a, input_data, axis=-1)
        return filtered_data
    
    @staticmethod
    def get_stft(x, fs, clip_fs=-1, **kwargs):
        f, t, Zxx = signal.stft(x, fs, **kwargs)
    
        Zxx = Zxx[:clip_fs]
        f = f[:clip_fs]
            
        clip = 5 #To handle boundary effects
        Zxx = Zxx[:,clip:-clip]
        t = t[clip:-clip]

        if np.isnan(Zxx).any():
            import pdb; pdb.set_trace()

        return f, t, Zxx

    @staticmethod
    def get_power_spectrogram(Zxx):
        """スペクトログラムのパワースペクトルを計算する関数"""
        Zxx = np.abs(Zxx) ** 2
        return Zxx

    @staticmethod
    def log_spectrogram(Zxx):
        """スペクトログラムをdBスケールに変換する関数"""
        Zxx = np.log10(np.abs(Zxx) + 1e-100)
        return Zxx

    @staticmethod
    def normalize_spectrogram(Zxx, normalizing="zscore_all", mask_hz_list=None):
        """スペクトログラムの正規化を行う関数"""
        if normalizing == "zscore_all":
            if mask_hz_list is not None:
                # マスク領域を一時的に0に置き換え
                masked_Zxx = Zxx.copy()
                # for start_hz, end_hz in mask_hz_list:
                #     masked_Zxx[start_hz:end_hz] = 0
                
                # マスク領域以外の部分で平均と標準偏差を計算
                non_zero_mask = masked_Zxx != 0
                mean_val = np.mean(masked_Zxx[non_zero_mask])
                std_val = np.std(masked_Zxx[non_zero_mask])
                
                # 全体を標準化
                Zxx = (Zxx - mean_val) / std_val
            else:
                # マスク領域がない場合は全体を標準化
                Zxx = (Zxx - np.mean(Zxx)) / np.std(Zxx)
        # BrainBERTのよくわからないやつ
        # elif normalizing == "zscore_axis1":
        #     Zxx = stats.zscore(Zxx, axis=-1)
        # elif normalizing == "baselined":
        #     Zxx = self.baseline(Zxx)
        # elif normalizing == "db":
        #     Zxx = np.log2(Zxx)
        return Zxx

    @staticmethod
    def hz_to_idx(hz: float) -> int:
        """周波数（Hz）をインデックスに変換する関数"""
        # return np.abs(freq_axis - hz).argmin()
        return hz // 5  # 400Hzのスペクトログラムを想定しているため、5Hzごとにインデックスを割り当てる

    @staticmethod
    def mask_spectrogram(spec: np.ndarray, mask_hz_list: List[tuple]) -> np.ndarray:
        """スペクトログラムの特定の周波数帯域をマスクする関数"""
        if mask_hz_list is None:
            print("Masking is not applied.")
            return spec

        print(f"Masking frequencies: {mask_hz_list}")
        masked_spec = spec.copy()
        for mask_start, mask_end in mask_hz_list:
            start_idx = Preprocess.hz_to_idx(mask_start)
            end_idx = Preprocess.hz_to_idx(mask_end)
            masked_spec[start_idx:end_idx, :] = 0
        return masked_spec
    
    @staticmethod
    def plot_spectrograms(f, t, Sxx):
        """スペクトログラムをプロットする関数"""
        plt.figure(figsize=(15))
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.title('Original Signal Spectrogram')
        plt.ylabel('Frequency [Hz]')

if __name__ == "__main__":
    # Example usage
    ecog_path = "path/to/ecog.npy"
    print(f"Processing ECoG data from: {ecog_path}")

    # preprocess
    ecog = np.load(ecog_path)
    ecog = Preprocess.downsampling(ecog, original_sf=9600, re_sf=2048)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1, 1]})

    # 1. Original ECoG Signal
    time = np.arange(ecog.shape[0]) / pp.preprocess_config.re_sf
    axes[0].plot(time, ecog[:, 10], label='Original ECoG Signal')
    axes[0].set_title('Original ECoG Signal')
    axes[0].set_xlabel('Time [s]')  # 単位をsamplesからsに変更
    axes[0].set_ylabel('Amplitude [uV]')
    axes[0].set_xlim(time[0], time[-1])  # x軸の範囲をデータの開始から終了までに設定
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    # ダミーのスペースを追加
    dummy = np.zeros((1, 1))
    plt.colorbar(plt.pcolormesh(dummy, cmap='Greys'), cax=cax0, label='dummy')

    # 2. bb spectrogram
    f, t, Zxx = pp.get_stft(ecog[:, 10], fs=pp.preprocess_config.re_sf, clip_fs=80, nperseg=pp.preprocess_config.win_len, noverlap=pp.preprocess_config.win_step, return_onesided=True)
    Zxx = np.abs(Zxx)
    zscore_axis1 = stats.zscore(Zxx, axis=-1)
    axes[1].pcolormesh(t, f, zscore_axis1, shading='none')
    axes[1].set_title('Z-Score Normalized axis 1 Spectrogram (previous)')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Frequency [Hz]')
    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(axes[1].collections[0], cax=cax1, label='Amplitude')
    
    # 3. zscore spectrogram
    # 4. Notch Filtered Spectrogram
    ecog_high = pp._highpass_filter(ecog)
    ecog_notch = pp._notch_filter(ecog_high)
    f, t, Zxx = pp.get_stft(ecog_notch[:, 10], fs=pp.preprocess_config.re_sf, clip_fs=80, nperseg=pp.preprocess_config.win_len, noverlap=pp.preprocess_config.win_step, return_onesided=True)
    Zxx = pp.get_power_spectrogram(Zxx)  # パワースペクトルを計算
    Zxx = pp.log_spectrogram(Zxx)  # dBスケールに変換
    zscore_Zxx = pp.normalize_spectrogram(Zxx)  # 標準化
    im4 = axes[2].pcolormesh(t, f, zscore_Zxx, shading='none')
    axes[2].set_title('Notch Filtered and High-pass Filtered ECoG Signal Spectrogram (new)')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Frequency [Hz]')
    divider3 = make_axes_locatable(axes[2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im4, cax=cax3, label='Power')

    plt.tight_layout()
    plt.suptitle('Spectrograms of ECoG Signals', fontsize=16)
    plt.subplots_adjust(top=0.9)  # タイトルのスペースを確保

    plt.savefig(f'power_spectrogram_{pp.subject_config.name}_old_new.png')


