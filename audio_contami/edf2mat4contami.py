
import numpy as np
import mne
from mne.io import Raw
from mne import Epochs
from scipy.io import savemat

def raw2ndarray(raw: Raw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    rawデータからeeg, voice, sound_r, eogのnumpy配列を抽出して返す
    Extract eeg, voice, and sound_r numpy arrays from raw data
    """
    # 1. 神経データ（eeg）を抽出
    # sEEGまたはECoGチャンネルがあるかチェック
    has_seeg = any(ch_type == 'seeg' for ch_type in raw.get_channel_types())
    has_ecog = any(ch_type == 'ecog' for ch_type in raw.get_channel_types())
    
    if has_seeg:
        neural_picks = mne.pick_types(raw.info, seeg=True, stim=False)
    elif has_ecog:
        neural_picks = mne.pick_types(raw.info, ecog=True, stim=False)
    else:
        # sEEGもECoGもない場合はEEGにフォールバック
        neural_picks = mne.pick_types(raw.info, eeg=True, stim=False)
    
    eeg = raw[neural_picks, :][0]
    
    # 2. voiceチャンネルを抽出
    voice_picks = ['voice']
    voice_idx = np.where(np.isin(raw.ch_names, voice_picks))[0]
    if len(voice_idx) > 0:
        voice = raw[voice_idx, :][0]
    else:
        raise ValueError("voiceチャンネルが見つかりません / voice channel not found")
    
    # 3. sound_rチャンネルを抽出
    sound_r_picks = ['sound_r']
    sound_r_idx = np.where(np.isin(raw.ch_names, sound_r_picks))[0]
    if len(sound_r_idx) > 0:
        sound_r = raw[sound_r_idx, :][0]
    else:
        raise ValueError("sound_rチャンネルが見つかりません / sound_r channel not found")

    # 4. eogチャンネルを抽出
    # eog_picks = mne.pick_types(raw.info, eog=True, stim=False)
    # eog = raw[eog_picks, :][0]
    eog_picks = ['right', 'r_up']
    eog_idx = np.where(np.isin(raw.ch_names, eog_picks))[0]
    if len(eog_idx) > 0:
        eog = raw[eog_idx, :][0]
    else:
        raise ValueError("eogチャンネルが見つかりません / eog channel not found")
    
    return eeg, voice, sound_r, eog

def epochs2ndarray(epochs: Epochs) -> list[list[np.ndarray]]:
    """
    epochsからeeg, voice, sound_r, eogのnumpy配列を抽出して返す
    args:
        epochs: MNE Epochs object. ex. (80, ch, time)
    returns:
        list of numpy arrays. ex. [[eeg, voice, sound_r, eog], [eeg, voice, sound_r, eog], ...], len(list) = n_epochs
    """
# 1. 神経データ（eeg）を抽出
    # sEEGまたはECoGチャンネルがあるかチェック
    has_seeg = any(ch_type == 'seeg' for ch_type in epochs.get_channel_types())
    has_ecog = any(ch_type == 'ecog' for ch_type in epochs.get_channel_types())
    
    if has_seeg:
        neural_picks = mne.pick_types(epochs.info, seeg=True, stim=False)
    elif has_ecog:
        neural_picks = mne.pick_types(epochs.info, ecog=True, stim=False)
    else:
        # sEEGもECoGもない場合はEEGにフォールバック
        neural_picks = mne.pick_types(epochs.info, eeg=True, stim=False)
    
    # 2. voiceチャンネルを抽出
    voice_picks = ['voice']
    voice_idx = np.where(np.isin(epochs.ch_names, voice_picks))[0]
    
    # 3. sound_rチャンネルを抽出
    sound_r_picks = ['sound_r']
    sound_r_idx = np.where(np.isin(epochs.ch_names, sound_r_picks))[0]

    # 4. eogチャンネルを抽出
    # eog_picks = mne.pick_types(raw.info, eog=True, stim=False)
    # eog = raw[eog_picks, :][0]
    eog_picks = ['right', 'r_up']
    eog_idx = np.where(np.isin(epochs.ch_names, eog_picks))[0]

    epoch_list = []
    for epoch in epochs:
        eeg = epoch[neural_picks, :] # (ch, time)
        if len(voice_idx) > 0:
            voice = epoch[voice_idx, :]
        else:
            raise ValueError("voiceチャンネルが見つかりません / voice channel not found")
        if len(sound_r_idx) > 0:
            sound_r = epoch[sound_r_idx, :]
        else:
            raise ValueError("sound_rチャンネルが見つかりません / sound_r channel not found")
        if len(eog_idx) > 0:
            eog = epoch[eog_idx, :]
        else:
            raise ValueError("eogチャンネルが見つかりません / eog channel not found")

        epoch_list.append([eeg, voice, sound_r, eog])
    return epoch_list


def ndarray2mat(data, sf, filename: str) -> None:
    """
    numpy配列をmatファイルに保存する
    Save numpy arrays to mat file

    .matの構造（これに合わせて変換）
    - fs: 1 x 1
    - time: 1 x timepoint (double)
    - values: timepoit x n_ch
    - channels（なくても良い）: table(index,id)

    Args:
        data: 保存するデータ（numpy配列）/ Data to save (numpy array)
        sf: サンプリングレート / Sampling frequency
        filename: matファイルのパス / path to mat file

    Returns:
        None
    """

    mat_dict = {
        'fs': sf,
        'time': np.arange(0, data.shape[1])/sf,
        'values': data.T,
        # 'channels': np.arange(data.shape[0]) if data.ndim > 1 else np.arange(1)
    }
    savemat(filename, mat_dict)