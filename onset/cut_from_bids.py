#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd
import mne
import mne_bids
from mne_bids import BIDSPath, read_raw_bids, get_bids_path_from_fname
import argparse
from pathlib import Path
from typing import List, Optional
import soundfile as sf

def create_epochs_from_bids_path(bids_path: str, event_names: List[str], 
                                tmin: float = -0.15, tmax: float = 3.0,
                                baseline: Optional[tuple] = None) -> mne.Epochs:
    """
    BIDS形式のデータからエポックを作成し、メタデータを設定する
    Create epochs from BIDS format data and set metadata
    
    Args:
        bids_path (str): BIDSPathオブジェクトまたはEDFファイルのパス / BIDSPath object or path to EDF file
        event_names (List[str]): 抽出したいイベント名のリスト / List of event names to extract
        tmin (float): イベント前の開始時間（秒）、デフォルト: -0.15 / Start time before event (seconds), default: -0.15
        tmax (float): イベント後の終了時間（秒）、デフォルト: 3.0 / End time after event (seconds), default: 3.0
        baseline (Optional[tuple]): ベースライン期間 (tmin, tmax)、デフォルト: None / Baseline period (tmin, tmax), default: None
    
    Returns:
        mne.Epochs: メタデータ付きのエポック / Epochs with metadata
    """
    # 1. BIDSパスを解析してBIDSPathオブジェクトを作成
    if isinstance(bids_path, str):
        path = get_bids_path_from_fname(bids_path)
    else:
        path = bids_path
    
    # 2. mne_bidsを使用してRawデータを読み込み
    raw = read_raw_bids(path, extra_params={"infer_types": True})
    
    # 3. アノテーションからイベントとevent_idを取得
    events, event_id = mne.events_from_annotations(raw, event_id=None)
    
    # 4. BIDSPathを使用してevents.tsvからメタデータを取得
    try:
        events_path = path.copy().update(suffix='events', extension='.tsv')
        metadata_df = pd.read_csv(events_path.fpath, sep='\t', dtype_backend='numpy_nullable')
    except FileNotFoundError:
        print(f"Warning: events.tsv not found for {path}")
        metadata_df = None
    
    # 5. エポックを作成
    epochs = mne.Epochs(raw, events, event_id=event_id, 
                       tmin=tmin, tmax=tmax, baseline=baseline,
                       preload=True, metadata=metadata_df, )
    epochs.crop(tmin, tmax, include_tmax=False)  # エポックの時間範囲を設定
    
    # 6. 指定されたイベントのエポックのみを選択
    if event_names and epochs.metadata is not None:
        # trial_type列が存在する場合の選択
        if 'trial_type' in epochs.metadata.columns:
            mask = epochs.metadata['trial_type'].isin(event_names)
            epochs = epochs[mask]
            # エポックのインデックスに合わせてメタデータのインデックスをリセット
            epochs.metadata = epochs.metadata.reset_index(drop=True)
    
    return epochs, path

def save_speech_epochs_npy(epochs: mne.Epochs, save_dir: str, 
                      tmin: float = -0.15, tmax: float = 3.0, 
                      bids_path: BIDSPath = None):
    """
    エポックを個別の.npyファイルとして保存し、音声データをWAVファイルとしても保存
    
    Args:
        epochs (mne.Epochs): 保存するエポック
        save_dir (str): 保存ディレクトリ
        tmin (float): エポック開始時間（ファイル名用）
        tmax (float): エポック終了時間（ファイル名用）
        bids_path (BIDSPath): ファイル名生成用のBIDSPathオブジェクト
    """
    # 保存ディレクトリを作成
    os.makedirs(save_dir, exist_ok=True)
    
    # BIDSPathからファイル名コンポーネントを取得
    base_filename = f"sub-{bids_path.subject}_run-{bids_path.run}"
    
    # tminとtmaxをms単位に変換（負の値には'neg'プレフィックスを使用）
    if tmin < 0:
        tmin_ms = f"neg{int(abs(tmin) * 1000)}"
    else:
        tmin_ms = f"{int(tmin * 1000)}"
    
    if tmax < 0:
        tmax_ms = f"neg{int(abs(tmax) * 1000)}"
    else:
        tmax_ms = f"{int(tmax * 1000)}"
    
    saved_count = 0
    
    for idx, epoch in enumerate(epochs):
        # メタデータからy_nをチェック（存在する場合）
        if epochs.metadata is not None and 'y_n' in epochs.metadata.columns:
            y_n_value = epochs.metadata.iloc[idx]['y_n']
            if y_n_value == 'n':
                print(f"Skipping epoch {idx}: y_n = 'n'")
                continue
        
        # 神経データを抽出（sEEGまたはECoG、STIMチャンネルを除外）
        ieeg_data = epoch.copy()
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
        
        eog_picks = mne.pick_types(epochs.info, eog=True, stim=False)

        neural_only_data = ieeg_data[neural_picks]
        eog_picks = ['right', 'r_up']
        wav_picks = ['voice']
        
        eog_idx = np.where(np.isin(epochs.ch_names, eog_picks))[0]
        eog = ieeg_data[eog_idx, :]
        wav_idx = np.where(np.isin(epochs.ch_names, wav_picks))[0]
        voice = ieeg_data[wav_idx, :]
        # ファイル名を作成
        filename_parts = []
        
        # ベースファイル名を使用
        filename_parts.append(base_filename)
        # メタデータから追加情報を取得
        if epochs.metadata is not None:
            row = epochs.metadata.iloc[idx]
            
            # メタデータからセッション番号
            filename_parts.append(f'ss-{int(row["num"]):02d}')
        
            filename_parts.append(f'label-{row["label"]}')
            
            # 時間窓情報
            filename_parts.append(f'tmin-{tmin_ms}')
            filename_parts.append(f'tmax-{tmax_ms}')
            
            filename_parts.append(f'tt-{row["trial_type"]}')
        
        # filename_parts.append('ieeg.npy')
        filename = '_'.join(filename_parts)
        
        # 保存
        save_path = os.path.join(save_dir, filename)
        np.save(f"{save_path}_ieeg.npy", neural_only_data)
        np.save(f"{save_path}_eog.npy", eog)
        np.save(f"{save_path}_voice.npy", voice)
        
        # 音声データをWAVファイルとして保存（soundfileを使用）
        voice_wav = voice.flatten()  # 1次元配列に変換
        # 正規化
        voice_wav = voice_wav / np.max(np.abs(voice_wav))
        wav_path = f"{save_path}_voice.wav"
        sf.write(wav_path, voice_wav, int(epochs.info['sfreq']))
        
        print(f"Saved: {save_path} (.npy files and .wav)")
        saved_count += 1
    
    print(f"Total saved files: {saved_count}")

def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='Cut epochs from BIDS EDF files and save as .npy')
    parser.add_argument('--edf_path', type=str, required=True, help='Path to EDF file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save .npy files')
    parser.add_argument('--event_names', nargs='+', default=['overt'], 
                      help='Event names to extract (default: overt)')
    parser.add_argument('--tmin', type=float, default=-0.15, 
                      help='Start time before event (in seconds, default: -0.15)')
    parser.add_argument('--tmax', type=float, default=3.0, 
                      help='End time after event (in seconds, default: 3.0)')
    parser.add_argument('--baseline', nargs=2, type=float, metavar=('START', 'END'),
                      help='Baseline period (start, end) in seconds. Example: --baseline -0.15 0')
    return parser.parse_args()

def main():
    """メイン関数"""
    DEBUG = False  # デバッグモード
    
    if DEBUG:
        # デバッグ用パス設定
        edf_path = 'junten_bids/sub-js01/ieeg/sub-js01_task-Speech8sen_run-1_ieeg.edf'
        save_dir = 'ouput_epochs'
        event_names = ['overt']  # 抽出するイベント名
        tmin = -0.1
        tmax = 2.9
        baseline = None
    else:
        args = parse_arguments()
        edf_path = args.edf_path
        save_dir = args.save_dir
        event_names = args.event_names
        tmin = args.tmin
        tmax = args.tmax
        baseline = tuple(args.baseline) if args.baseline else None
    
    print(f"Processing EDF file: {edf_path}")
    print(f"Save directory: {save_dir}")
    print(f"Event names to extract: {event_names}")
    print(f"Time window: {tmin}s to {tmax}s")
    print(f"Baseline: {baseline}")
    
    # 1. エポックを作成
    print("エポックを作成中...")
    epochs, bids_path = create_epochs_from_bids_path(edf_path, event_names, tmin, tmax, baseline)
    print(f"Created {len(epochs)} epochs")
    
    # 2. .npyファイルとして保存
    print(".npyファイルとしてエポックを保存中...")
    save_speech_epochs_npy(epochs, save_dir, tmin, tmax, bids_path)
    
    print("処理完了！")

if __name__ == '__main__':
    main()