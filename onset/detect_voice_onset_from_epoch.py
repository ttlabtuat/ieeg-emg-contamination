import librosa
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def detect_voice_onset(y, sr, threshold_db=-40, offset=0.1, output='onset'):
    """
    音声データから音声開始点（オンセット）を検出する
    
    Parameters:
    y (np.array): 音声データのnumpy配列
    sr (int): サンプリング周波数
    threshold_db (float): 音声検出の閾値（dB）
    min_duration (float): 継続時間の最小値（秒）
    
    Returns:
    float: 音声開始点の時間（秒）
    """
    
    # 9600Hzに最適化したフレーム長を設定
    # 約20ms分のフレーム長（9600Hz * 0.02s = 192サンプル）
    frame_length = int(sr * 0.02)  # 192サンプル
    hop_length = frame_length // 4  # 48サンプル
    
    # RMS（Root Mean Square）エネルギーを計算
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    print(f"RMSエネルギーの計算: {len(rms)} フレーム")
    print(rms.shape)
    
    # dBに変換
    rms_db = librosa.amplitude_to_db(rms)
    
    # フレームを時間に変換するための配列
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # 閾値を超える最初のフレームを検出
    voice_frames = rms_db > threshold_db
    
    # 開始から0.1秒以下は無視
    min_frames = int(offset * sr / hop_length)  # 0.1秒分のフレーム数
    if len(voice_frames) > min_frames:
        voice_frames[:min_frames] = False
    
    if not np.any(voice_frames):
        print("音声が検出されませんでした")
        return 0.0
    
    # 最初の音声フレームのインデックス
    first_voice_frame = np.where(voice_frames)[0][0]    # voice_framesがTrueの最初のインデックスを取得
    last_voice_frame = np.where(voice_frames)[0][-1]  # 最後の音声フレームのインデックスを取得
    
    # 継続時間をチェック
    onset_time = times[first_voice_frame]
    offset_time = times[last_voice_frame]
    
    if output == 'onset':
        return onset_time
    elif output == 'step':
        return onset_time, offset_time

def visualize_onset(y, sr, onset_time, offset_time=None, threshold_db=-40):
    """
    音声波形とオンセット位置を可視化する
    """
    plt.figure(figsize=(12, 6))
    
    # 時間軸
    time = np.linspace(0, len(y) / sr, len(y))
    
    # 波形をプロット
    plt.subplot(2, 1, 1)
    plt.plot(time, y)
    plt.axvline(x=onset_time, color='red', linestyle='--', label=f'Onset: {onset_time:.3f}s')
    if offset_time is not None:
        plt.axvline(x=offset_time, color='orange', linestyle='--', label=f'Offset: {offset_time:.3f}s')
    plt.title('音声波形')
    plt.xlabel('時間 (秒)')
    plt.ylabel('振幅')
    plt.legend()
    plt.grid(True)
    
    # RMSエネルギーをプロット
    plt.subplot(2, 1, 2)
    frame_length = int(sr * 0.02)  # 20ms
    hop_length = frame_length // 4
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    plt.plot(rms_times, librosa.amplitude_to_db(rms))
    plt.axvline(x=onset_time, color='red', linestyle='--', label=f'Onset: {onset_time:.3f}s')
    if offset_time is not None:
        plt.axvline(x=offset_time, color='orange', linestyle='--', label=f'Offset: {offset_time:.3f}s')
    plt.axhline(y=threshold_db, color='green', linestyle=':', label=f'閾値: {threshold_db}dB')
    plt.title('RMSエネルギー (dB)')
    plt.xlabel('時間 (秒)')
    plt.ylabel('エネルギー (dB)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 使用例
if __name__ == "__main__":
    # # 音声ファイルのパス（9600Hzのファイル）
    # audio_pattern = "/Users/murakamishoya/git/tt-lab/tt_github/M2_work/20250620_check_same_file/komeiji_data/js05/*-voice-t3.wav"  # ここにファイルパスを入力
    # audio_file_list = sorted(glob(audio_pattern))
    # for audio_file in audio_file_list:
    #     print(f"処理中のファイル: {audio_file}")
    #     try:
    #         # 音声ファイルを読み込み
    #         y, sr = librosa.load(audio_file)
            
    #         # オンセットを検出
    #         onset_time = detect_voice_onset(y, sr, threshold_db=-35)
            
    #         print(f"音声開始時間: {onset_time:.3f} 秒")
    #         print(f"サンプリング周波数: {sr}Hz")
            
    #         # 可視化（オプション）
    #         visualize_onset(y, sr, onset_time)
            
    #     except FileNotFoundError:
    #         print("音声ファイルが見つかりません．ファイルパスを確認してください．")
    #     except Exception as e:
    #         print(f"エラーが発生しました: {e}")
    from cut_from_bids import create_epochs_from_bids_path
    from detect_voice_onset_from_epoch import detect_voice_onset, visualize_onset
    from mne_bids import BIDSPath, read_raw_bids

    path = BIDSPath(
        subject='js05',
        task='Speech8sen',
        run='5',
        datatype='ieeg',
        root='/Users/murakamishoya/git/tt-lab/tt_github/cut-juntendata-iBIDS/junten_bids'
    )

    event_names = ['overt']  # 抽出するイベント名
    tmin = 0
    tmax = 3.0
    epochs, path = create_epochs_from_bids_path(
        path,
        event_names=event_names,
        tmin=tmin,
        tmax=tmax,
    )
    voice_epochs = epochs.copy().pick(picks='voice')
    # df = voice_epochs.metadata.reset_index(drop=True)
    voice_df = voice_epochs.metadata.copy().reset_index(drop=True)
    for idx, voice in enumerate(voice_epochs):
        # メタデータからy_nをチェック（存在する場合）
        if voice_epochs.metadata is not None and 'y_n' in voice_epochs.metadata.columns:
            y_n_value = voice_epochs.metadata.iloc[idx]['y_n']
            if y_n_value == 'n':
                print(f"Skipping epoch {idx}: y_n = 'n'")
                continue
        
            print(f"Processing epoch {idx}: y_n = {y_n_value}")
            # 標準化
            voice = voice[0]
            voice = (voice - voice.mean()) / voice.std()
            onset_time, offset_time = detect_voice_onset(y=voice, sr=voice_epochs.info['sfreq'], threshold_db=-15, offset=0.2, output='step')
            print(f"Detected onset time: {onset_time:.3f} seconds")
            # visualize_onset(y=voice, sr=voice_epochs.info['sfreq'], onset_time=onset_time, offset_time=offset_time)
            voice_df.loc[idx, 'onset'] = voice_df.loc[idx, 'onset'] + onset_time
            voice_df.loc[idx, 'voice_offset_sec'] = voice_df.loc[idx, 'onset'] + offset_time
            voice_df.loc[idx, 'duration'] = offset_time - onset_time
            voice_df.loc[idx, 'sample'] = int(voice_df.loc[idx, 'onset'] * voice_epochs.info['sfreq'])
            
            # df.loc[idx, 'voice_onset_sample'] = int(onset_time * voice_epochs.info['sfreq'])
    
    print(voice_df)
    voice_df_path = path.update(root='./voice_onset_events', suffix='events', extension='.tsv')
    # voice_df.to_csv(f"voice_onset_evetns/{voice_df_path.basename}", sep='\t', index=False)

            # break