import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os
import glob
from tqdm import tqdm
import argparse
import soundfile as sf
import re

from experiment_config import SubjectConfig, PreprocessConfig, ExperimentConfig
from data_preparation.preprocess import Preprocess as pp
from model.svm_common import build_grid_serch, do_grid_serch
from test import test_plot

def get_labels_from_file(ecog_path: str) -> np.ndarray:
    """ECoGファイルからラベルを取得する関数
    例：sub-js01_run-1_ss-02_label-8_tmin-150_tmax-3636_tt-overt_ieeg.npy → 8
    """
    filename = os.path.basename(ecog_path)
    match = re.search(r'label-(\d+)', filename)
    if not match:
        raise ValueError(f"Could not find label in filename {filename}.")
    
    label = int(match.group(1))  # 正規表現で取得したラベルを整数に変換
    return label

def load_data(ecog_path: str):
    if not os.path.exists(ecog_path):
        raise FileNotFoundError(f"ECoG file not found: {ecog_path}")
    # ECoGデータの読み込み
    ecog = np.load(ecog_path)  # (time, ch) → (ch, time)
    return ecog

def preprocess_data(
        ecog: np.ndarray,
        subject_config: SubjectConfig, 
        preprocess_config: PreprocessConfig,
        ch: int) -> np.ndarray:
    
    """データの読み込みと前処理を行う関数"""
    ch_idx = ch - 1 # 0-indexedに変換
    if ch > ecog.shape[0]:
        raise ValueError(f"Channel {ch} is out of range. Available channels: 1-{ecog.shape[0]}")
    
    # 指定されたチャンネルのデータを抽出
    ecog = ecog[ch_idx]  # (ch, time) → (time, )
    # 1. リサンプリング
    ecog = pp.downsampling(ecog, original_sf=subject_config.original_sf, re_sf=preprocess_config.re_sf)
    
    # 2. ハイパスフィルター
    ecog_high = pp.highpass_filter(
        ecog,
        re_sf=preprocess_config.re_sf,
        cutoff=preprocess_config.highpass_cutoff,
        order=preprocess_config.highpass_order
    )
    
    # 3. ノッチフィルター
    ecog_notch = pp.notch_filter(
        ecog_high, 
        re_sf=preprocess_config.re_sf, 
        notch_freq=preprocess_config.notch_freq
    )
    
    # 4. 短時間フーリエ変換
    f, t, Zxx = pp.get_stft(
        ecog_notch,
        fs=preprocess_config.re_sf,
        clip_fs=preprocess_config.stft_clip_fs,
        nperseg=preprocess_config.win_len,
        noverlap=preprocess_config.win_step,
        return_onesided=True
    )
    
    # 5. パワー算出
    # Zxx = pp.get_power_spectrogram(Zxx)   # logを取って標準化するため不要
    
    # 6. logスケールに変換
    Zxx = pp.log_spectrogram(np.abs(Zxx))
    
    # 7. マスキング
    Zxx = pp.mask_spectrogram(Zxx, preprocess_config.mask_hz_list)
    
    # 8. 標準化（mask_hz_list以外の部分のみで）
    Zxx = pp.normalize_spectrogram(Zxx, normalizing=preprocess_config.normalizing, mask_hz_list=preprocess_config.mask_hz_list)

    # 9.再度マスキング（標準化でマスク領域が0でなくなってしまうため）
    Zxx = pp.mask_spectrogram(Zxx, preprocess_config.mask_hz_list)
    
    return f, t, Zxx

def run_experiment_each_ch(config: ExperimentConfig, 
                           subject_config: SubjectConfig,
                           ch: int):

    # 結果を保存するDataFrame
    results = []
    
    # ECoGファイルのリストを取得
    ecog_files = sorted(glob.glob(os.path.join(config.data_dir, f"{subject_config.name}/*tt-{config.task}_ieeg.npy"), recursive=True))
    
    if not ecog_files:
        raise FileNotFoundError(f"No ECoG files found in {os.path.join(config.data_dir, subject_config.name)}")
    
    # 全データの前処理を一度だけ行う
    all_features = []
    all_labels = []
    for file in tqdm(ecog_files, desc="Preprocessing data"):
        ecog = load_data(file)
        f, t, features = preprocess_data(ecog, subject_config, config.preprocess, ch)
        all_features.append(features.reshape(-1))  # 特徴量を1次元に変換
        all_labels.append(get_labels_from_file(file))
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    print(f"Loaded {all_features.shape} features. (= {all_features.shape[-1]/config.preprocess.stft_clip_fs} timepoints)")
    
    # 10回の繰り返し
    for repeat in tqdm(range(config.n_repeats), desc="Repeats"):
        # 5分割交差検証
        kf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42+repeat)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(all_features, all_labels)):
            # 学習データと評価データの分割
            X_train = all_features[train_idx]
            y_train = all_labels[train_idx]
            X_test = all_features[test_idx]
            y_test = all_labels[test_idx]
            test_plot(f, t, all_features[0])  # テスト用のスペクトログラムをプロット
            
            print(f"Repeat {repeat + 1}, Fold {fold}: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            # グリッドサーチの設定
            grid_search = build_grid_serch(config.svm_kernel)
            
            # 学習
            best_model = do_grid_serch(X_train, y_train, grid_search)
            
            # 予測
            y_pred = best_model.predict(X_test)
            
            # 正解率の計算
            accuracy = accuracy_score(y_test, y_pred)
            
            # 結果の保存
            results.append({
                'channel': ch,
                'repeat': repeat + 1,
                'fold': fold,
                'accuracy': accuracy,
                'y_test': y_test.tolist(),
                'y_pred': y_pred.tolist()
            })
            
    # 結果のDataFrameを作成
    results_df = pd.DataFrame(results)
    
    
    # 平均正解率の計算と表示
    mean_accuracy = results_df['accuracy'].mean()
    print(f"\nch{ch:02d}の平均正解率: {mean_accuracy:.4f}")
    
    return results_df

def svm_each_subject(config: ExperimentConfig, 
                     subject_config: SubjectConfig, 
                     experiment_dir: str):
    """各被験者ごとに実験を実行する関数"""
    # 被験者設定の読み込み
    subject_results_df = pd.DataFrame(columns=['channel', 'repeat', 'fold', 'accuracy', 'y_test', 'y_pred'])
    # acc_df = pd.DataFrame(columns=[f"ch{i:02d}" for i in subject_config.usable_ch])
    acc_df = pd.DataFrame()
    
    # 各電極の結果を保存するディレクトリを作成
    channel_results_dir = os.path.join(experiment_dir, "channel_results")
    os.makedirs(channel_results_dir, exist_ok=True)
    
    # 実験の実行
    all_ch_list = [i for i in range(1, subject_config.n_ch + 1)]
    # bad_ch_list = set(all_ch_list) - set(subject_config.usable_ch)
    for ch in all_ch_list:
        print(f"Running experiment for channel {ch}...")
        # 実験の実行
        results_df = run_experiment_each_ch(config, subject_config, ch)
        
        # 各電極の結果を個別に保存
        channel_result_path = os.path.join(channel_results_dir, f"{subject_config.name}_ch{ch:02d}_results.tsv")
        if os.path.exists(channel_result_path):
            raise FileExistsError(f"{channel_result_path} は既に存在します。上書き防止のため処理を中断します。")
        results_df.to_csv(channel_result_path, sep='\t', index=False)
        print(f"Channel {ch} results saved to {channel_result_path}")
        
        # 平均正解率の計算
        mean_accuracy = results_df['accuracy'].mean()
        print(f"Channel {ch} mean accuracy: {mean_accuracy:.4f}")
        
        # 結果を保存
        acc_df[f"ch{ch:02d}"] = results_df['accuracy']
        
        # 全体の結果に追加
        subject_results_df = pd.concat([subject_results_df, results_df], ignore_index=True)

    # 全ての電極の処理が完了したら、全体の結果を保存
    all_channels_path = os.path.join(experiment_dir, f"{subject_config.name}_all_channels_acc.tsv")
    if os.path.exists(all_channels_path):
        raise FileExistsError(f"{all_channels_path} は既に存在します。上書き防止のため処理を中断します。")
    subject_results_df.to_csv(all_channels_path, sep='\t', index=False)
    print(f"All results saved to {all_channels_path}")
    
    # 正解率の保存
    accuracy_path = os.path.join(experiment_dir, f"{subject_config.name}_accuracy.tsv")
    if os.path.exists(accuracy_path):
        raise FileExistsError(f"{accuracy_path} は既に存在します。上書き防止のため処理を中断します。")
    acc_df.to_csv(accuracy_path, sep='\t', index=False)
    print(f"Accuracy saved to {accuracy_path}")

# def run_experiment_wav(config: ExperimentConfig, 
#                        subject_config: SubjectConfig):

#     # 結果を保存するDataFrame
#     results = []
    
#     # ECoGファイルのリストを取得
#     wav_files = sorted(glob.glob(os.path.join(config.data_dir, f"{subject_config.name}/*voice-{config.task}.wav"), recursive=True))
    
#     if not wav_files:
#         raise FileNotFoundError(f"No wav files found in {os.path.join(config.data_dir, subject_config.name)}")
    
#     # 全データの前処理を一度だけ行う
#     all_features = []
#     all_labels = []
#     for file in tqdm(wav_files, desc="Preprocessing data"):
#         # wav = load_data(file)
#         # 音声ファイルの読み込み
#         wav, _ = sf.read(file)
#         wav = wav.reshape(1, -1)
#         f, t, features = preprocess_data(wav, subject_config, config.preprocess, ch=1)  # 1次元のデータなので擬似的に1
#         all_features.append(features.reshape(-1))  # 特徴量を1次元に変換
#         all_labels.append(get_labels_from_file(file))
    
#     all_features = np.array(all_features)
#     all_labels = np.array(all_labels)
#     print(f"Loaded {all_features.shape} features. (= {all_features.shape[-1]/config.preprocess.stft_clip_fs} timepoints)")
    
#     # 10回の繰り返し
#     for repeat in tqdm(range(config.n_repeats), desc="Repeats"):
#         # 5分割交差検証
#         kf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42+repeat)
        
#         for fold, (train_idx, test_idx) in enumerate(kf.split(all_features, all_labels)):
#             # 学習データと評価データの分割
#             X_train = all_features[train_idx]
#             y_train = all_labels[train_idx]
#             X_test = all_features[test_idx]
#             y_test = all_labels[test_idx]
#             # test_plot(f, t, all_features[0])  # テスト用のスペクトログラムをプロット
            
#             print(f"Repeat {repeat + 1}, Fold {fold}: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
#             # グリッドサーチの設定
#             grid_search = build_grid_serch(config.svm_kernel)
            
#             # 学習
#             best_model = do_grid_serch(X_train, y_train, grid_search)
            
#             # 予測
#             y_pred = best_model.predict(X_test)
            
#             # 正解率の計算
#             accuracy = accuracy_score(y_test, y_pred)
            
#             # 結果の保存
#             results.append({
#                 # 'channel': ch,
#                 'repeat': repeat + 1,
#                 'fold': fold,
#                 'accuracy': accuracy,
#                 'y_test': y_test.tolist(),
#                 'y_pred': y_pred.tolist()
#             })
            
#     # 結果のDataFrameを作成
#     results_df = pd.DataFrame(results)
    
    
#     # 平均正解率の計算と表示
#     mean_accuracy = results_df['accuracy'].mean()
#     print(f"\n平均正解率: {mean_accuracy:.4f}")
    
#     return results_df

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='SVM実験を実行するスクリプト')
    parser.add_argument('--js_yaml', type=str, required=True, help='被験者設定YAMLファイルのパス')
    parser.add_argument('--exp_yaml', type=str, required=False, help='実験設定YAMLファイルのパス')
    args = parser.parse_args()

    # 実験設定の作成
    if not args.exp_yaml:
        config = ExperimentConfig()
        exp_name = "default_experiment"
    else:
        config = ExperimentConfig.load_config(args.exp_yaml)
        exp_name = os.path.splitext(os.path.basename(args.exp_yaml))[0]
    
    experiment_dir = os.path.join(config.output_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 設定の保存
    config.save_config(os.path.join(experiment_dir, f"{exp_name}.yaml"))

    # 被験者設定の読み込み    
    subject_config = SubjectConfig.from_yaml(args.js_yaml)

    print(f"Running experiment for subject: {subject_config.name}")
    # 各被験者ごとに実験を実行
    results_df = svm_each_subject(config, subject_config, experiment_dir=experiment_dir)
    # results_df.to_csv(f"{experiment_dir}/{subject_config.name}_results.tsv", sep='\t', index=False)




