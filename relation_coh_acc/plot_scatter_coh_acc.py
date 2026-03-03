import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
import re
import yaml


def get_eog_colors():
    """HEOG/VEOGの色を取得する（カラーコード固定）
    
    - HEOG: #1b9e77（青緑）
    - VEOG: #d95f02（オレンジ）
    """
    heog_color = '#1b9e77'
    veog_color = '#d95f02'
    return heog_color, veog_color


def get_significance_marker(p_value):
    """p値から有意性マーカーを返す関数
    
    引数:
        p_value: p値
    戻り値:
        str: 有意性マーカー（'***', '**', '*', ''のいずれか）
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


def apply_fdr_correction(p_values, names):
    """Benjamini–Hochberg (FDR) による多重比較補正を適用して辞書で返す"""
    if len(p_values) == 0:
        return {}
    _, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    return {names[j]: float(p_corrected[j]) for j in range(len(names))}


def apply_bonferroni_correction(p_values, names, n_tests=None):
    """Bonferroni補正を適用して辞書で返す
    
    n_tests を指定した場合，補正回数を固定して p_corrected = min(p * n_tests, 1) とする
    """
    if len(p_values) == 0:
        return {}
    m = int(n_tests) if n_tests is not None else len(p_values)
    p_corrected = np.minimum(np.asarray(p_values, dtype=float) * m, 1.0)
    return {names[j]: float(p_corrected[j]) for j in range(len(names))}

def plot_individual_participants(ieeg_acc_path, coh_path, save_path, participant):
    """各参加者について個別の散布図を作成する関数（入出力を他関数と統一）

    引数:
        ieeg_acc_path: iEEG の正解率 CSV パス（列=参加者，行=チャネル）
        coh_path: コヒーレンス CSV パス（列=参加者，行=チャネル）
        save_path: 保存先（ファイルパス もしくは ディレクトリ）
    戻り値:
        なし（図ファイルを save_path に保存）
    """
    acc_df = pd.read_csv(ieeg_acc_path, index_col=0)
    coh_df = pd.read_csv(coh_path, index_col=0)

    acc_data = acc_df[participant].dropna()
    coh_data = coh_df[participant].dropna()

    # 共通チャネルのみを使用
    common_channels = acc_data.index.intersection(coh_data.index)

    acc_common = acc_data[common_channels]
    coh_common = coh_data[common_channels]

    # 図作成（色・サイズ・αを統一）
    heog_color, _ = get_eog_colors()
    plt.figure(figsize=(10, 8))
    plt.scatter(coh_common, acc_common, alpha=0.7, s=70, color=heog_color)

    # 相関（データ数が2以上のときのみ）
    has_enough_points = len(coh_common) > 1
    if has_enough_points:
        r_value, p_value = stats.pearsonr(coh_common, acc_common)
        if p_value < 0.001:
            p_text = '***'
        elif p_value < 0.01:
            p_text = '**'
        elif p_value < 0.05:
            p_text = '*'
        else:
            p_text = ''
        title_text = f'{participant}\nr = {r_value:.3f} {p_text}'
    else:
        title_text = f'{participant}\nNo data'

    # 体裁（軸ラベル文言・フォントサイズ統一）
    plt.xlabel('Coherence', fontsize=16)
    plt.ylabel('Classification Accuracy', fontsize=16)
    plt.title(title_text, fontsize=18)
    plt.grid(True, alpha=0.3)

    # 軸範囲（ylim固定，xlimはデータに基づき少しパディング）
    plt.ylim(0, 1)
    if len(coh_common) > 0:
        x_min = float(np.min(coh_common))
        x_max = float(np.max(coh_common))
        if x_min == x_max:
            pad = 0.01 if x_min == 0 else abs(x_min) * 0.05
            x_min -= pad
            x_max += pad
        else:
            pad = (x_max - x_min) * 0.05
            x_min -= pad
            x_max += pad
        plt.xlim(x_min, x_max)

    # 電極名ラベル
    for i, channel in enumerate(common_channels):
        plt.annotate(channel, (coh_common.iloc[i], acc_common.iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=14, alpha=0.7)

    plt.tight_layout()

    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 統計情報
    print(f"{participant}参加者の統計情報:")
    print(f"電極数: {len(common_channels)}")
    # print(f"相関係数: {correlation:.3f}")
    print(f"Accuracy - 平均: {acc_common.mean():.3f}，標準偏差: {acc_common.std():.3f}")
    print(f"Coherence - 平均: {coh_common.mean():.3f}，標準偏差: {coh_common.std():.3f}")
    print("-" * 50)


# print("全参加者の散布図が保存されました。")

# 全参加者を1枚の図に収める関数
def plot_all_participants_combined(ieeg_acc_path, coh_eog_h_path, coh_eog_v_path, eog_acc_path, save_path, change_color_by_bad_ch=False, yaml_dir='js_yamls', js_list=None, layout=None):
    plt.rcParams["font.size"] = 14
    acc_df = pd.read_csv(ieeg_acc_path, index_col=0)
    coh_eog_h_df = pd.read_csv(coh_eog_h_path, index_col=0)
    coh_eog_v_df = pd.read_csv(coh_eog_v_path, index_col=0)
    if eog_acc_path.endswith('.tsv'):
        eog_acc_df = pd.read_csv(eog_acc_path, index_col=0, sep='\t')
    else:
        eog_acc_df = pd.read_csv(eog_acc_path, index_col=0)

    # 全参加者のリストを取得
    all_participants = acc_df.columns.tolist()
    
    # js_listが指定されている場合，その参加者のみをフィルタリング
    if js_list is not None:
        participants = [p for p in all_participants if p in js_list]
    else:
        participants = all_participants

    # サブプロットのレイアウトを決定
    if layout is not None:
        row, col = layout
    else:
        raise ValueError("layout must be specified")

    # 各サブプロットを正方形にするため，サイズを可変に設定
    # 各サブプロットを4x4インチに設定
    subplot_size = 4
    figsize = (col * subplot_size, row * subplot_size)
    fig, axes = plt.subplots(row, col, figsize=figsize)
    axes = axes.flatten()  # 1次元配列に変換
    
    # 全データのaccuracy範囲とcoherence範囲を計算（軸統一のため）
    all_acc_min = float('inf')
    all_acc_max = float('-inf')
    all_coh_min = float('inf')
    all_coh_max = float('-inf')
    
    # 図の縦軸範囲と横軸範囲を決めるために、全参加者のデータの最大値と最小値を計算
    for participant in participants:
        acc_data = acc_df[participant].dropna()
        coh_eog_h_data = coh_eog_h_df[participant].dropna()
        coh_eog_v_data = coh_eog_v_df[participant].dropna()
        common_channels = acc_data.index.intersection(coh_eog_h_data.index)

        # 図の縦軸範囲と横軸範囲を統一するために、各データの範囲を計算
        if len(common_channels) > 1:
            acc_common = acc_data[common_channels]
            coh_eog_h_common = coh_eog_h_data[common_channels]
            coh_eog_v_common = coh_eog_v_data[common_channels]
            
            # Accuracyの範囲を更新
            all_acc_min = min(all_acc_min, acc_common.min())
            all_acc_max = max(all_acc_max, acc_common.max())
            
            # Coherenceの範囲を更新（水平と垂直の両方）
            all_coh_min = min(all_coh_min, coh_eog_h_common.min(), coh_eog_v_common.min())
            all_coh_max = max(all_coh_max, coh_eog_h_common.max(), coh_eog_v_common.max())
    
    # 各参加者についてサブプロットを作成
    heog_color, veog_color = get_eog_colors()
    for i, participant in enumerate(participants):
        if i < len(axes):  # サブプロット数以内の場合のみ処理
            # 参加者のデータを取得
            acc_data = acc_df[participant].dropna()
            coh_eog_h_data = coh_eog_h_df[participant].dropna()
            coh_eog_v_data = coh_eog_v_df[participant].dropna()
            
            # 共通の電極のみを使用
            common_channels = acc_data.index.intersection(coh_eog_h_data.index)
            acc_common = acc_data[common_channels]
            coh_eog_h_common = coh_eog_h_data[common_channels]
            coh_eog_v_common = coh_eog_v_data[common_channels]
            
            if len(common_channels) > 1:
                # bad_chの処理
                if change_color_by_bad_ch:
                    bad_ch = get_bad_ch(f"{yaml_dir}/{participant}.yaml")
                else:
                    bad_ch = []
                
                # bad_chに含まれるチャネルとそれ以外を分ける
                bad_channels = [ch for ch in common_channels if ch in bad_ch]
                good_channels = [ch for ch in common_channels if ch not in bad_ch]
                
                # 通常のチャネルをプロット（水平EOGと垂直EOGを色分け）
                if len(good_channels) > 0:
                    acc_good = acc_common[good_channels]
                    coh_eog_h_good = coh_eog_h_common[good_channels]
                    coh_eog_v_good = coh_eog_v_common[good_channels]
                    # 最初のサブプロットのみlabelを設定（凡例用）
                    if i == 0:
                        axes[i].scatter(coh_eog_h_good, acc_good, alpha=0.7, s=30, color=heog_color, label='HEOG Coherence vs Accuracy')
                        axes[i].scatter(coh_eog_v_good, acc_good, alpha=0.7, s=30, color=veog_color, label='VEOG Coherence vs Accuracy')
                    else:
                        axes[i].scatter(coh_eog_h_good, acc_good, alpha=0.7, s=30, color=heog_color)
                        axes[i].scatter(coh_eog_v_good, acc_good, alpha=0.7, s=30, color=veog_color)
                
                # bad_chに含まれるチャネルを灰色でプロット
                if len(bad_channels) > 0:
                    acc_bad = acc_common[bad_channels]
                    coh_eog_h_bad = coh_eog_h_common[bad_channels]
                    coh_eog_v_bad = coh_eog_v_common[bad_channels]
                    # 最初のサブプロットのみlabelを設定（凡例用）
                    if i == 0:
                        axes[i].scatter(coh_eog_h_bad, acc_bad, alpha=0.7, s=30, color='gray', label='Bad channels')
                    else:
                        axes[i].scatter(coh_eog_h_bad, acc_bad, alpha=0.7, s=30, color='gray')
                        axes[i].scatter(coh_eog_v_bad, acc_bad, alpha=0.7, s=30, color='gray')
                
                # 回帰直線を計算（全チャネルを使用）
                slope_h, intercept_h, r_value_h, p_value_h, std_err_h = stats.linregress(coh_eog_h_common, acc_common)
                slope_v, intercept_v, r_value_v, p_value_v, std_err_v = stats.linregress(coh_eog_v_common, acc_common)
                
                # 回帰直線のプロット
                # x_line_h = np.linspace(coh_eog_h_common.min(), coh_eog_h_common.max(), 100)
                # y_line_h = slope_h * x_line_h + intercept_h
                # axes[i].plot(x_line_h, y_line_h, 'r-', linewidth=1.5, alpha=0.8, label=f'HEOG line (R²={r_value_h**2:.3f})')
                
                # x_line_v = np.linspace(coh_eog_v_common.min(), coh_eog_v_common.max(), 100)
                # y_line_v = slope_v * x_line_v + intercept_v
                # axes[i].plot(x_line_v, y_line_v, 'b-', linewidth=1.5, alpha=0.8, label=f'VEOG line (R²={r_value_v**2:.3f})')
                
                # EOG正解率の横線を追加
                if participant in eog_acc_df.columns:
                    eog_h_acc = eog_acc_df.loc['overt_eog_h', participant]
                    eog_v_acc = eog_acc_df.loc['overt_eog_v', participant]
                    
                    # 水平EOGの横線（濃い緑色，散布図と統一）
                    # 最初のサブプロットのみlabelを設定（凡例用）
                    if i == 0:
                        axes[i].axhline(y=eog_h_acc, color=heog_color, linestyle='--', linewidth=1.5, alpha=0.7, 
                                       label='HEOG Accuracy')
                        axes[i].axhline(y=eog_v_acc, color=veog_color, linestyle='--', linewidth=1.5, alpha=0.7, 
                                       label='VEOG Accuracy')
                    else:
                        axes[i].axhline(y=eog_h_acc, color=heog_color, linestyle='--', linewidth=1.5, alpha=0.7)
                        axes[i].axhline(y=eog_v_acc, color=veog_color, linestyle='--', linewidth=1.5, alpha=0.7)
                
                # サブプロットの設定
                if i % col == 0:  # 左端の列のみ縦軸ラベルを表示
                    axes[i].set_ylabel('Classification Accuracy', fontsize=16)
                else:
                    axes[i].set_ylabel('')
                
                if i >= row * (col - 1):  # 下段のみ横軸ラベルを表示
                    axes[i].set_xlabel('Coherence', fontsize=16)
                else:
                    axes[i].set_xlabel('')
                
                axes[i].set_title(f'{participant}\nHEOG Corr: {r_value_h:.3f}\nVEOG Corr: {r_value_v:.3f}', fontsize=18)
                axes[i].grid(True, alpha=0.3)
                
                # 縦軸と横軸の範囲を統一
                axes[i].set_ylim(0, 1)
                axes[i].set_xlim(all_coh_min - 0.01, all_coh_max + 0.01)
                
                # # 電極名をラベルとして表示（簡略化）
                # for j, channel in enumerate(common_channels):
                #     if j < 5:  # 最初の5つの電極のみラベル表示
                #         axes[i].annotate(channel, (coh_common.iloc[j], acc_common.iloc[j]), 
                #                        xytext=(3, 3), textcoords='offset points', fontsize=6, alpha=0.6)
            else:
                axes[i].text(0.5, 0.5, f'{participant}\nNo data', ha='center', va='center', 
                           transform=axes[i].transAxes)
                axes[i].set_title(participant, fontsize=14)
                # データがない場合も軸範囲を統一
                axes[i].set_ylim(0, 1)
                axes[i].set_xlim(all_coh_min - 0.01, all_coh_max + 0.01)
    
    # 使用されていないサブプロットを非表示
    for i in range(len(participants), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 全体の凡例を図の外に配置
    fig.legend(loc='upper right', bbox_to_anchor=(1.02, 1.0), fontsize=12, frameon=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"全参加者を1枚にまとめた図が保存されました:{save_path}")
    print(f"統一された縦軸範囲（Accuracy）: {all_acc_min:.3f} - {all_acc_max:.3f}")
    print(f"統一された横軸範囲（Coherence）: {all_coh_min:.3f} - {all_coh_max:.3f}")


# 全参加者を1枚の図に収める関数
def plot_all_participants_hveog_combined(
    ieeg_acc_path,
    coh_eog_h_path,
    eog_acc_path,
    eog_name,
    save_path,
    change_color_by_bad_ch=False,
    yaml_dir='js_yamls',
    js_list=None,
    layout=None,
    correction_type='fdr',
    n_tests=None,
):
    plt.rcParams["font.size"] = 14
    acc_df = pd.read_csv(ieeg_acc_path, index_col=0)
    coh_eog_h_df = pd.read_csv(coh_eog_h_path, index_col=0)
    if eog_acc_path.endswith('.tsv'):
        eog_acc_df = pd.read_csv(eog_acc_path, index_col=0, sep='\t')
    else:
        eog_acc_df = pd.read_csv(eog_acc_path, index_col=0)

    # 全参加者のリストを取得
    all_participants = acc_df.columns.tolist()
    
    # js_listが指定されている場合，その参加者のみをフィルタリング
    if js_list is not None:
        participants = [p for p in all_participants if p in js_list]
    else:
        participants = all_participants

    # サブプロットのレイアウトを決定
    if layout is not None:
        row, col = layout
    else:
        raise ValueError("layout must be specified")

    # 各サブプロットを正方形にするため，サイズを可変に設定
    # 各サブプロットを4x4インチに設定
    subplot_size = 4
    figsize = (col * subplot_size, row * subplot_size)
    fig, axes = plt.subplots(row, col, figsize=figsize)
    axes = axes.flatten()  # 1次元配列に変換
    
    # 全データのaccuracy範囲とcoherence範囲を計算（軸統一のため）
    all_acc_min = float('inf')
    all_acc_max = float('-inf')
    all_coh_min = float('inf')
    all_coh_max = float('-inf')
    
    # 図の縦軸範囲と横軸範囲を決めるために、全参加者のデータの最大値と最小値を計算
    for participant in participants:
        acc_data = acc_df[participant].dropna()
        coh_eog_h_data = coh_eog_h_df[participant].dropna()
        common_channels = acc_data.index.intersection(coh_eog_h_data.index)

        # 図の縦軸範囲と横軸範囲を統一するために、各データの範囲を計算
        if len(common_channels) > 1:
            acc_common = acc_data[common_channels]
            coh_eog_h_common = coh_eog_h_data[common_channels]
            
            # Accuracyの範囲を更新
            all_acc_min = min(all_acc_min, acc_common.min())
            all_acc_max = max(all_acc_max, acc_common.max())
            
            # Coherenceの範囲を更新
            all_coh_min = min(all_coh_min, coh_eog_h_common.min())
            all_coh_max = max(all_coh_max, coh_eog_h_common.max())
    
    # ===== 多重比較補正のため，まずp値を収集（2パス） =====
    per_participant_stats = {}  # participant -> dict
    p_values = []
    p_names = []
    for participant in participants:
        acc_data = acc_df[participant].dropna()
        coh_eog_h_data = coh_eog_h_df[participant].dropna()
        common_channels = acc_data.index.intersection(coh_eog_h_data.index)
        if len(common_channels) <= 1:
            continue
        if change_color_by_bad_ch:
            bad_ch = get_bad_ch(f"{yaml_dir}/{participant}.yaml")
        else:
            bad_ch = []
        good_channels = [ch for ch in common_channels if ch not in bad_ch]
        if len(good_channels) <= 1:
            continue
        acc_common = acc_data[common_channels]
        coh_common = coh_eog_h_data[common_channels]
        r_value_h, p_value_h = stats.pearsonr(coh_common[good_channels], acc_common[good_channels])
        per_participant_stats[participant] = {
            'r_value_h': float(r_value_h),
            'p_raw_h': float(p_value_h),
            'good_channels': good_channels,
            'common_channels': common_channels,
        }
        p_values.append(float(p_value_h))
        p_names.append(participant)
    
    # 補正後p値を作成
    if correction_type == 'fdr':
        corrected_p_dict = apply_fdr_correction(p_values, p_names)
    elif correction_type == 'bonferroni':
        corrected_p_dict = apply_bonferroni_correction(p_values, p_names, n_tests=n_tests)
    else:
        corrected_p_dict = {p_names[j]: p_values[j] for j in range(len(p_names))}
    
    # ===== 第2パス: 各参加者についてサブプロットを作成（補正後p値でマーカー決定） =====
    if eog_name == 'right':
        color, _ = get_eog_colors()
    elif eog_name == 'r_up':
        _, color = get_eog_colors()
    else:
        raise ValueError(f"Invalid eog_name: {eog_name}")
    
    r_value_h_dict = {}
    p_value_h_dict = {}
    p_value_h_corrected_dict = {}
    sig_h_dict = {}
    
    for i, participant in enumerate(participants):
        if i >= len(axes):
            continue
        acc_data = acc_df[participant].dropna()
        coh_eog_h_data = coh_eog_h_df[participant].dropna()
        common_channels = acc_data.index.intersection(coh_eog_h_data.index)
        acc_common = acc_data[common_channels]
        coh_common = coh_eog_h_data[common_channels]
        
        if participant in per_participant_stats:
            good_channels = per_participant_stats[participant]['good_channels']
            if len(good_channels) > 0:
                axes[i].scatter(coh_common[good_channels], acc_common[good_channels], alpha=0.7, s=30, color=color)
            
            r_value_h = per_participant_stats[participant]['r_value_h']
            p_value_h = per_participant_stats[participant]['p_raw_h']
            p_value_used = corrected_p_dict.get(participant, p_value_h)
            p_text = get_significance_marker(p_value_used)
            
            r_value_h_dict[participant] = r_value_h
            p_value_h_dict[participant] = p_value_h
            p_value_h_corrected_dict[participant] = float(p_value_used)
            sig_h_dict[participant] = p_text
            
            # EOG正解率の横線を追加
            if participant in eog_acc_df.columns:
                if eog_name == 'r_up':
                    eog_h_acc = eog_acc_df.loc['overt_eog_v', participant]
                    axes[i].axhline(y=eog_h_acc, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
                elif eog_name == 'right':
                    eog_h_acc = eog_acc_df.loc['overt_eog_h', participant]
                    axes[i].axhline(y=eog_h_acc, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
                else:
                    raise ValueError(f"Invalid eog_name: {eog_name}")
            
            # サブプロットの設定
            if i % col == 0:  # 左端の列のみ縦軸ラベルを表示
                axes[i].set_ylabel('Classification Accuracy', fontsize=16)
            else:
                axes[i].set_ylabel('')
            
            if i >= row * (col - 1):  # 下段のみ横軸ラベルを表示
                axes[i].set_xlabel('Coherence', fontsize=16)
            else:
                axes[i].set_xlabel('')
            
            axes[i].set_title(f'{participant}\nr = {r_value_h:.3f} {p_text}', fontsize=18)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)
            axes[i].set_xlim(all_coh_min - 0.01, all_coh_max + 0.01)
        else:
            axes[i].text(0.5, 0.5, f'{participant}\nNo data', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(participant, fontsize=14)
            axes[i].set_ylim(all_acc_min - 0.05, all_acc_max + 0.05)
            axes[i].set_xlim(all_coh_min - 0.01, all_coh_max + 0.01)
    
    # 使用されていないサブプロットを非表示
    for i in range(len(participants), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"全参加者を1枚にまとめた図が保存されました:{save_path}")
    print(f"統一された縦軸範囲（Accuracy）: {all_acc_min:.3f} - {all_acc_max:.3f}")
    print(f"統一された横軸範囲（Coherence）: {all_coh_min:.3f} - {all_coh_max:.3f}")
    
    # 多重比較補正の確認用出力
    print(f"\n=== 多重比較補正の結果（correction_type={correction_type}，検定数={len(p_values)}） ===")
    for name, p_raw, p_corr in zip(p_names, p_values, [corrected_p_dict[n] for n in p_names]):
        sig = get_significance_marker(p_corr)
        print(f"  {name}: p_raw={p_raw:.4f} -> p_{correction_type}={p_corr:.4f} {sig}")
    print("=" * 60)

    r_df = pd.Series(r_value_h_dict, name='r_value_h')
    p_raw_df = pd.Series(p_value_h_dict, name='p_raw_h')
    p_corr_col = f"p_{correction_type}_h" if correction_type != 'raw' else "p_raw_used_h"
    sig_col = f"sig_{correction_type}_h" if correction_type != 'raw' else "sig_raw_used_h"
    p_corr_df = pd.Series(p_value_h_corrected_dict, name=p_corr_col)
    sig_df = pd.Series(sig_h_dict, name=sig_col)
    concat_df = pd.concat([r_df, p_raw_df, p_corr_df, sig_df], axis=1)
    return concat_df

def get_bad_ch(js_yaml):
    with open(js_yaml, 'r') as f:
        js_data = yaml.load(f, Loader=yaml.SafeLoader)
    usable_ch = js_data['usable_ch']
    bad_ch = set(range(1, js_data['total_num_ch'] + 1)) - set(usable_ch)    
    
    bad_ch = [f"ch{i:02d}" for i in bad_ch]

    return sorted(list(bad_ch))

# # 全参加者のデータを1つの散布図に重ねる関数
# def plot_all_participants_overlaid():
#     plt.figure(figsize=(12, 10))
    
#     # 色のパレットを準備
#     colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))
    
#     # 各参加者のデータを重ねてプロット
#     for i, participant in enumerate(participants):
#         # 参加者のデータを取得
#         acc_data = acc_df[participant].dropna()
#         coh_data = coh_df[participant].dropna()
        
#         # 共通の電極のみを使用
#         common_channels = acc_data.index.intersection(coh_data.index)
#         acc_common = acc_data[common_channels]
#         coh_common = coh_data[common_channels]
        
#         if len(common_channels) > 1:
#             # 散布図を作成（色分け）
#             plt.scatter(coh_common, acc_common, alpha=0.6, s=40, 
#                        c=[colors[i]], label=f'{participant} (n={len(common_channels)})')
            
#             # 回帰直線を計算
#             slope, intercept, r_value, p_value, std_err = stats.linregress(coh_common, acc_common)
            
#             # 回帰直線のプロット（薄い線）
#             x_line = np.linspace(coh_common.min(), coh_common.max(), 100)
#             y_line = slope * x_line + intercept
#             plt.plot(x_line, y_line, color=colors[i], alpha=0.3, linewidth=1)
    
#     # 全体の回帰直線を計算（全データを結合）
#     all_acc = []
#     all_coh = []
#     for participant in participants:
#         acc_data = acc_df[participant].dropna()
#         coh_data = coh_df[participant].dropna()
#         common_channels = acc_data.index.intersection(coh_data.index)
#         all_acc.extend(acc_data[common_channels].values)
#         all_coh.extend(coh_data[common_channels].values)
    
#     # 全体の回帰直線を計算（全データを結合）
#     all_slope, all_intercept, all_r_value, all_p_value, all_std_err = stats.linregress(all_coh, all_acc)
    
#     # 全体の回帰直線（太い黒線）
#     x_all = np.linspace(min(all_coh), max(all_coh), 100)
#     y_all = all_slope * x_all + all_intercept
#     plt.plot(x_all, y_all, 'k-', linewidth=3, alpha=0.8, 
#              label=f'Overall (R² = {all_r_value**2:.3f})')
    
#     # グラフの設定
#     plt.xlabel('Coherence', fontsize=14)
#     plt.ylabel('Accuracy', fontsize=14)
#     plt.title(f'All Participants: Accuracy vs Coherence\nOverall Correlation: {all_r_value:.3f}, p-value: {all_p_value:.3f}', 
#               fontsize=16)
#     plt.grid(True, alpha=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
#     plt.tight_layout()
#     plt.savefig('scatter_all_participants_overlaid.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print("全参加者を重ねた散布図が保存されました: scatter_all_participants_overlaid.png")
#     print(f"全体統計:")
#     print(f"総データポイント数: {len(all_acc)}")
#     print(f"全体相関係数: {all_r_value:.3f}")
#     print(f"全体p値: {all_p_value:.3f}")
#     print(f"全体R²: {all_r_value**2:.3f}")


if __name__ == "__main__":
    # # CSVファイルを読み込み
    # low_f = 0
    # high_f = 70
    # # ieeg_acc_path = 'acc0-350_ep2600_overt.csv'
    # # ieeg_acc_path = 'to_plot_tsv_results_sp_ieeg_2600/spec0_350_acc.csv'
    # # ieeg_acc_path = 'to_plot_tsv_results_sp_ieeg_2600/spec70_140_acc.csv'
    # # ieeg_acc_path = 'to_plot_tsv_results_sp_ieeg_2600/spec280_350_acc.csv'
    # ieeg_acc_path = f'to_plot_tsv_results_sp_ieeg_2600/spec{low_f}_{high_f}_acc.csv'
    
    # # coh_eog_h_path = 'coh150-350_ep2600_all_mean_h.csv' 
    # coh_eog_h_path = f'coh_csv/coh_0_600_ep2600_right_{low_f}_{high_f}_all_mean.csv' 
    # # coh_eog_h_path = 'coh70_140_all_mean.csv' 
    # # coh_eog_h_path = 'coh280_350_all_mean.csv' 

    # # coh_eog_h_path = 'pli_150_350_all_mean.csv' 
    # # coh_eog_h_path = 'wpli_150_350_all_mean.csv' 
    
    # eog_acc_path = f'overt_eog_acc_'
    # # eog_acc_path = 'overt_eog_acc.csv'
    # # eog_acc_path = 'overt_eog_acc_70140.csv'
    # # eog_acc_path = 'overt_eog_acc_280350.csv'
    # # coh_eog_v_path = 'coh150-350_ep2600_all_mean_v.csv'
    # # save_path = 'scatter_all_participants_combined_h.png'
    # save_path = f'scatter_{coh_eog_h_path.split(".")[0].split(".")[0].split("/")[-1]}_{ieeg_acc_path.split("/")[-1].split(".")[0]}.pdf'
    
    # # out_dir = 'cohの帯域も変える/fig_acc280_350'
    # out_dir = './'
    # # 全参加者を1枚の図にまとめる関数を実行
    # # plot_all_participants_combined(ieeg_acc_path, coh_eog_h_path, coh_eog_v_path, eog_acc_path, save_path)
    # plot_all_participants_heog_combined(ieeg_acc_path, coh_eog_h_path, eog_acc_path, f'{out_dir}/{save_path}')
    # # plot_all_participants_overlaid()
    # # participant = 'js01'
    # # plot_individual_participants(ieeg_acc_path, coh_eog_h_path, save_path=f'{out_dir}/{participant}.png', participant=participant)

    out_dir = './ms_thesis'
    # f_list = [0, 70, 140, 210, 280, 350]
    # js_list = ["js01", "js07", "js08", "js11", "js13","js14", "js15"] # overt-covertでコヒーレンスに差があった人
    # js_list = ["js02", "js04", "js05", "js16"] # 差がない人
    js_list = ["js01", "js02", "js04", "js05", "js07", "js08", "js11", "js13","js14", "js15", "js16"] # 全員
    f_list = [70, 140]
    layout = (4, 3)
    eog_name = 'right'
    for l_f, h_f in zip(f_list[:-1], f_list[1:]):
        ieeg_acc_path = f'to_plot_tsv_results_sp_ieeg_2600/spec{l_f}_{h_f}_acc.csv'
        coh_eog_h_path = f'coh_csv/coh_0_600_ep2600_{eog_name}_{l_f}_{h_f}_all_mean.csv'
        # coh_eog_v_path = f'coh_csv/coh_0_600_ep2600_{eog_name}_overt_{l_f}_{h_f}_all_mean.csv'
        eog_acc_path = f'to_plot_tsv_results_sp_eog_2600/spec{l_f}_{h_f}_eog.csv'
        save_path = f'scatter_{eog_name}_{l_f}_{h_f}_all_mean.pdf'
        # save_path = f'scatter_{coh_eog_h_path.split(".")[0].split(".")[0].split("/")[-1]}_{ieeg_acc_path.split("/")[-1].split(".")[0]}.pdf'
        r_p_df = plot_all_participants_hveog_combined(ieeg_acc_path, 
                                                     coh_eog_h_path, 
                                                     eog_acc_path, 
                                                     eog_name,
                                                     f'{out_dir}/{save_path}', 
                                                     change_color_by_bad_ch=True, 
                                                     yaml_dir='js_yamls', 
                                                     js_list=js_list, 
                                                     layout=layout,
                                                     n_tests=len(js_list))
        # plot_all_participants_combined(ieeg_acc_path,
        #                                coh_eog_h_path,
        #                                coh_eog_v_path,
        #                                eog_acc_path,
        #                                f'{out_dir}/{save_path}',
        #                                change_color_by_bad_ch=True,
        #                                yaml_dir='js_yamls',
        #                                js_list=js_list,
        #                                layout=layout)
        # r_p_df.to_csv(f'{out_dir}/r_p_value_{l_f}_{h_f}.csv', index=True)
        break
