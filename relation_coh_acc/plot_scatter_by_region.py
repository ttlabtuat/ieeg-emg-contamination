import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
import math
import yaml
import japanize_matplotlib


def load_region_order(yaml_path, with_categories=False):
    """YAMLファイルから領域の順序を読み込む関数
    
    引数:
        yaml_path: YAMLファイルのパス
        with_categories: Trueの場合，カテゴリ情報も返す
    戻り値:
        with_categories=False: list: 領域名のリスト（順序付き）
        with_categories=True: tuple: (領域リスト, カテゴリ辞書)
            - 領域リスト: 領域名のリスト
            - カテゴリ辞書: {領域名: カテゴリ名}
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # brain_regionsから順序を抽出
    order = []
    region_to_category = {}
    if 'brain_regions' in data:
        for category, regions in data['brain_regions'].items():
            if regions:
                for region in regions:
                    order.append(region)
                    region_to_category[region] = category
    
    if with_categories:
        return order, region_to_category
    return order


def load_region_anatomical_mapping(yaml_path, lang='en'):
    """YAMLファイルから領域名と解剖学的表記のマッピングを読み込む関数
    
    引数:
        yaml_path: YAMLファイルのパス
        lang: 使用する言語 ('en' または 'jp')
    戻り値:
        dict: {領域名: 解剖学的表記} の辞書
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    mapping = {}
    if 'region_anatomical_mapping' in data:
        for region, info in data['region_anatomical_mapping'].items():
            if isinstance(info, dict) and lang in info:
                mapping[region] = info[lang]
            else:
                # マッピングがない場合は元の領域名を使用
                mapping[region] = region
    
    return mapping


def calc_correlation(coh_data, acc_data):
    """相関係数rとp値を計算する関数
    
    引数:
        coh_data: コヒーレンスのデータ（array-like）
        acc_data: 正解率のデータ（array-like）
    戻り値:
        r_value: 相関係数
        p_value: p値
    """
    r_value, p_value = stats.pearsonr(coh_data, acc_data)
    return r_value, p_value


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


def apply_fdr_correction(region_p_values, region_names, n_tests=None):
    _, p_corrected, _, _ = multipletests(region_p_values, alpha=0.05, method='fdr_bh')
    return {region_names[j]: p_corrected[j] for j in range(len(region_names))}

def apply_bonferroni_correction(region_p_values, region_names, n_tests=None):
    _, p_corrected, _, _ = multipletests(region_p_values, alpha=0.05, method='bonferroni')
    return {region_names[j]: p_corrected[j] for j in range(len(region_names))}


def calc_correlation_ci(r_value, n, alpha=0.05):
    """相関係数の信頼区間を計算する関数（フィッシャーのz変換を使用）
    
    引数:
        r_value: 相関係数
        n: サンプルサイズ
        alpha: 有意水準（デフォルト: 0.05 → 95%信頼区間）
    戻り値:
        ci_lower: 信頼区間の下限
        ci_upper: 信頼区間の上限
    """
    # サンプルサイズが小さすぎる場合はNaNを返す
    if n <= 3:
        return np.nan, np.nan
    
    # フィッシャーのz変換
    z = np.arctanh(r_value)
    
    # z変換後の標準誤差
    se = 1 / np.sqrt(n - 3)
    
    # z値（95%信頼区間の場合は1.96）
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    # z変換空間での信頼区間
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se
    
    # 逆変換してrの信頼区間を求める
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)
    
    return ci_lower, ci_upper

def calc_cohens_d(data, mu0):
    """Cohen's dを計算する関数
    
    引数:
        data: データ（array-like）
        mu0: 比較する基準値（チャンスレベルなど）
    戻り値:
        d: Cohen's d
    """
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # 不偏標準偏差
    if std == 0:
        return np.nan
    d = (mean - mu0) / std
    return d


def get_acc_stats_df(csv_path, participant=None, to_plot_region=None, n_tests=None, chance_level=None):
    """領域ごとのacc平均とチャンスレベルとのt検定結果のDataFrameを出力する関数
    
    引数:
        csv_path: all_sub_coh_acc.csv のパス
        participant: 参加者リスト（タプルまたはリスト）。Noneの場合は全参加者
        to_plot_region: 対象の領域リスト。Noneの場合はすべての領域
        n_tests: 多重比較補正のための検定回数。Noneの場合は実際の検定数を使用
        chance_level: チャンスレベル。指定した場合，accがチャンスレベルより有意に高いかをt検定で評価
    戻り値:
        pd.DataFrame: 以下の列を含むDataFrame
            - region: 領域名
            - n: データ数
            - acc_mean: accの平均
            - acc_cohens_d: Cohen's d（効果量）
            - acc_t_stat: t統計量
            - acc_p_raw: 生のp値（片側検定）
            - acc_p_fdr: FDR補正後のp値
            - acc_p_bonferroni: ボンフェローニ補正後のp値
            - acc_sig_raw, acc_sig_fdr, acc_sig_bonferroni: 有意性マーカー
    """
    # CSVファイルを読み込み
    df = pd.read_csv(csv_path)
    if participant is not None:
        df = df[df['sub'].isin(participant)]
    
    # accが存在するデータのみを使用
    df_valid = df.dropna(subset=['acc'])
    
    # 領域ごとにグループ化（UnknownとNaNを除外）
    regions = [r for r in df_valid['region'].unique() if pd.notna(r) and r != 'Unknown']
    regions = sorted(regions)
    
    # to_plot_regionが指定されている場合は，そのリストに含まれる領域のみをフィルタリング
    if to_plot_region is not None:
        regions = [r for r in regions if r in to_plot_region]
    
    # 各領域の統計情報を収集
    results = []
    region_names = []
    acc_p_values = []
    
    for region in regions:
        region_data = df_valid[df_valid['region'] == region]
        acc_data = region_data['acc'].values
        
        if len(acc_data) > 1:
            acc_mean = acc_data.mean()
            
            result_dict = {
                'region': region,
                'N (Elec.)': len(acc_data),
                'acc_mean': acc_mean
            }
            
            # チャンスレベルとのt検定（片側検定: チャンスレベルより高いか）
            if chance_level is not None:
                t_stat, acc_p = stats.ttest_1samp(acc_data, chance_level, alternative='greater')
                cohens_d = calc_cohens_d(acc_data, chance_level)
                result_dict['acc_cohens_d'] = cohens_d
                result_dict['acc_t_stat'] = t_stat
                result_dict['acc_p_raw'] = acc_p
                acc_p_values.append(acc_p)
            
            results.append(result_dict)
            region_names.append(region)
    
    # 多重比較補正を適用
    if chance_level is not None and len(acc_p_values) > 0:
        fdr_dict = apply_fdr_correction(acc_p_values, region_names)
        bonferroni_dict = apply_bonferroni_correction(acc_p_values, region_names, n_tests)
        
        for result in results:
            region = result['region']
            result['acc_p_fdr'] = fdr_dict[region]
            result['acc_p_bonferroni'] = bonferroni_dict[region]
            result['acc_sig_raw'] = get_significance_marker(result['acc_p_raw'])
            result['acc_sig_fdr'] = get_significance_marker(result['acc_p_fdr'])
            result['acc_sig_bonferroni'] = get_significance_marker(result['acc_p_bonferroni'])
    
    # DataFrameを作成
    df_stats = pd.DataFrame(results)
    
    # 列の順序を整理
    if chance_level is not None:
        column_order = ['region', 'N (Elec.)', 'acc_mean', 'acc_cohens_d', 'acc_t_stat', 'acc_p_raw', 'acc_sig_raw', 
                        'acc_p_fdr', 'acc_sig_fdr', 'acc_p_bonferroni', 'acc_sig_bonferroni']
    else:
        column_order = ['region', 'N (Elec.)', 'acc_mean']
    df_stats = df_stats[column_order]
    
    return df_stats

def get_correlation_stats_df(csv_path, participant=None, to_plot_region=None, n_tests=None):
    """領域ごとの相関係数とp値（ボンフェローニとFDR），有意性マーカーのDataFrameを出力する関数
    
    引数:
        csv_path: all_sub_coh_acc.csv のパス
        participant: 参加者リスト（タプルまたはリスト）。Noneの場合は全参加者
        to_plot_region: 対象の領域リスト。Noneの場合はすべての領域
        n_tests: 多重比較補正のための検定回数。Noneの場合は実際の検定数を使用
    戻り値:
        pd.DataFrame: 以下の列を含むDataFrame
            - region: 領域名
            - n: データ数
            - corr_r: 相関係数
            - corr_ci_lower: 95%信頼区間の下限
            - corr_ci_upper: 95%信頼区間の上限
            - corr_p_raw: 生のp値
            - corr_p_fdr: FDR補正後のp値
            - corr_p_bonferroni: ボンフェローニ補正後のp値
            - corr_sig_raw: 生のp値に基づく有意性マーカー
            - corr_sig_fdr: FDR補正後の有意性マーカー
            - corr_sig_bonferroni: ボンフェローニ補正後の有意性マーカー
    """
    # CSVファイルを読み込み
    df = pd.read_csv(csv_path)
    if participant is not None:
        df = df[df['sub'].isin(participant)]
    
    # cohとaccが両方存在するデータのみを使用
    df_valid = df.dropna(subset=['coh', 'acc'])
    
    # 領域ごとにグループ化（UnknownとNaNを除外）
    regions = [r for r in df_valid['region'].unique() if pd.notna(r) and r != 'Unknown']
    regions = sorted(regions)
    
    # to_plot_regionが指定されている場合は，そのリストに含まれる領域のみをフィルタリング
    if to_plot_region is not None:
        regions = [r for r in regions if r in to_plot_region]
    
    # 各領域の統計情報を収集
    results = []
    region_names = []
    region_p_values = []
    
    for region in regions:
        region_data = df_valid[df_valid['region'] == region]
        coh_data = region_data['coh'].values
        acc_data = region_data['acc'].values
        
        if len(coh_data) > 1:
            r_value, p_value = calc_correlation(coh_data, acc_data)
            ci_lower, ci_upper = calc_correlation_ci(r_value, len(coh_data))
            
            result_dict = {
                'region': region,
                'N (Elec.)': len(coh_data),
                'corr_r': r_value,
                'corr_ci_lower': ci_lower,
                'corr_ci_upper': ci_upper,
                'corr_p_raw': p_value
            }
            
            results.append(result_dict)
            region_names.append(region)
            region_p_values.append(p_value)
    
    # 多重比較補正を適用（相関係数）
    if len(region_p_values) > 0:
        fdr_dict = apply_fdr_correction(region_p_values, region_names)
        bonferroni_dict = apply_bonferroni_correction(region_p_values, region_names, n_tests)
        
        # 補正後のp値と有意性マーカーを追加
        for result in results:
            region = result['region']
            result['corr_p_fdr'] = fdr_dict[region]
            result['corr_p_bonferroni'] = bonferroni_dict[region]
            result['corr_sig_raw'] = get_significance_marker(result['corr_p_raw'])
            result['corr_sig_fdr'] = get_significance_marker(result['corr_p_fdr'])
            result['corr_sig_bonferroni'] = get_significance_marker(result['corr_p_bonferroni'])
    
    # DataFrameを作成
    df_stats = pd.DataFrame(results)
    
    # 列の順序を整理
    column_order = ['region', 'N (Elec.)', 'corr_r', 'corr_ci_lower', 'corr_ci_upper', 
                    'corr_p_raw', 'corr_sig_raw', 'corr_p_fdr', 'corr_sig_fdr', 
                    'corr_p_bonferroni', 'corr_sig_bonferroni']
    df_stats = df_stats[column_order]
    
    return df_stats


def plot_forest(df_stats, output_path, correction_type='fdr', sort_by='r', region_order=None, figsize=None, show_table=True, anatomical_label_lang='en', show_ci=True, fontsize_label=16, fontsize_axis=16, fontsize_title=16, fontsize_table_header=16, fontsize_table_data=16):
    """フォレストプロットを作成する関数
    
    引数:
        df_stats: get_correlation_stats_dfとget_acc_stats_dfをマージしたDataFrame，またはCSVパス
        output_path: 保存先ファイルパス
        correction_type: 有意性マーカーに使用する補正タイプ ('raw', 'fdr', 'bonferroni')
        sort_by: ソート基準 ('r': 相関係数順, 'region': 領域名順, 'p': p値順, None: ソートなし)
                 region_orderが指定されている場合は無視される
        region_order: 領域の表示順序（リストまたはYAMLファイルパス）
        figsize: 図のサイズ (width, height)。Noneの場合は自動計算
        show_table: Trueの場合，右側に表（n, acc_mean, r [95% CI]）を表示
        anatomical_label_lang: 解剖学的表記の言語 ('en' または 'jp')
        show_ci: Trueの場合，相関係数に95%信頼区間を表示
        fontsize_label: 領域名・カテゴリラベルのフォントサイズ（デフォルト: 11）
        fontsize_axis: 軸ラベルのフォントサイズ（デフォルト: 14）
        fontsize_title: タイトルのフォントサイズ（デフォルト: 14）
        fontsize_table_header: 表のヘッダーのフォントサイズ（デフォルト: 12）
        fontsize_table_data: 表のデータのフォントサイズ（デフォルト: 11）
    戻り値:
        なし（図ファイルを保存）
    """
    # DataFrameまたはCSVパスを処理
    if isinstance(df_stats, str):
        df = pd.read_csv(df_stats)
    else:
        df = df_stats.copy()
    
    # 領域名と解剖学的表記のマッピングを読み込む
    region_name_mapping = {}
    if region_order is not None and isinstance(region_order, str) and region_order.endswith('.yaml'):
        region_name_mapping = load_region_anatomical_mapping(region_order, lang=anatomical_label_lang)
    elif region_order is None:
        # region_orderが指定されていない場合でも，デフォルトのYAMLファイルを探す
        default_yaml = 'region_plot_order.yaml'
        if os.path.exists(default_yaml):
            region_name_mapping = load_region_anatomical_mapping(default_yaml, lang=anatomical_label_lang)
    
    # 領域名を解剖学的表記に変換する関数
    def get_anatomical_name(region):
        return region_name_mapping.get(region, region)
    
    # acc_mean列があるかチェック
    has_acc_mean = 'acc_mean' in df.columns
    
    # 相関係数用の有意性マーカーの列名を決定（corr_プレフィックス対応）
    corr_sig_col = f'corr_sig_{correction_type}'
    if corr_sig_col not in df.columns:
        # 旧形式の列名も試す
        corr_sig_col = f'sig_{correction_type}'
        if corr_sig_col not in df.columns:
            print(f"警告: corr_sig_{correction_type}列が見つかりません。corr_sig_rawを使用します。")
            corr_sig_col = 'corr_sig_raw' if 'corr_sig_raw' in df.columns else 'sig_raw'
    
    # acc用の有意性マーカー列
    acc_sig_col = f'acc_sig_{correction_type}'
    
    # 相関係数列の名前を決定（corr_プレフィックス対応）
    r_col = 'corr_r' if 'corr_r' in df.columns else 'r'
    ci_lower_col = 'corr_ci_lower' if 'corr_ci_lower' in df.columns else 'ci_lower'
    ci_upper_col = 'corr_ci_upper' if 'corr_ci_upper' in df.columns else 'ci_upper'
    p_raw_col = 'corr_p_raw' if 'corr_p_raw' in df.columns else 'p_raw'
    
    # ソート
    region_to_category = None
    if region_order is not None:
        # YAMLファイルパスの場合は読み込む
        if isinstance(region_order, str) and region_order.endswith('.yaml'):
            region_order, region_to_category = load_region_order(region_order, with_categories=True)
        
        # 指定された順序でソート（逆順にして下から上に表示）
        order_dict = {region: i for i, region in enumerate(reversed(region_order))}
        df['_sort_order'] = df['region'].map(order_dict)
        # 順序が定義されていない領域は最後に
        df['_sort_order'] = df['_sort_order'].fillna(len(region_order))
        df = df.sort_values('_sort_order', ascending=True)
        df = df.drop('_sort_order', axis=1)
    elif sort_by == 'r':
        df = df.sort_values(r_col, ascending=True)
    elif sort_by == 'region':
        df = df.sort_values('region', ascending=True)
    elif sort_by == 'p':
        df = df.sort_values(p_raw_col, ascending=False)
    
    df = df.reset_index(drop=True)
    
    # 図のサイズを設定
    n_regions = len(df)
    
    # カテゴリラベル用のサブプロットを追加するかどうか
    show_category_labels = region_to_category is not None
    
    # acc_cohens_d列があるかチェック
    has_cohens_d = 'acc_cohens_d' in df.columns
    
    # figsizeを調整（カテゴリラベルがある場合は幅を追加）
    if figsize is None:
        base_height = max(5, n_regions * 0.4)
        if show_table:
            if has_acc_mean:
                base_width = 11  # Cohen's dはMean Acc列に統合
            else:
                base_width = 9
        else:
            base_width = 8
        if show_category_labels:
            base_width += 1.2  # カテゴリラベル分の幅を追加
        figsize = (base_width, base_height)
    
    ax_region = None  # 領域名用のサブプロット
    
    # 領域名の最大文字数に基づいて幅を計算（解剖学的表記 + 電極数を含む）
    max_region_len = 0
    if len(df) > 0:
        for _, row in df.iterrows():
            anatomical_name = get_anatomical_name(row['region'])
            n = int(row['N (Elec.)'])
            region_label = f'{anatomical_name} ({n})'
            max_region_len = max(max_region_len, len(region_label))
    else:
        max_region_len = 10
    region_width_ratio = max(0.8, max_region_len * 0.09)  # 文字数に応じた幅
    
    # カテゴリ名の最大文字数に基づいて幅を計算
    if region_to_category is not None:
        unique_categories = set(region_to_category.values())
        max_category_len = max(len(str(cat)) for cat in unique_categories) if unique_categories else 8
        category_width_ratio = max(0.4, max_category_len * 0.06)  # 文字数に応じた幅
    else:
        category_width_ratio = 0.5
    
    if show_table:
        # show_ciがFalseの場合は表の幅を少し狭くする
        table_width_factor = 0.75 if not show_ci else 1.0
        if show_category_labels:
            # 4つのサブプロット（カテゴリラベル + 領域名 + フォレストプロット + 表）
            if has_acc_mean:
                width_ratios = [category_width_ratio, region_width_ratio, 3, 1.8]
            else:
                width_ratios = [category_width_ratio, region_width_ratio, 3, 1.5]
            fig, (ax_category, ax_region, ax, ax_table) = plt.subplots(1, 4, figsize=figsize, 
                                                gridspec_kw={'width_ratios': width_ratios, 'wspace': 0})
        else:
            # 2つのサブプロット（フォレストプロット + 表）
            if has_acc_mean:
                width_ratios = [3, 3.5 * table_width_factor]
            else:
                width_ratios = [3, 3.0 * table_width_factor]
            fig, (ax, ax_table) = plt.subplots(1, 2, figsize=figsize, 
                                                gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.02})
            ax_category = None
    else:
        if show_category_labels:
            # 3つのサブプロット（カテゴリラベル + 領域名 + フォレストプロット）
            width_ratios = [category_width_ratio, region_width_ratio, 3]
            fig, (ax_category, ax_region, ax) = plt.subplots(1, 3, figsize=figsize, 
                                                gridspec_kw={'width_ratios': width_ratios, 'wspace': 0})
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax_category = None
    
    # y座標
    y_pos = np.arange(n_regions)
    
    # 色の設定（有意な場合は色を変える）
    colors = []
    for _, row in df.iterrows():
        sig = row[corr_sig_col] if corr_sig_col in df.columns and pd.notna(row[corr_sig_col]) else ''
        if sig:  # 有意な場合
            colors.append('#1f77b4')  # 青
        else:
            colors.append('black')  # 黒
    
    # カテゴリごとの背景色
    category_groups = []  # (カテゴリ名, 開始位置, 終了位置) のリスト
    if region_to_category is not None:
        # カテゴリの色（交互に）
        category_colors = ['#f0f0f0', '#ffffff']
        
        # 各領域のカテゴリを取得
        df_categories = [region_to_category.get(row['region'], 'Unknown') for _, row in df.iterrows()]
        
        # カテゴリごとにグループ化して背景を描画
        current_category = None
        category_idx = 0
        group_start = 0
        
        for i, cat in enumerate(df_categories):
            if cat != current_category:
                if current_category is not None:
                    # 前のグループを記録
                    category_groups.append((current_category, group_start, i - 1))
                    # 前のグループの背景を描画
                    bg_color = category_colors[category_idx % 2]
                    ax.axhspan(group_start - 0.5, i - 0.5, facecolor=bg_color, alpha=0.5, zorder=0)
                    if show_table:
                        ax_table.axhspan(group_start - 0.5, i - 0.5, facecolor=bg_color, alpha=0.5, zorder=0)
                    if ax_category is not None:
                        ax_category.axhspan(group_start - 0.5, i - 0.5, facecolor=bg_color, alpha=0.5, zorder=0)
                    if ax_region is not None:
                        ax_region.axhspan(group_start - 0.5, i - 0.5, facecolor=bg_color, alpha=0.5, zorder=0)
                    category_idx += 1
                current_category = cat
                group_start = i
        
        # 最後のグループ
        if current_category is not None:
            category_groups.append((current_category, group_start, n_regions - 1))
            bg_color = category_colors[category_idx % 2]
            ax.axhspan(group_start - 0.5, n_regions - 0.5, facecolor=bg_color, alpha=0.5, zorder=0)
            if show_table:
                ax_table.axhspan(group_start - 0.5, n_regions - 0.5, facecolor=bg_color, alpha=0.5, zorder=0)
            if ax_category is not None:
                ax_category.axhspan(group_start - 0.5, n_regions - 0.5, facecolor=bg_color, alpha=0.5, zorder=0)
            if ax_region is not None:
                ax_region.axhspan(group_start - 0.5, n_regions - 0.5, facecolor=bg_color, alpha=0.5, zorder=0)
        
        # カテゴリラベルを描画
        if ax_category is not None:
            ax_category.set_xlim(0, 1)
            ax_category.set_ylim(-0.5, n_regions - 0.5)
            ax_category.axis('off')
            
            for cat_name, start, end in category_groups:
                # グループの中央にラベルを配置
                y_center = (start + end) / 2
                ax_category.text(0.5, y_center, cat_name, fontsize=fontsize_label, fontweight='bold',
                               ha='center', va='center', rotation=0)
        
        # 領域名ラベルを描画
        if ax_region is not None:
            ax_region.set_xlim(0, 1)
            ax_region.set_ylim(-0.5, n_regions - 0.5)
            ax_region.axis('off')
            
            # 領域名のヘッダーを追加
            header_y = n_regions - 0.3
            ax_region.text(0.95, header_y, 'Region (N electrodes)', fontsize=fontsize_table_header, fontweight='bold', ha='right', va='bottom')
            
            for i, (_, row) in enumerate(df.iterrows()):
                region = row['region']
                anatomical_name = get_anatomical_name(region)
                n = int(row['N (Elec.)'])
                region_label = f'{anatomical_name} ({n})'
                ax_region.text(0.95, i, region_label, fontsize=fontsize_label, ha='right', va='center', color=colors[i])
    
    # acc_meanが存在する場合、点のサイズをaccの値に基づいて計算
    if has_acc_mean:
        acc_values = df['acc_mean'].values
        acc_min = acc_values.min()
        acc_max = acc_values.max()
        # 点のサイズを50から200の範囲にスケール
        if acc_max > acc_min:
            point_sizes = 50 + (acc_values - acc_min) / (acc_max - acc_min) * 250
        else:
            point_sizes = np.full(len(acc_values), 100)
    else:
        point_sizes = np.full(len(df), 100)
    
    # エラーバー付きの点をプロット
    for i, (_, row) in enumerate(df.iterrows()):
        r = row[r_col]
        ci_lower = row[ci_lower_col]
        ci_upper = row[ci_upper_col]
        
        # エラーバー
        ax.plot([ci_lower, ci_upper], [i, i], color=colors[i], linewidth=2, solid_capstyle='round')
        # 点（accの値に応じてサイズを変更）
        ax.scatter(r, i, color=colors[i], s=point_sizes[i], zorder=5)
    
    # 0の参照線
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    # y軸の設定
    ax.set_yticks(y_pos)
    if ax_region is not None:
        # 領域名は別のサブプロットに表示するので、ここでは非表示
        ax.set_yticklabels([])
    else:
        # 領域名をy軸ラベルとして表示
        y_labels = []
        for _, row in df.iterrows():
            region = row['region']
            anatomical_name = get_anatomical_name(region)
            n = int(row['N (Elec.)'])
            region_label = f'{anatomical_name} ({n})'
            y_labels.append(region_label)
        ax.set_yticklabels(y_labels, fontsize=fontsize_label)
    
    # 軸の設定
    ax.set_xlabel('Correlation coefficient (r)', fontsize=fontsize_axis)
    ax.set_xlim(-0.5, 1)
    ax.set_ylim(-0.5, n_regions - 0.5)
    
    # グリッド
    ax.grid(True, axis='x', alpha=0.3)
    
    # サブプロット間の境界を非表示にして背景色を連続させる
    if ax_region is not None:
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
    
    # タイトル
    correction_labels = {'raw': 'uncorrected', 'fdr': 'FDR corrected', 'bonferroni': 'Bonferroni corrected'}
    title_suffix = correction_labels.get(correction_type, correction_type)
    # ax.set_title(f'({title_suffix})', fontsize=fontsize_title)
    
    # 表を追加
    if show_table:
        ax_table.set_xlim(0, 1)
        ax_table.set_ylim(-0.5, n_regions - 0.5)
        ax_table.axis('off')
        
        # acc_cohens_d列があるかチェック
        has_cohens_d = 'acc_cohens_d' in df.columns
        
        # 各列の最大文字数を計算して列幅を決定
        # Mean Acc列: Cohen's dを統合した場合は「0.00*** (d=0.00)」で約18文字
        if has_cohens_d:
            acc_header_len = len('Acc (Cohen\'s d)')
            acc_data_max_len = 18  # 「0.00*** (d=-0.00)」の約18文字
        else:
            acc_header_len = len('Acc  (Cohen\'s d)')
            acc_data_max_len = 7  # 「0.00***」の7文字
        acc_col_max_len = max(acc_header_len, acc_data_max_len)
        
        # r列: ヘッダーとデータの最大文字数を計算
        if show_ci:
            r_header_len = len('r [95% CI]')
            # データは「0.00*** [-0.00, 0.00]」のような形式、最大約25文字
            r_data_max_len = 25
        else:
            r_header_len = len('r')
            r_data_max_len = 7  # 「0.00***」の7文字
        r_col_max_len = max(r_header_len, r_data_max_len)
        
        # 列間のスペース（文字数単位、各列の15%）
        spacing_factor = 0.15
        
        # 各列の幅を計算（文字数 + スペース）
        if has_acc_mean:
            acc_col_width = acc_col_max_len * (1 + spacing_factor)
            r_col_width = r_col_max_len * (1 + spacing_factor)
            total_width = r_col_width + acc_col_width
        else:
            r_col_width = r_col_max_len * (1 + spacing_factor)
            total_width = r_col_width
        
        # 各列の中心位置を計算（左から累積、0-1の範囲に正規化）
        # rを左、Accを右に配置
        if has_acc_mean:
            r_col_start = (r_col_width / 2) / total_width
            acc_col_start = (r_col_width + acc_col_width / 2) / total_width
        else:
            r_col_start = (r_col_width / 2) / total_width
        
        # ヘッダー行を少し下に配置
        header_y = n_regions - 0.3  # 元の -0.1 より下へ
        
        # ヘッダーラベルのオフセット
        acc_header_offset = 0.05  # Mean Accラベルを右にずらす
        
        if has_acc_mean:
            # ヘッダー（acc_mean列あり，Cohen's dを統合）
            if has_cohens_d:
                ax_table.text(acc_col_start + acc_header_offset, header_y, 'Acc (Cohen\'s d)', fontsize=fontsize_table_header, fontweight='bold', ha='center', va='bottom')
            else:
                ax_table.text(acc_col_start + acc_header_offset, header_y, 'Acc (Cohen\'s d)', fontsize=fontsize_table_header, fontweight='bold', ha='center', va='bottom')
            if show_ci:
                ax_table.text(r_col_start, header_y, 'r [95% CI]', fontsize=fontsize_table_header, fontweight='bold', ha='center', va='bottom')
            else:
                ax_table.text(r_col_start, header_y, 'r', fontsize=fontsize_table_header, fontweight='bold', ha='center', va='bottom')
        else:
            # ヘッダー
            if show_ci:
                ax_table.text(r_col_start, header_y, 'r [95% CI]', fontsize=fontsize_table_header, fontweight='bold', ha='center', va='bottom')
            else:
                ax_table.text(r_col_start, header_y, 'r', fontsize=fontsize_table_header, fontweight='bold', ha='center', va='bottom')
        # 数値とマーカーを分けて配置するためのオフセット（座標系の0-1での値）
        # r列用のオフセット（列の中心から右にずらす）
        r_num_offset = 0.08
        # acc列用のオフセット（列の中心から左にずらす）
        acc_num_offset = 0.08
        
        # 各行のデータ
        for i, (_, row) in enumerate(df.iterrows()):
            r = row[r_col]
            ci_lower = row[ci_lower_col]
            ci_upper = row[ci_upper_col]
            r_sig = row[corr_sig_col] if corr_sig_col in df.columns and pd.notna(row[corr_sig_col]) else ''
            
            if has_acc_mean:
                acc_mean = row['acc_mean']
                acc_sig = row[acc_sig_col] if acc_sig_col in df.columns and pd.notna(row[acc_sig_col]) else ''
                
                # Mean Acc列（数値とマーカーを分けて配置，Cohen's dを統合）
                if has_cohens_d:
                    cohens_d = row['acc_cohens_d']
                    if pd.notna(cohens_d):
                        # 数値部分（右寄せ）
                        acc_num_text = f'{acc_mean:.2f}'
                        # マーカー + Cohen's d部分（左寄せ）
                        acc_sig_text = f'{acc_sig} (d={cohens_d:.2f})'
                    else:
                        acc_num_text = f'{acc_mean:.2f}'
                        acc_sig_text = f'{acc_sig}'
                else:
                    acc_num_text = f'{acc_mean:.2f}'
                    acc_sig_text = f'{acc_sig}'
                
                # 数値を右寄せで配置
                ax_table.text(acc_col_start - acc_num_offset, i, acc_num_text, fontsize=fontsize_table_data, ha='right', va='center', color=colors[i])
                # マーカーを左寄せで配置
                ax_table.text(acc_col_start - acc_num_offset, i, acc_sig_text, fontsize=fontsize_table_data, ha='left', va='center', color=colors[i])
                
                # r [95% CI]列（数値とマーカーを分けて配置）
                if show_ci:
                    # 数値部分（右寄せ）
                    r_num_text = f'{r:.2f}'
                    # マーカー + CI部分（左寄せ）
                    r_sig_text = f'{r_sig} [{ci_lower:.2f}, {ci_upper:.2f}]'
                    ax_table.text(r_col_start + r_num_offset, i, r_num_text, fontsize=fontsize_table_data, ha='right', va='center', color=colors[i])
                    ax_table.text(r_col_start + r_num_offset, i, r_sig_text, fontsize=fontsize_table_data, ha='left', va='center', color=colors[i])
                else:
                    r_num_text = f'{r:.2f}'
                    r_sig_text = f'{r_sig}'
                    ax_table.text(r_col_start + r_num_offset, i, r_num_text, fontsize=fontsize_table_data, ha='right', va='center', color=colors[i])
                    ax_table.text(r_col_start + r_num_offset, i, r_sig_text, fontsize=fontsize_table_data, ha='left', va='center', color=colors[i])
            else:
                # r [95% CI]列（数値とマーカーを分けて配置）
                if show_ci:
                    r_num_text = f'{r:.2f}'
                    r_sig_text = f'{r_sig} [{ci_lower:.2f}, {ci_upper:.2f}]'
                    ax_table.text(r_col_start + r_num_offset, i, r_num_text, fontsize=fontsize_table_data, ha='right', va='center', color=colors[i])
                    ax_table.text(r_col_start + r_num_offset, i, r_sig_text, fontsize=fontsize_table_data, ha='left', va='center', color=colors[i])
                else:
                    r_num_text = f'{r:.2f}'
                    r_sig_text = f'{r_sig}'
                    ax_table.text(r_col_start + r_num_offset, i, r_num_text, fontsize=fontsize_table_data, ha='right', va='center', color=colors[i])
                    ax_table.text(r_col_start + r_num_offset, i, r_sig_text, fontsize=fontsize_table_data, ha='left', va='center', color=colors[i])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    
    print(f"フォレストプロットを保存しました: {output_path}")


def save_individual_plots(df_valid, regions, participants, participant_color_map, corrected_dict, 
                          all_coh_min, all_coh_max, output_dir, get_anatomical_name=None):
    """各領域の散布図を個別に保存する関数
    
    引数:
        df_valid: 有効なデータを含むDataFrame
        regions: 領域のリスト
        participants: 参加者のリスト
        participant_color_map: 参加者ごとの色マップ
        corrected_dict: 補正後のp値の辞書
        all_coh_min: 横軸の最小値
        all_coh_max: 横軸の最大値
        output_dir: 出力ディレクトリ
        get_anatomical_name: 領域名を解剖学的表記に変換する関数。Noneの場合は元の領域名を使用
    戻り値:
        なし（図ファイルを保存）
    """
    # 領域名を解剖学的表記に変換する関数（デフォルト）
    if get_anatomical_name is None:
        def _get_anatomical_name(region):
            return region
        get_anatomical_name = _get_anatomical_name
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 各領域の散布図を個別に保存
    for region in regions:
        region_data = df_valid[df_valid['region'] == region]
        coh_data = region_data['coh'].values
        acc_data = region_data['acc'].values
        
        if len(coh_data) > 1:
            # 個別のfigureを作成
            fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
            
            # 参加者ごとに色分けしてプロット
            for participant in participants:
                participant_data = region_data[region_data['sub'] == participant]
                if len(participant_data) > 0:
                    participant_coh = participant_data['coh'].values
                    participant_acc = participant_data['acc'].values
                    ax_ind.scatter(participant_coh, participant_acc, 
                                 alpha=0.7, s=70, 
                                 color=participant_color_map[participant],
                                 label=participant)
            
            # 相関を計算
            r_value, p_value = calc_correlation(coh_data, acc_data)
            
            # 補正後のp値を使用（補正が適用されている場合）
            if region in corrected_dict:
                p_value_used = corrected_dict[region]
            else:
                p_value_used = p_value
            
            # 有意性マーカーを決定
            p_text = get_significance_marker(p_value_used)
            
            # 軸の設定
            ax_ind.set_xlabel('Coherence', fontsize=16)
            ax_ind.set_ylabel('Classification Accuracy', fontsize=16)
            # 領域名を解剖学的表記に変換
            anatomical_name = get_anatomical_name(region)
            ax_ind.set_title(f'{anatomical_name}\nr = {r_value:.3f} {p_text}\nn = {len(coh_data)}', fontsize=18)
            ax_ind.grid(True, alpha=0.3)
            
            # 軸範囲を統一
            ax_ind.set_ylim(0, 1)
            if all_coh_min < all_coh_max:
                pad = (all_coh_max - all_coh_min) * 0.05
                ax_ind.set_xlim(all_coh_min - pad, all_coh_max + pad)
            else:
                pad = 0.01
                ax_ind.set_xlim(all_coh_min - pad, all_coh_max + pad)
            
            # 凡例を追加
            ax_ind.legend(loc='best', fontsize=10, frameon=True)
            
            # ファイル名を生成（領域名から特殊文字を除去）
            safe_region_name = region.replace('/', '_').replace('\\', '_').replace(' ', '_')
            individual_output_path = os.path.join(output_dir, f'{safe_region_name}.png')
            
            plt.tight_layout()
            plt.savefig(individual_output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"個別図を保存しました: {individual_output_path}")
        else:
            # データがない場合も空の図を保存
            fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
            # 領域名を解剖学的表記に変換
            anatomical_name = get_anatomical_name(region)
            ax_ind.text(0.5, 0.5, f'{anatomical_name}\nNo data', ha='center', va='center', 
                       transform=ax_ind.transAxes, fontsize=14)
            ax_ind.set_title(anatomical_name, fontsize=18)
            ax_ind.set_ylim(0, 1)
            if all_coh_min < all_coh_max:
                pad = (all_coh_max - all_coh_min) * 0.05
                ax_ind.set_xlim(all_coh_min - pad, all_coh_max + pad)
            else:
                pad = 0.01
                ax_ind.set_xlim(all_coh_min - pad, all_coh_max + pad)
            
            safe_region_name = region.replace('/', '_').replace('\\', '_').replace(' ', '_')
            individual_output_path = os.path.join(output_dir, f'{safe_region_name}.png')
            
            plt.tight_layout()
            plt.savefig(individual_output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"個別図を保存しました（データなし）: {individual_output_path}")


def plot_scatter_by_region(csv_path, output_path, participant=None, layout=None, to_plot_region=None, n_tests=None, save_individual=False, output_dir=None, correction_type='fdr', region_order=None, anatomical_label_lang='en'):
    """全参加者のデータを読み込み，脳領域ごとにcohとaccの散布図を1つの画像にまとめて作成する関数
    
    引数:
        csv_path: all_sub_coh_acc.csv のパス
        output_path: 保存先ファイルパス（まとめた図）
        participant: 参加者リスト（タプルまたはリスト）
        layout: サブプロットのレイアウト (row, col)。Noneの場合は自動計算
        to_plot_region: 表示する領域のリスト。Noneの場合はすべての領域を表示
        n_tests: 多重比較補正のための検定回数。Noneの場合は補正なし
        save_individual: Trueの場合，各領域の散布図を個別に保存
        output_dir: 個別ファイルの保存先ディレクトリ（save_individual=Trueの場合に使用）
        correction_type: 多重比較補正のタイプ ('fdr', 'bonferroni', 'raw')
        region_order: 領域の表示順序（リストまたはYAMLファイルパス）。Noneの場合はソート順
        anatomical_label_lang: 解剖学的表記の言語 ('en' または 'jp')
    戻り値:
        なし（図ファイルを保存）
    """
    plt.rcParams["font.size"] = 14
    
    # 領域名と解剖学的表記のマッピングを読み込む
    region_name_mapping = {}
    region_order_list = None
    if region_order is not None:
        if isinstance(region_order, str) and region_order.endswith('.yaml'):
            region_order_list, _ = load_region_order(region_order, with_categories=True)
            region_name_mapping = load_region_anatomical_mapping(region_order, lang=anatomical_label_lang)
        elif isinstance(region_order, list):
            region_order_list = region_order
    
    # 領域名を解剖学的表記に変換する関数
    def get_anatomical_name(region):
        return region_name_mapping.get(region, region)
    
    # CSVファイルを読み込み
    df = pd.read_csv(csv_path)
    if participant is not None:
        df = df[df['sub'].isin(participant)]
    # cohとaccが両方存在するデータのみを使用
    df_valid = df.dropna(subset=['coh', 'acc'])
    
    # 領域ごとにグループ化（UnknownとNaNを除外）
    regions = [r for r in df_valid['region'].unique() if pd.notna(r) and r != 'Unknown']
    
    # to_plot_regionが指定されている場合は，そのリストに含まれる領域のみをフィルタリング
    if to_plot_region is not None:
        regions = [r for r in regions if r in to_plot_region]
    
    # データ数が2以上の領域のみをフィルタリング
    valid_regions = []
    for region in regions:
        region_data = df_valid[df_valid['region'] == region]
        if len(region_data) > 1:
            valid_regions.append(region)
    
    regions = valid_regions
    
    # region_orderが指定されている場合は，その順序でソート
    if region_order_list is not None:
        # 指定された順序でソート（順序に含まれていない領域は最後に）
        order_dict = {region: i for i, region in enumerate(region_order_list)}
        regions = sorted(regions, key=lambda x: (order_dict.get(x, len(region_order_list)), x))
    else:
        # デフォルトはソート順
        regions = sorted(regions)
    
    print(f"見つかった領域数: {len(regions)}")
    print(f"領域リスト: {regions}")
    print("-" * 50)
    
    if len(regions) == 0:
        print("有効な領域が見つかりませんでした。")
        return
    
    # サブプロットのレイアウトを決定
    if layout is not None:
        row, col = layout
    else:
        # 自動計算：領域数に応じて適切なレイアウトを決定
        n_regions = len(regions)
        col = math.ceil(math.sqrt(n_regions))
        row = math.ceil(n_regions / col)
    
    # 各サブプロットを4x4インチに設定
    subplot_size = 4
    figsize = (col * subplot_size, row * subplot_size)
    fig, axes = plt.subplots(row, col, figsize=figsize)
    if row == 1 and col == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # 1次元配列に変換
    
    # 全データのaccuracy範囲とcoherence範囲を計算（軸統一のため）
    all_acc_min = float('inf')
    all_acc_max = float('-inf')
    all_coh_min = float('inf')
    all_coh_max = float('-inf')
    
    # 図の縦軸範囲と横軸範囲を決めるために，全領域のデータの最大値と最小値を計算
    for region in regions:
        region_data = df_valid[df_valid['region'] == region]
        coh_data = region_data['coh'].values
        acc_data = region_data['acc'].values
        
        if len(coh_data) > 0:
            all_acc_min = min(all_acc_min, acc_data.min())
            all_acc_max = max(all_acc_max, acc_data.max())
            all_coh_min = min(all_coh_min, coh_data.min())
            all_coh_max = max(all_coh_max, coh_data.max())
    
    # データが存在しない場合のデフォルト値
    if all_acc_min == float('inf'):
        all_acc_min = 0.0
        all_acc_max = 1.0
    if all_coh_min == float('inf'):
        all_coh_min = 0.0
        all_coh_max = 0.1
    
    # 参加者リストを取得して色を割り当て
    participants = sorted(df_valid['sub'].unique())
    n_participants = len(participants)
    # カラーマップを使用して各参加者に色を割り当て
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_participants, 10)))
    if n_participants > 10:
        # 10色以上必要な場合は追加のカラーマップを使用
        colors2 = plt.cm.Set3(np.linspace(0, 1, n_participants - 10))
        colors = np.vstack([colors, colors2])
    participant_color_map = {p: colors[j] for j, p in enumerate(participants)}
    
    # 凡例用のハンドルとラベルを保存（すべての参加者を含む）
    legend_handles = []
    legend_labels = []
    first_plot_with_data = None  # データがある最初のサブプロットのインデックス
    
    # 多重比較補正のために，まずすべての領域のp値を収集
    region_p_values = []
    region_info = []  # (region, r_value, p_value, coh_data, acc_data, index) を保存
    
    # 第1パス: すべての領域のp値を収集
    for i, region in enumerate(regions):
        region_data = df_valid[df_valid['region'] == region]
        coh_data = region_data['coh'].values
        acc_data = region_data['acc'].values
        
        if len(coh_data) > 1:
            r_value, p_value = calc_correlation(coh_data, acc_data)
            region_p_values.append(p_value)
            region_info.append((region, r_value, p_value, coh_data, acc_data, i))
    
    # 多重比較補正を適用
    region_names = [info[0] for info in region_info]
    if correction_type == 'fdr':
        corrected_dict = apply_fdr_correction(region_p_values, region_names, n_tests)
    elif correction_type == 'bonferroni':
        corrected_dict = apply_bonferroni_correction(region_p_values, region_names, n_tests)
    else:  # 'raw' またはその他
        # 補正なしの場合は生のp値を使用
        corrected_dict = {region_names[j]: region_p_values[j] for j in range(len(region_names))}
    
    # 第2パス: プロット作成
    for i, region in enumerate(regions):
        if i < len(axes):  # サブプロット数以内の場合のみ処理
            region_data = df_valid[df_valid['region'] == region]
            
            # cohとaccのデータを取得
            coh_data = region_data['coh'].values
            acc_data = region_data['acc'].values
            
            if len(coh_data) > 1:
                # データがある最初のサブプロットを記録
                if first_plot_with_data is None:
                    first_plot_with_data = i
                
                # 参加者ごとに色分けしてプロット
                for participant in participants:
                    participant_data = region_data[region_data['sub'] == participant]
                    if len(participant_data) > 0:
                        participant_coh = participant_data['coh'].values
                        participant_acc = participant_data['acc'].values
                        # データがある最初のサブプロットのみlabelを設定（凡例用）
                        if i == first_plot_with_data:
                            scatter = axes[i].scatter(participant_coh, participant_acc, 
                                          alpha=0.7, s=70, 
                                          color=participant_color_map[participant],
                                          label=participant)
                            # 凡例用のハンドルとラベルを保存
                            if participant not in legend_labels:
                                legend_handles.append(scatter)
                                legend_labels.append(participant)
                        else:
                            axes[i].scatter(participant_coh, participant_acc, 
                                          alpha=0.7, s=70, 
                                          color=participant_color_map[participant])
                
                # 相関を計算（全データを使用）
                r_value, p_value = calc_correlation(coh_data, acc_data)
                
                # 補正後のp値を使用（補正が適用されている場合）
                if region in corrected_dict:
                    p_value_used = corrected_dict[region]
                else:
                    p_value_used = p_value
                
                # 有意性マーカーを決定
                p_text = get_significance_marker(p_value_used)
                
                # サブプロットの設定
                if i % col == 0:  # 左端の列のみ縦軸ラベルを表示
                    axes[i].set_ylabel('Classification Accuracy', fontsize=16)
                else:
                    axes[i].set_ylabel('')
                
                if i >= (row - 1) * col:  # 下段のみ横軸ラベルを表示
                    axes[i].set_xlabel('Coherence', fontsize=16)
                else:
                    axes[i].set_xlabel('')
                
                # 領域名を解剖学的表記に変換
                anatomical_name = get_anatomical_name(region)
                axes[i].set_title(f'{anatomical_name}\nr = {r_value:.2f} {p_text}\nn = {len(coh_data)}', fontsize=14)
                axes[i].grid(True, alpha=0.3)
                
                # 縦軸と横軸の範囲を統一
                axes[i].set_ylim(0, 1)
                if all_coh_min < all_coh_max:
                    pad = (all_coh_max - all_coh_min) * 0.05
                    axes[i].set_xlim(all_coh_min - pad, all_coh_max + pad)
                else:
                    pad = 0.01
                    axes[i].set_xlim(all_coh_min - pad, all_coh_max + pad)
                
                # 統計情報
                print(f"{region}の統計情報:")
                print(f"データポイント数: {len(coh_data)}")
                correction_label = {'fdr': 'FDR補正後', 'bonferroni': 'Bonferroni補正後', 'raw': '生'}.get(correction_type, '補正後')
                print(f"相関係数: {r_value:.3f}, p値: {p_value:.4f}", end="")
                if region in corrected_dict:
                    print(f", {correction_label}p値: {p_value_used:.4f}")
                else:
                    print()
                print(f"Accuracy - 平均: {acc_data.mean():.3f}，標準偏差: {acc_data.std():.3f}")
                print(f"Coherence - 平均: {coh_data.mean():.3f}，標準偏差: {coh_data.std():.3f}")
                print("-" * 50)
            else:
                # 領域名を解剖学的表記に変換
                anatomical_name = get_anatomical_name(region)
                axes[i].text(0.5, 0.5, f'{anatomical_name}\nNo data', ha='center', va='center', 
                           transform=axes[i].transAxes)
                axes[i].set_title(anatomical_name, fontsize=14)
                # データがない場合も軸範囲を統一
                axes[i].set_ylim(0, 1)
                if all_coh_min < all_coh_max:
                    pad = (all_coh_max - all_coh_min) * 0.05
                    axes[i].set_xlim(all_coh_min - pad, all_coh_max + pad)
                else:
                    pad = 0.01
                    axes[i].set_xlim(all_coh_min - pad, all_coh_max + pad)
    
    # すべての参加者が凡例に含まれるように、不足している参加者のハンドルを作成
    if first_plot_with_data is not None:
        for participant in participants:
            if participant not in legend_labels:
                # ダミーの散布図を作成して凡例に追加
                dummy_scatter = axes[first_plot_with_data].scatter([], [], 
                                                                  alpha=0.7, s=70, 
                                                                  color=participant_color_map[participant],
                                                                  label=participant)
                legend_handles.append(dummy_scatter)
                legend_labels.append(participant)
    
    # 使用されていないサブプロットを非表示
    for i in range(len(regions), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 全体の凡例を図の外に配置（すべての参加者を含む）
    if len(legend_handles) > 0 and len(legend_labels) > 0:
        # ラベルの順序を参加者の順序に合わせる
        sorted_legend_items = sorted(zip(legend_handles, legend_labels), 
                                     key=lambda x: participants.index(x[1]) if x[1] in participants else len(participants))
        sorted_handles, sorted_labels = zip(*sorted_legend_items) if sorted_legend_items else ([], [])
        fig.legend(sorted_handles, sorted_labels, loc='upper right', bbox_to_anchor=(1.02, 1.0), 
                  fontsize=10, frameon=True, ncol=1)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"全領域を1枚にまとめた図が保存されました: {output_path}")
    print(f"統一された縦軸範囲（Accuracy）: {all_acc_min:.3f} - {all_acc_max:.3f}")
    print(f"統一された横軸範囲（Coherence）: {all_coh_min:.3f} - {all_coh_max:.3f}")
    
    # 個別保存の処理
    if save_individual:
        # 出力ディレクトリの設定
        if output_dir is None:
            # output_pathからディレクトリを取得
            output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        
        save_individual_plots(df_valid, regions, participants, participant_color_map, corrected_dict, all_coh_min, all_coh_max, output_dir, get_anatomical_name=get_anatomical_name)


if __name__ == "__main__":
    eog_type = 'heog'
    csv_path = f'all_sub_coh_acc_multi5_{eog_type}.csv'
    participant = ('js01', 'js02', 'js04', 'js05', 'js07', 'js08', 'js11','js14', 'js15', 'js16')
    brain_region_groups = [
        "Frontal_L",
        "Frontal_Broca_L",
        "Orbitofrontal_L",
        "Frontal_Motor_L",

        "Parietal_L",
        "Parietal_Sensory_L",
        "Angular_Superior_L",

        "Temporal_Inf_L",
        "Temporal_Mid_L",
        "Temporal_Sup_L",
        "Temporal_Pole_L",
        "Temporal_Fusiform_L",

        "Occipital_L",

        "ParaHippocampal_L",
        # "Other",
        # "Unknown",
        "Right_Hemisphere"
    ]
    # 多重比較補正のための検定回数（15個のグループで検定する場合）
    n_tests = 15
    # チャンスレベル（8クラス分類の場合は1/8=0.125）
    chance_level = 0.125
    # 補正タイプ: 'fdr' または 'bonferroni' または 'raw'（補正なし）
    correction_type = 'fdr'
    # 領域の表示順序（YAMLファイル）
    region_order_yaml = 'region_plot_order_anat.yaml'
    layout = (4, 4)
    label = 'abbr'
    
    plot_scatter_by_region(csv_path, output_path=f'scatter_by_region_{eog_type}_{correction_type}_{label}.pdf', 
                          participant=participant, to_plot_region=brain_region_groups, n_tests=n_tests, save_individual=False, output_dir=f'./', correction_type=correction_type, region_order=region_order_yaml, anatomical_label_lang=label, layout=layout)

    # plot_scatter_by_region(csv_path, output_path=f'./{csv_path.split("/")[-1].split(".")[0]}_by_region.png', 
    #                       participant=participant, to_plot_region=brain_region_groups, n_tests=n_tests, save_individual=True, output_dir=f'./{csv_path.split("/")[-1].split(".")[0]}_by_region_individual', correction_type=correction_type, region_order=region_order_yaml)
    
    # 相関係数の統計情報を取得
    df_corr = get_correlation_stats_df(
        csv_path, 
        participant=participant, 
        to_plot_region=brain_region_groups, 
        n_tests=n_tests
    )
    
    # accの統計情報を取得
    df_acc = get_acc_stats_df(
        csv_path,
        participant=participant,
        to_plot_region=brain_region_groups,
        n_tests=n_tests,
        chance_level=chance_level
    )
    print(df_acc)
    print("-" * 50)
    print(df_corr)
    # regionをキーにしてマージ
    df_merged = pd.merge(df_corr, df_acc, on=['region', 'N (Elec.)'], how='outer')
    
    # CSVに保存
    df_merged.to_csv(f'修論_修正/all_correlation_stats_{eog_type}_{correction_type}_{label}.csv', index=False)
    print(df_merged)
    
    # フォレストプロットを作成（acc列付き，YAML順序で表示）
    plot_forest(df_merged, f'修論_修正/forest_plot_{eog_type}_{correction_type}_{label}.pdf', 
                correction_type=correction_type, region_order=region_order_yaml, show_ci=False, anatomical_label_lang=label)