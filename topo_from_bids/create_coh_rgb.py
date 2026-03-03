import pandas as pd
import os
import copy
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import copy
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import json
from sklearn import preprocessing
import glob
import yaml
import numpy as np

# def vertex_to_df(vertex_file):
#     # 電極の位置座標をdictにまとめる
#     Elec = {}
#     with open(vertex_file, 'r') as f:
#         ch = 1
#         Cood = {}
#         for line in f.readlines():
#             Elec[ch] = []
#             cols = line.strip().split()
#             Cood['x'] = -1*float(cols[0])
#             Cood['y'] = float(cols[1])
#             Cood['z'] = float(cols[2])
#             Elec[ch] = copy.copy(Cood)
#             ch += 1
#     return pd.DataFrame(Elec)

def ter_to_rgb_df(to_rgb_s, vmin=None, vmax=None):
    cmsm = _set_cmap(to_rgb_s, vmin=vmin, vmax=vmax)
    # Elec_ter = {}
    # color = {}
    # # chとter
    # for index, ter in to_rgb_s.items():
    #     ch = index
    #     Elec_ter[ch] = []
    #     rgba = cmsm.to_rgba(ter)
    #     color['R'] = int(255 * rgba[0])
    #     color['G'] = int(255 * rgba[1])
    #     color['B'] = int(255 * rgba[2])
    #     Elec_ter[ch] = copy.copy(color)
    coh_rgb_list = []
    for coh in to_rgb_s:
        # if coh < 1:
        #     coh_rgb_list.append(np.array([0, 0, 1, 1]))  # 真っ青(R=0,G=0,B=1)にする
        # else:
        #     coh_rgb_list.append(cmsm.to_rgba(coh))
        coh_rgb_list.append(cmsm.to_rgba(coh))
    return np.array(coh_rgb_list)

# def select_ch(df, select_ch_list):
#     df = df.loc[:, select_ch_list]
#     return df

# def save_ter_rank_csv(df, filename, rgb_list=['R', 'G', 'B']):
#     print(df)
#     df = df.loc[['x', 'y', 'z']+rgb_list, :]
#     df = df.T
#     df.to_csv(filename, header=0, index=0)
#     print(f"save: {filename}")

def adjust_copper(cmap, start=0.0, stop=1.0, n=256):
    colors = cmap(np.linspace(start, stop, n))
    # 赤色成分を増加
    # colors[:, 0] = np.linspace(0.5, 1, n)  # 赤色成分を0.5から1に調整
    colors[:, 0] = 1  # 赤色成分を調整
    new_cmap = mcolors.LinearSegmentedColormap.from_list('adjusted_copper', colors, n)
    return new_cmap

def _set_cmap(rgb_ter_s, vmin=None, vmax=None):
    # TERとカラーマップの対応
    # vmin = 13
    # vmax = 50   # 50%（チャンスレベル）が最大TERがとして描画
    print(rgb_ter_s)
    # vmin = rgb_ter_s.min()
    # vmax = rgb_ter_s.max()
    if (vmin is None) or (vmax is None):
        vmin = 0.026
        vmax = 0.2
    cmap = cm.get_cmap('copper')    # カラーマップの選択
    # cmap = cm.get_cmap('seismic')    # カラーマップの選択
    inverted_cmap = cmap.reversed()  # カラーマップを反転
    # カラーマップを反転させる cmap(range(256))[::-1]で反転，'inverted_jet'という新しいcmap作成
    # inverted_cmap = LinearSegmentedColormap.from_list('inverted_copper', cmap(range(256))[::-1], N=256)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)    # ここの値をいじる
    # cmsm = cm.ScalarMappable(norm=norm, cmap=inverted_cmap)
    # cmsm = cm.ScalarMappable(norm=norm, cmap=cmap)

    adjusted_copper = adjust_copper(inverted_cmap)
    cmsm = cm.ScalarMappable(norm=norm, cmap=adjusted_copper)

    return cmsm

def create_colorbar(
    vmin=None,
    vmax=None,
    figsize=(2.5, 6),  # デフォルト太さ指定：幅2.5,高さ6
    save_path=None,
    format='png',
    dpi=300,
    label='coherence',
    bar_width=0.8  # barの幅(インチ)。0.35くらいで「ダイレクトに太さ指定」
):
    """
    シンプルな縦型カラーバーを出力する関数

    Parameters:
    -----------
    vmin : float, 必須
        カラーバーの最小値
    vmax : float, 必須
        カラーバーの最大値
    figsize : tuple, default (2.5, 6)
        図のサイズ（幅, 高さ）。全体キャンバスのサイズ。カラーバー自体の太さはbar_widthで調整
    save_path : str, optional
        保存先のパス（指定しない場合は表示のみ）
    format : str, default 'png'
        保存形式（'png', 'pdf', 'eps'など）
    dpi : int, default 300
        解像度
    label : str, default 'coherence'
        カラーバーのラベル
    bar_width : float, default 0.35
        実際のカラーバーの太さ（インチ）

    Returns:
    --------
    fig : matplotlib.figure.Figure
        作成された図オブジェクト
    """
    if (vmin is None) or (vmax is None):
        raise ValueError("vmin and vmax must be specified")

    # カラーマップ設定
    cmap = cm.get_cmap('copper')
    inverted_cmap = cmap.reversed()
    adjusted_copper = adjust_copper(inverted_cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmsm = cm.ScalarMappable(norm=norm, cmap=adjusted_copper)

    # 幅を調整するためにグリッド仕様のaxes領域を指定
    fig = plt.figure(figsize=figsize)
    # left, bottom, width, height でカラーバーの位置・サイズ（絶対値指定）
    # bar_width : カラーバー自体の幅（インチ単位）。高さはfigsizeで決まる
    left = 0.2  # figure左端からの相対位置（0.0 ~ 1.0）
    bottom = 0.05
    height = 0.9
    width = bar_width / figsize[0]  # bar_widthをfigsize幅で割って相対化

    cbar_ax = fig.add_axes([left, bottom, width, height])  # [left, bottom, width, height]
    cbar = plt.colorbar(
        cmsm, cax=cbar_ax, orientation='vertical'
    )
    cbar.set_label(label, rotation=90, va='bottom', fontsize=22)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.3f}', f'{vmax:.3f}'], fontsize=22)

    # 枠線など消す/余白詰める
    cbar.outline.set_linewidth(1)
    cbar_ax.tick_params(axis='y', length=8, width=1, labelsize=22)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if save_path is not None:
        plt.savefig(save_path, format=format, dpi=dpi, bbox_inches='tight')
        print(f"カラーバーを保存しました: {save_path}")

    plt.show()

    return fig


# if __name__ == '__main__':
#     coh_csv = 'coh150-350_ep2600_all_mean_h.csv'
#     js_name = 'js01'
#     out_dir = 'fig'
    
#     # mean_TERをdfに
#     df = pd.read_csv(coh_csv, index_col=0)
#     sub_coh_df = df[[js_name]].dropna()
#     # only_mean_ter_df = df[['mean_TER']].T
#     # sub_coh_df = sub_df.T   # 転置して電極を列に
#     # only_mean_ter_df.columns = df.index
#     # print(only_mean_ter_df)
#     # 電極座標をdfにしてmean_TERと合体
#     # vertex_df = vertex_to_df(vertex_file)
#     # conbined_df = pd.concat([vertex_df, only_mean_ter_df])
#     # print(conbined_df)
#     # ch_listに基づいて電極を選択（電極除去）
#     # select_ch_df = select_ch(conbined_df, ch_list)    # 目視による電極除去
#     # select_ch_df.loc['ter_rank', :] = select_ch_df.loc['mean_TER', :].rank()
#     # TERを正規化
#     # select_ch_df.loc['ter_minmax', :] = preprocessing.minmax_scale(select_ch_df.loc['mean_TER', :])
#     # print(select_ch_df)
#     # 
#     # rgb_ter_df = ter_to_rgb_df(select_ch_df.loc['ter_minmax', :]) # 正規化したTERを入れる
#     rgb_ter_np = ter_to_rgb_df(sub_coh_df[js_name]) # 正規化したTERを入れる
#     # rgb_ter_df = ter_to_rgb_df(select_ch_df.loc['ter_rank', :]) # chの順位入れる
#     # print(pd.concat([vertex_df, rgb_ter_df]))
#     # output_df = pd.concat([vertex_df, rgb_ter_df])  # vertex_dfとconcat（除去した電極はNanになる）
#     # output_df = rgb_ter_df.fillna(0) # Nanを0で埋めて黒色の点にする
#     # colors = list(rgb_ter_df)


#     # output_df = pd.concat([select_ch_df, rgb_ter_df])
#     # out_file = out_dir + '/' + f'{coh_csv.replace("/","_")}'
#     # save_ter_rank_csv(output_df, out_file)

#     import mne
#     from mne_bids import convert_montage_to_mri
#     import mne_bids

#     # edf_path = "/Users/murakamishoya/git/tt-lab/tt_github/junten-ibids-preprocess/junten_bids/sub-js01/ieeg/sub-js01_task-Speech8sen_run-1_ieeg.edf"
#     bidspath = mne_bids.BIDSPath(
#         subject="js01",
#         task="Speech8sen",
#         run="1",
#         root="/Users/murakamishoya/git/tt-lab/tt_github/junten-ibids-preprocess/junten_bids",
#     )
#     raw = mne_bids.read_raw_bids(bidspath, extra_params={"infer_types": True}, verbose=True)
#     # bidspath.suffix

#     subjects_dir = f"{bidspath.root}/sourcedata/sub-{bidspath.subject}"

#     montage = raw.get_montage()

#     # we need to go from scanner RAS back to surface RAS (requires recon-all)
#     convert_montage_to_mri(montage, "JS1_FT", subjects_dir=subjects_dir)

#     # this uses Freesurfer recon-all subject directory
#     montage.add_estimated_fiducials("JS1_FT", subjects_dir=subjects_dir)

#     # get head->mri trans, invert from mri->head
#     # trans2 = mne.transforms.invert_transform(mne.channels.compute_native_head_t(montage))

#     # now the montage is properly in "head" and ready for analysis in MNE
#     raw.set_montage(montage)


#     # compute the transform to head for plotting
#     trans = mne.channels.compute_native_head_t(montage)
#     # note that this is the same as:
#     # ``mne.transforms.invert_transform(
#     #      mne.transforms.combine_transforms(head_mri_t, mri_mni_t))``

#     # EEGチャンネル数を取得してcolors配列を作成（全て赤色）
#     import numpy as np
#     eeg_picks = mne.pick_types(raw.info, ecog=True)
#     n_eeg = len(eeg_picks)
#     # colors = np.array([[1.0, 0.0, 0.0]] * n_eeg)  # (n_eeg, 3)

#     view_kwargs = dict(azimuth=105, elevation=100, focalpoint=(0, 0, -15))
#     brain = mne.viz.Brain(
#         "JS1_FT",
#         subjects_dir=subjects_dir,
#         cortex="low_contrast",
#         alpha=0.25,
#         background="white",
#     )

#     brain.add_sensors(raw.info, trans=trans, sensor_colors=rgb_ter_np)
#     # brain.add_head(alpha=0.25, color="tan")
#     brain.show_view(distance=400, **view_kwargs)
#     # brain.save_image(filename='test.pdf', mode='rgb')

from create_coh_rgb import create_colorbar
vmin = 0.026
vmax = 0.245
create_colorbar(vmin=vmin, vmax=vmax, format='svg', label='coherence', save_path='coherence_colorbar.svg')