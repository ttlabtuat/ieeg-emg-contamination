#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import yaml
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from sklearn.metrics import accuracy_score


def _transform_label(tokens_array):
    """
    cv_listファイルのtranscriptsのトークン列を数値に変換
    """
    with open("conf/label.yaml") as label_file:
        label_dict = yaml.safe_load(label_file)
    inv_label_dict = {value: key for key, value in label_dict.items()}
    return [inv_label_dict[tokens] for tokens in tokens_array]

def load_single_ch_data(path, ch):
    """
    単一電極用のload_data
    単純に単一電極の特徴表現を平らにreshapeして返す
    pca用ではない
    """
    all_df = pd.read_csv(path, index_col=0)
    
    X_path = all_df["ecog_filename"].values
    X = [np.load(path)[ch-1] for path in X_path]
    reshaped_X = [x.reshape(-1) for x in X]
    print(reshaped_X[0].shape)
    
    y = all_df["transcripts"].values
    int_y = _transform_label(y)
    return np.array(reshaped_X), np.array(int_y)

def load_to_pca_data(path, ch):
    """
    pathのファイルの電極chをreshapeしてラベルとともに返す
    """
    all_df = pd.read_csv(path, index_col=0)
    
    X_path = all_df["ecog_filename"].values
    X = [np.load(path)[ch-1] for path in X_path]
    reshaped_X = [x.reshape(-1) for x in X]
    
    y = all_df["transcripts"].values
    int_y = _transform_label(y)
    return np.array(reshaped_X), np.array(int_y)


########### test ####################
def my_predict(model_path, X_test, y_test):
    # 保存したモデルを読み込む
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    # 予測
    pred_y = loaded_model.predict(X_test)

    # print(pred_y)
    ter = _calc_ter(y_test, pred_y)
    acc = _calc_acc(y_test, pred_y)
    print(f"TER: {ter}, acc: {acc}")
    return pred_y, ter, acc

def _calc_acc(y_test, pred_y):
    return accuracy_score(y_test, pred_y)

def _calc_ter(y_test, pred_y):
    error_token = 0
    if (len(y_test) == len(pred_y)):
        all_token = 3 * len(y_test) # 1文は3トークン
    else:
        print('error: len(y_test) != len(pred_y)')
        return None
    
    with open("conf/label.yaml") as label_file:
        label_dict = yaml.safe_load(label_file)
    for y_t, p_y in zip(y_test, pred_y):
        intersection_token = len(set(label_dict[y_t]) & set(label_dict[p_y]))
        error_token += (3 - intersection_token)
        # print(set(label_dict[y_t]) & set(label_dict[p_y]))

    return (error_token / all_token) * 100

def result_save(to_save_list, save_path):
    # テキストファイルへの書き込み
    with open(save_path, "w") as file:
        for item in to_save_list:
            file.write("%s\n" % item)

# multi_ch用
def load_all_feat(path):
    all_df = pd.read_csv(path, index_col=0)
    X_path = all_df["ecog_filename"].values
    X = [np.load(path) for path in X_path]
    reshaped_X = [x.reshape(-1) for x in X]

    y = all_df["transcripts"].values
    int_y = _transform_label(y)
    return np.array(reshaped_X), np.array(int_y)