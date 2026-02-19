#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import yaml
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def _set_pipline(kernel):
    print(kernel)
    pipeline = Pipeline([
        # ('scaler', StandardScaler()),  # データの標準化
        ('svm', SVC(kernel=kernel))  # サポートベクターマシン
    ])
    return pipeline

def build_grid_serch(kernel, cv=5):
    # パイプラインを作成
    pipeline = _set_pipline(kernel)

    # グリッドサーチのためのパラメータグリッド
    param_grid = {
        # 'svm__C': [0.001, 0.01, 0.1, 1, 10, 100]
        'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }

    if kernel == 'rbf':
        # param_grid['svm__gamma'] = [0.001, 0.01, 0.1, 1, 'auto']
        param_grid['svm__gamma'] = [0.001, 0.01, 0.1, 1, 10, 100, 1000]


    print(pipeline)
    print(param_grid)
    # グリッドサーチの実行
    return GridSearchCV(pipeline, param_grid, cv=cv)

def do_grid_serch(X, y, grid_search):
    grid_search.fit(X, y)

    # ベストなモデルとそのパラメータを表示
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best Model:", best_model)
    # print("Best Parameters:", best_params)

    return best_model