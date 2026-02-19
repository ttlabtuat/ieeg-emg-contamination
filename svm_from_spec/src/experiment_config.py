from dataclasses import dataclass
from typing import Optional, List, Tuple
import yaml
import os

def tuple_constructor(loader, node):
    """YAMLからタプルを読み込むためのコンストラクタ"""
    return tuple(loader.construct_sequence(node))

# タプルコンストラクタを登録
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

@dataclass
class SubjectConfig:
    """被験者ごとの設定を管理するクラス"""
    name: str
    original_sf: int
    n_ch: int
    usable_ch: List[int]
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SubjectConfig':
        """YAMLファイルから設定を読み込み"""
        with open(yaml_path) as js_file:
            js_yaml_dict = yaml.safe_load(js_file)
        
        # 電極番号80以上はなぜか保存されていない
        n_ch = js_yaml_dict['total_num_ch']
        # if n_ch > 80:
        #     n_ch = 80
            
        return cls(
            name=js_yaml_dict['name'],
            n_ch=n_ch,
            original_sf=js_yaml_dict['sr'],
            usable_ch=js_yaml_dict['usable_ch'],
            # task='t3'  # タスク名が指定されていない場合は't3'を使用
        )

@dataclass
class PreprocessConfig:
    """前処理の設定を管理するクラス"""
    re_sf: int = 2048  # リサンプリング後のサンプリングレート
    highpass_cutoff: float = 0.5  # ハイパスフィルターのカットオフ周波数
    highpass_order: int = 2  # ハイパスフィルターの次数
    notch_freq: float = 50.0  # ノッチフィルターの周波数
    win_len: int = 400  # ウィンドウ長（サンプル数）
    win_step: int = 350  # ウィンドウステップ（サンプル数）
    # stft_nperseg: int = 1024  # STFTのウィンドウ長
    # stft_noverlap: int = 512  # STFTのオーバーラップ
    stft_clip_fs: int = 80  # STFTの周波数クリップ
    normalizing: str = "zscore_all"  # 正規化方法
    mask_hz_list: Tuple[Tuple[int, int]] = None # マスクする周波数帯域のリスト [(start_hz, end_hz), ...]

@dataclass
class ExperimentConfig:
    """実験全体の設定を管理するクラス"""
    # データパス
    data_dir: str = "/shared/home/murakami/work/renamed_data"
    output_dir: str = "./results"
    task: str = "t3"
    
    # 実験設定
    n_repeats: int = 10  # 実験の繰り返し回数
    n_folds: int = 5  # 交差検証の分割数
    
    # 前処理設定
    preprocess: PreprocessConfig = PreprocessConfig()
    
    # SVM設定
    svm_kernel: str = "linear"
    svm_c_values: List[float] = None
    
    def __post_init__(self):
        if self.svm_c_values is None:
            self.svm_c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        
        # 出力ディレクトリの作成
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_config(self, save_path: str):
        """設定をYAMLファイルとして保存"""
        config_dict = {
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "task": self.task,
            "n_repeats": self.n_repeats,
            "n_folds": self.n_folds,
            "preprocess": {
                "re_sf": self.preprocess.re_sf,
                "highpass_cutoff": self.preprocess.highpass_cutoff,
                "highpass_order": self.preprocess.highpass_order,
                "notch_freq": self.preprocess.notch_freq,
                "win_len": self.preprocess.win_len,
                "win_step": self.preprocess.win_step,
                "stft_clip_fs": self.preprocess.stft_clip_fs,
                "normalizing": self.preprocess.normalizing,
                "mask_hz_list": self.preprocess.mask_hz_list
            },
            "svm": {
                "kernel": self.svm_kernel,
                "c_values": self.svm_c_values
            }
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load_config(cls, config_path: str) -> 'ExperimentConfig':
        """YAMLファイルから設定を読み込み"""
        with open(config_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        # 前処理設定の読み込み
        preprocess_config = PreprocessConfig(**config_dict["preprocess"])
        config_dict["preprocess"] = preprocess_config
        
        # SVM設定の読み込み
        if "svm" in config_dict:
            svm_config = config_dict.pop("svm")
            config_dict["svm_kernel"] = svm_config["kernel"]
            config_dict["svm_c_values"] = svm_config["c_values"]
        
        return cls(**config_dict)

if __name__ == "__main__":
    # 実験設定の初期化
    exp_config = ExperimentConfig()
    
    # 設定をYAMLファイルに保存
    exp_config.save_config("test.yaml")
