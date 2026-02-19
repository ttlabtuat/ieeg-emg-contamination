import os
import glob
import mne_bids
from edf2mat4contami import raw2ndarray, epochs2ndarray, ndarray2mat
import sys
sys.path.append('junten-ibids-preprocess')
from common import get_all_run_bidspath, create_all_run_epochs
import yaml

def main():
    # 被験者IDを指定
    js_list = ("js01", "js02", "js04", "js05", "js07", "js08", "js11", "js13", "js14", "js15", "js16")
    # subject_id = 'js01'
    for subject_id in js_list:
        # BIDSディレクトリのパス
        bids_root = 'junten-ibids-preprocess/junten_bids'
        
        # 指定された被験者のEDFファイルを全て取得
        edf_pattern = os.path.join(bids_root, f'sub-{subject_id}', 'ieeg', '*.edf')
        edf_files = sorted(glob.glob(edf_pattern))
        
        if not edf_files:
            print(f"Warning: No EDF files found for subject {subject_id}")
            continue
        
        print(f"Found {len(edf_files)} EDF file(s) for subject {subject_id}")
        
        # 出力ディレクトリ構造を作成
        save_base_dir = 'output_usable_ieeg'
        ieeg_dir = os.path.join(save_base_dir, subject_id, 'ieeg')
        voice_dir = os.path.join(save_base_dir, subject_id, 'voice')
        sound_dir = os.path.join(save_base_dir, subject_id, 'sound')
        # eog_dir = os.path.join(save_base_dir, subject_id, 'eog')
        heog_dir = os.path.join(save_base_dir, subject_id, 'heog')
        veog_dir = os.path.join(save_base_dir, subject_id, 'veog')
        
        os.makedirs(ieeg_dir, exist_ok=True)
        os.makedirs(voice_dir, exist_ok=True)
        os.makedirs(sound_dir, exist_ok=True)
        # os.makedirs(eog_dir, exist_ok=True)
        os.makedirs(heog_dir, exist_ok=True)
        os.makedirs(veog_dir, exist_ok=True)
        
        # # 各EDFファイルを処理
        # for idx, path_edf in enumerate(edf_files, 1):
        #     print(f"\n[{idx}/{len(edf_files)}] Processing: {os.path.basename(path_edf)}")
            
        #     try:
        #         # EDFファイルを読み込み
        #         bids_path = mne_bids.get_bids_path_from_fname(path_edf)
        #         raw = mne_bids.read_raw_bids(bids_path)
        #         # eeg, voice, sound_r, eog = raw2ndarray(raw)
        #         eeg, voice, sound_r, eog = epochs2ndarray(raw)
        #         sf = raw.info['sfreq']
                
        #         # 保存名の作成
        #         base_name = os.path.basename(path_edf)
        #         base_name = base_name.replace(".edf", "")
                
        #         # 各ディレクトリに保存
        #         ndarray2mat(eeg, sf, os.path.join(ieeg_dir, f"{base_name}_ieeg.mat"))
        #         ndarray2mat(voice, sf, os.path.join(voice_dir, f"{base_name}_voice.mat"))
        #         ndarray2mat(sound_r, sf, os.path.join(sound_dir, f"{base_name}_sound_r.mat"))
        #         ndarray2mat(eog, sf, os.path.join(eog_dir, f"{base_name}_eog.mat"))
                
        #         print(f"  ✓ Successfully saved all MAT files for {base_name}")
                
        #     except Exception as e:
        #         print(f"  ✗ Error processing {os.path.basename(path_edf)}: {e}")
        #         continue
        
        # print(f"\n{'='*60}")
        # print(f"All files processed! Output directory: {save_base_dir}/{subject_id}/")
        # print(f"{'='*60}")
        # example usage

        root = 'junten-ibids-preprocess/junten_bids'

        js_name = subject_id
        task = "Speech8sen"
        
        bids_paths = get_all_run_bidspath(js_name, task, root)
        epochs = create_all_run_epochs(bids_paths, event_names=['overt'], tmin=-0.1, tmax=2.5)
        epoch_list = epochs2ndarray(epochs)
        sf = epochs.info['sfreq']

        js_yaml = yaml.load(open(f"js_yamls/{js_name}.yaml"), Loader=yaml.SafeLoader)
        usable_ch_idx = [x - 1 for x in js_yaml['usable_ch']]

        
        for i in range(len(epoch_list)):
            eeg, voice, sound_r, eog = epoch_list[i]
            h_eog = eog[0, :].reshape(1, -1)
            v_eog = eog[1, :].reshape(1, -1)
            usable_eeg = eeg[usable_ch_idx, :]
            ndarray2mat(usable_eeg, sf, os.path.join(ieeg_dir, f"{subject_id}_ss{i+1}_ieeg.mat"))
            ndarray2mat(voice, sf, os.path.join(voice_dir, f"{subject_id}_ss{i+1}_voice.mat"))
            ndarray2mat(sound_r, sf, os.path.join(sound_dir, f"{subject_id}_ss{i+1}_sound_r.mat"))
            ndarray2mat(h_eog, sf, os.path.join(heog_dir, f"{subject_id}_ss{i+1}_heog.mat"))
            ndarray2mat(v_eog, sf, os.path.join(veog_dir, f"{subject_id}_ss{i+1}_veog.mat"))

def print_bad_ch_count(js_list):
    for js_name in js_list:
        js_yaml = yaml.load(open(f"js_yamls/{js_name}.yaml"), Loader=yaml.SafeLoader)
        usable_ch = js_yaml['usable_ch']
        total_ch = js_yaml['total_num_ch']
        bad_ch = total_ch - len(usable_ch)
        print(f"{js_name}: {bad_ch}")
    return
    
if __name__ == "__main__":
    # main()
    js_list = ("js01", "js02", "js04", "js05", "js07", "js08", "js11", "js13", "js14", "js15", "js16")
    print_bad_ch_count(js_list)