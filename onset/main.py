from cut_from_bids import create_epochs_from_bids_path
from detect_voice_onset_from_epoch import detect_voice_onset, visualize_onset
from preprocess import Preprocess as pp
from mne_bids import BIDSPath
import japanize_matplotlib
import soundfile as sf

if __name__ == "__main__":

    # js_list = ['js01', 'js02', 'js04', 'js05', 'js07', 'js08', 'js11', 'js13', 'js14', 'js15', 'js16']
    js_list = ['js11']
    # run_list = [4, 5]  # 対象のrun番号
    run_list = [2]  # 対象のrun番号
    for js in js_list:
        for run in run_list:
            path = BIDSPath(
                subject=js,
                task='Speech8sen',
                run=str(run),
                datatype='ieeg',
                root='/Users/murakamishoya/git/tt-lab/tt_github/cut-juntendata-iBIDS/junten_bids'
            )

            event_names = ['overt']  # 抽出するイベント名
            tmin = 0
            tmax = 3.3
            thresh_db = -20
            offset = 0.2
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
                    voice = pp.notch_filter(voice, re_sf=voice_epochs.info['sfreq'], notch_freq=50)
                    voice = (voice - voice.mean()) / voice.std()
                    onset_time, offset_time = detect_voice_onset(y=voice, sr=voice_epochs.info['sfreq'], threshold_db=thresh_db, offset=offset, output='step')
                    print(f"Detected onset time: {onset_time:.3f} seconds")
                    voice_df.loc[idx, 'voice_onset_sec'] = onset_time   # エポック開始からの音声開始時間
                    voice_df.loc[idx, 'voice_offset_sec'] = offset_time    # エポック開始からの音声終了時間
                    voice_df.loc[idx, 'duration'] = offset_time - onset_time    # 音声の持続時間（トリガではない）
                    voice_df.loc[idx, 'onset'] = voice_df.loc[idx, 'onset'] + onset_time    # rawの開始からの音声開始時間
                    voice_df.loc[idx, 'sample'] = int(voice_df.loc[idx, 'onset'] * voice_epochs.info['sfreq'])
                    print(voice_df)
                    visualize_onset(y=voice, sr=voice_epochs.info['sfreq'], onset_time=onset_time, offset_time=offset_time, threshold_db=thresh_db)

                    if voice_df.loc[idx, 'num'] == 51:
                        sf.write(
                            'test.wav',
                            voice,
                            int(voice_epochs.info['sfreq']),
                        )


            
            print(voice_df)
            voice_df_path = path.update(root='./voice_onset_events', suffix='events', extension='.tsv')
            # voice_df.to_csv(f"voice_onset_evetns/{voice_df_path.basename}", sep='\t', index=False)