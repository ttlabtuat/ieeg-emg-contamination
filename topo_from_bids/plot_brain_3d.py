import mne_bids
import mne
import numpy as np
import os.path as op
from mne_bids import convert_montage_to_mri

def plot_brain_3d(bidspath, fs_sub_name, colors, view_kwargs):
    # edf_path = "/Users/murakamishoya/git/tt-lab/tt_github/junten-ibids-preprocess/junten_bids/sub-js01/ieeg/sub-js01_task-Speech8sen_run-1_ieeg.edf"
    raw = mne_bids.read_raw_bids(bidspath, extra_params={"infer_types": True}, verbose=True)
    # bidspath.suffix

    subjects_dir = f"{bidspath.root}/sourcedata/sub-{bidspath.subject}"

    montage = raw.get_montage()

    # we need to go from scanner RAS back to surface RAS (requires recon-all)
    convert_montage_to_mri(montage, fs_sub_name, subjects_dir=subjects_dir)

    # this uses Freesurfer recon-all subject directory
    montage.add_estimated_fiducials(fs_sub_name, subjects_dir=subjects_dir)

    # get head->mri trans, invert from mri->head
    # trans2 = mne.transforms.invert_transform(mne.channels.compute_native_head_t(montage))

    # now the montage is properly in "head" and ready for analysis in MNE
    raw.set_montage(montage)


    # compute the transform to head for plotting
    trans = mne.channels.compute_native_head_t(montage)
    # note that this is the same as:
    # ``mne.transforms.invert_transform(
    #      mne.transforms.combine_transforms(head_mri_t, mri_mni_t))``

    # EEGチャンネル数を取得してcolors配列を作成（全て赤色）
    # eeg_picks = mne.pick_types(raw.info, ecog=True)
    # n_eeg = len(eeg_picks)
    # colors = np.array([[1.0, 0.0, 0.0]] * n_eeg)  # (n_eeg, 3)

    brain = mne.viz.Brain(
        fs_sub_name,
        subjects_dir=subjects_dir,
        cortex="low_contrast",
        alpha=0.25,
        background="white",
        size=(800, 600),
    )

    brain.add_sensors(raw.info, trans=trans, sensor_colors=colors, sensor_scales=5)
    # 追加されたセンサーのPyVistaオブジェクトにアクセス

    # brain.add_head(alpha=0.25, color="tan")
    brain.show_view(distance=270, **view_kwargs)

    return brain
    