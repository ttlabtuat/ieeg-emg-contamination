clear;

% --- 1. FieldTripのパス追加 ---
ft_path = '/Users/murakamishoya/git/tt-lab/tt_github/M2_work/20260117_fieldtrip/fieldtrip-20241219';
addpath(ft_path);
ft_defaults; % ここで標準のパスが構成される


% --- ここから変換 ----%
% 参加者リストの設定
base_dir = '/Users/murakamishoya/git/tt-lab/tt_github/junten-ibids-preprocess';


% 保存先フォルダの作成
output_dir = 'mni_coord_norm';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% SPM12の確認
% if ~ft_hastoolbox('spm12')
%     error('SPM12が必要です');
% end

% FieldTrip同梱のColin27テンプレートを読み込み
template_file = fullfile(ft_path, 'template', 'anatomy', 'single_subj_T1.nii');
display(template_file);
% template_mri  = ft_read_mri(template_file);
% template_mri.coordsys = 'mni'; % テンプレートはMNI空間であることを明示

subjects = {
struct('sub_id', 'sub-js01', 'elec_file', sprintf('%s/junten_bids/sourcedata/sub-js01/JS1_elec_acpc_grid_f.mat', base_dir), 't1_file', sprintf('%s/junten_bids/sourcedata/sub-js01/JS1_FT/mri/T1.mgz', base_dir));
struct('sub_id', 'sub-js02', 'elec_file', sprintf('%s/junten_bids/sourcedata/sub-js02/JS2_elec_acpc_grid_f.mat', base_dir), 't1_file', sprintf('%s/junten_bids/sourcedata/sub-js02/JS2_FT/mri/T1.mgz', base_dir));
struct('sub_id', 'sub-js04', 'elec_file', sprintf('%s/junten_bids/sourcedata/sub-js04/JS4_elec_acpc_grid_f.mat', base_dir), 't1_file', sprintf('%s/junten_bids/sourcedata/sub-js04/JS4_FT/mri/T1.mgz', base_dir));
struct('sub_id', 'sub-js05', 'elec_file', sprintf('%s/junten_bids/sourcedata/sub-js05/JS5_elec_acpc_grid_f.mat', base_dir), 't1_file', sprintf('%s/junten_bids/sourcedata/sub-js05/JS5_FT/mri/T1.mgz', base_dir));
struct('sub_id', 'sub-js07', 'elec_file', sprintf('%s/junten_bids/sourcedata/sub-js07/JS7_elec_acpc_grid_f.mat', base_dir), 't1_file', sprintf('%s/junten_bids/sourcedata/sub-js07/JS7_FT/mri/T1.mgz', base_dir));
struct('sub_id', 'sub-js08', 'elec_file', sprintf('%s/junten_bids/sourcedata/sub-js08/JS8_elec_acpc_grid_f.mat', base_dir), 't1_file', sprintf('%s/junten_bids/sourcedata/sub-js08/JS8_FT/mri/T1.mgz', base_dir));
struct('sub_id', 'sub-js11', 'elec_file', sprintf('%s/junten_bids/sourcedata/sub-js11/J20004_HM_elec_acpc_grid_f.mat', base_dir), 't1_file', sprintf('%s/junten_bids/sourcedata/sub-js11/J20004_HM/mri/T1.mgz', base_dir));
struct('sub_id', 'sub-js13', 'elec_file', sprintf('%s/junten_bids/sourcedata/sub-js13/J21002_SK_elec_acpc_grid_f.mat', base_dir), 't1_file', sprintf('%s/junten_bids/sourcedata/sub-js13/J21002_SK/mri/T1.mgz', base_dir));
struct('sub_id', 'sub-js14', 'elec_file', sprintf('%s/junten_bids/sourcedata/sub-js14/J21003_MM_elec_acpc_grid_f.mat', base_dir), 't1_file', sprintf('%s/junten_bids/sourcedata/sub-js14/J21003_MM/mri/T1.mgz', base_dir));
struct('sub_id', 'sub-js15', 'elec_file', sprintf('%s/junten_bids/sourcedata/sub-js15/J21005_MH_elec_acpc_grid_f.mat', base_dir), 't1_file', sprintf('%s/junten_bids/sourcedata/sub-js15/J21005_MH/mri/T1.mgz', base_dir));
struct('sub_id', 'sub-js16', 'elec_file', sprintf('%s/junten_bids/sourcedata/sub-js16/J21008_SS_elec_acpc_grid_f.mat', base_dir), 't1_file', sprintf('%s/junten_bids/sourcedata/sub-js16/J21008_SS/mri/T1.mgz', base_dir));
};


for i = 1:length(subjects)
    sub = subjects{i};
    fprintf('処理中: %s\n', sub.sub_id);

    % 1. MRIと電極データの読み込み
    mri_acpc = ft_read_mri(sub.t1_file);
    mri_acpc.coordsys = 'acpc';
    load(sub.elec_file); % elec_acpc が読み込まれると仮定

    % 2. MNI空間への正規化
    cfg = [];
    cfg.nonlinear = 'yes';
    cfg.spmversion = 'spm12';
    cfg.template = template_file;
    source_diff = ft_volumenormalise(cfg, mri_acpc);

    % 3. 電極の座標変換 (Individual/ACPC -> Standard/MNI)
    elec_mni = elec_acpc_grid_f; 
    elec_mni.elecpos = ft_warp_apply(source_diff.params, elec_acpc_grid_f.elecpos, 'individual2sn');
    elec_mni.chanpos = ft_warp_apply(source_diff.params, elec_acpc_grid_f.chanpos, 'individual2sn');
    elec_mni.coordsys = 'mni'; 

    % 4. FieldTrip形式（.mat）で保存
    mat_filename = fullfile(output_dir, sprintf('%s_mni.mat', sub.sub_id));
    save(mat_filename, 'elec_mni');

    % 5. CSV形式で保存（追加部分）
    csv_filename = fullfile(output_dir, sprintf('%s_mni.csv', sub.sub_id));

    % テーブルの作成（電極ラベル，X, Y, Z）
    T = table(elec_mni.label, ...
              elec_mni.elecpos(:,1), ...
              elec_mni.elecpos(:,2), ...
              elec_mni.elecpos(:,3), ...
              'VariableNames', {'Label', 'MNI_X', 'MNI_Y', 'MNI_Z'});

    writetable(T, csv_filename);

    fprintf('保存完了: %s および %s\n', mat_filename, csv_filename);
end