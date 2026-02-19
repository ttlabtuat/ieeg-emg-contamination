% criterion_valueを全ファイルから抽出して表にまとめる

clear; close all; clc;

% ベースディレクトリ
base_dir = 'Contamination Analysis Package/Scripts/';

% 被験者リストを自動取得
subjects = dir(fullfile(base_dir, 'js*'));
subjects = subjects([subjects.isdir]);
type_pair = 'usable_ieeg_vs_voice';

% 結果を格納する構造体
results = {};

% 各被験者のデータを処理し，個別csvを保存
for i = 1:length(subjects)
    subject_name = subjects(i).name;
    subj_results = {}; % 各被験者の結果用セル配列

    % 各runを処理（run1-80）
    for run_num = 1:80
        run_name = sprintf('run%d', run_num);

        % ファイルパス
        mat_file = fullfile(base_dir, subject_name, run_name, ...
                           type_pair, [type_pair '_object.mat']);

        % ファイルが存在する場合のみ処理
        if exist(mat_file, 'file')
            try
                % MATファイルを読み込み
                data = load(mat_file);

                % objからcriterion_valueを取得（objはオブジェクトなのでispropを使用）
                if isfield(data, 'obj') && isprop(data.obj, 'criterion_value')
                    criterion_value = data.obj.criterion_value;

                    % 結果を格納
                    subj_results{end+1, 1} = subject_name;
                    subj_results{end, 2} = run_name;
                    subj_results{end, 3} = criterion_value;

                    fprintf('%s/%s: %.4f\n', subject_name, run_name, criterion_value);
                else
                    fprintf('%s/%s: criterion_value not found\n', subject_name, run_name);
                end
            catch ME
                fprintf('%s/%s: Error - %s\n', subject_name, run_name, ME.message);
            end
        else
            fprintf('%s/%s: File not found\n', subject_name, run_name);
        end
    end

    % 被験者ごとにテーブル・csv保存
    if ~isempty(subj_results)
        T_subj = cell2table(subj_results, 'VariableNames', {'Subject', 'ss', 'CriterionValue'});

        disp(' ');
        disp(['=== Criterion Value Summary for ' subject_name ' ===']);
        disp(T_subj);

        % 保存先
        output_file = fullfile(base_dir, type_pair, sprintf('%s_%s.csv', subject_name, type_pair));
        % フォルダ作成
        mkdir(fullfile(base_dir, type_pair));
        writetable(T_subj, output_file);
        fprintf('\n%sの結果を %s に保存しました．\n', subject_name, output_file);
    else
        disp([subject_name 'のデータが見つかりませんでした．']);
    end
    
    % % 被験者ごとのサマリー
    % disp(' ');
    % disp('=== Subject Summary (Mean ± Std) ===');
    % unique_subjects = unique(T.Subject);
    % for i = 1:length(unique_subjects)
    %     subj = unique_subjects{i};
    %     idx = strcmp(T.Subject, subj);
    %     // values = cell2mat(T.CriterionValue(idx));
    %     // fprintf('%s: %.4f ± %.4f (n=%d)\n', subj, mean(values), std(values), sum(idx));
    % end

end
