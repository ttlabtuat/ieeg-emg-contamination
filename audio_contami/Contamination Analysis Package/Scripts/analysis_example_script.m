%% EXAMPLE OF CONTAMINATION ANALYSIS

% This script analyzes the spectrogram correlations between and audio file
% and a brain recording files. Both files should be MAT-files with a
% defined structure called RecordingMatfiles (see files in 'Test data'
% folder). 'Data_preparation_example' script shows how to create these
% files.

%% Add path to class and functions
addpath(genpath('../Toolbox/'));

%% Global parameters

% 被験者IDリスト
subject_id_list = {'js01', 'js02', 'js04', 'js05', 'js07', 'js08', 'js11', 'js13', 'js14', 'js15', 'js16'};
% subject_id_list = {'js14', 'js15', 'js16'};

% タスク情報
task_name = 'Speech8sen';

% 解析対象のデータ種別（手動で設定）
brain_type = 'ieeg';   % 脳電データの種類（'eog', 'ieeg'）
audio_type = 'voice';  % 音声データの種類（'voice', 'sound'）

%% Loop through all subjects
for subj_idx = 1:length(subject_id_list)
    subject_id = subject_id_list{subj_idx};
    
    fprintf('\n########################################\n');
    fprintf('### Processing Subject: %s ###\n', subject_id);
    fprintf('########################################\n\n');
    
    %% Loop through all runs
    for run_number = 1:80
        
        % ファイルパスを構築
        brain_matfile_path = sprintf('../../output_usable_ieeg/%s/%s/%s_ss%d_%s.mat', ...
            subject_id, brain_type, subject_id, run_number, brain_type);
        audio_matfile_path = sprintf('../../output_usable_ieeg/%s/%s/%s_ss%d_%s.mat', ...
            subject_id, audio_type, subject_id, run_number, audio_type);
        
        % 結果保存先とanalysis名を設定
        results_path = sprintf('%s/run%d', subject_id, run_number);
        analysis_name = sprintf('usable_%s_vs_%s', brain_type, audio_type);
            
        % ファイルの存在確認
        if ~exist(brain_matfile_path, 'file')
            fprintf('Warning: Brain file not found: %s\n', brain_matfile_path);
            continue;
        end
        if ~exist(audio_matfile_path, 'file')
            fprintf('Warning: Audio file not found: %s\n', audio_matfile_path);
            continue;
        end
        
        fprintf('\n========================================\n');
        fprintf('Processing: Run %d (%s vs %s)\n', run_number, brain_type, audio_type);
        fprintf('Brain file: %s\n', brain_matfile_path);
        fprintf('Audio file: %s\n', audio_matfile_path);
        fprintf('Results: %s/%s\n', results_path, analysis_name);
        fprintf('========================================\n\n');
        
        %% Create and store a ContaminationAnalysis object
        %
        % results_path:
        %   path to save the results
        % brain_matfile_path:
        %   brain data matfile path (should respect defined format)
        % brain_matfile_path:
        %   audio data matfile path (should respect defined format)
        % analysis_name (optional):
        %   name of files and figures related to the present analysis
        
        obj = ContaminationAnalysis(...
            results_path,...
            brain_matfile_path,...
            audio_matfile_path,...
            analysis_name);
        
        %% Select time samples that will be considered in the analysis
        %
        % select_periods:
        %   2-column array defining start and end times of the time periods to
        %   select.
        % exclude_periods:
        %   2-column array defining start and end times of the time periods to
        %   exclude.
        
        select_periods = [];
        % exclude_periods = [0 30]; % exclude the first 50 seconds
        exclude_periods = []; % exclude the first 50 seconds
        
        obj = selectTime(obj,...
            select_periods,...
            exclude_periods);
        
        %% Detect artifacts occuring on several channels
        %
        % moving_average_span:
        %   Duration (in seconds) of the moving average window that is used to
        %   detrend the data before artifact detection.
        % artifact_threshold_factor:
        %   'artifact_threshold_factor' multiplied by the MAD of a given channel
        %   defines the artifact threshold of this channel.
        % artifact_channel_ratio:
        %   Ratio of channels crossing their threshold for a sample to be
        %   considered as an artifact
        % artifact_safety_period:
        %   Period of time (in seconds) before and after artifact in which samples
        %   are also considered as artifacts
        
        moving_average_span = 0.5;
        artifact_threshold_factor = 10;
        artifact_channel_ratio = 1/10;
        artifact_safety_period = 0.5;
        
        obj = detectArtifacts(obj,...
            moving_average_span,...
            artifact_threshold_factor,...
            artifact_channel_ratio,...
            artifact_safety_period);
        
        %% Display the results of the artifact detection and save the figure
        %
        % display_channel_nb:
        %   Number of channels to show. The first half of the displayed channels
        %   are the channels with the highest numbers of artifact samples and the
        %   second half are the ones with the lowest numbers.
        %
        % Can return figure handle.
        
        display_channel_nb = min(4, obj.channel_nb);  % チャンネル数が少ない場合に対応
        
        displayArtifacts(obj, display_channel_nb)
        
        %% Compute the spectrograms of the audio and brain recordings
        %
        % window_duration:
        %   Duration of the spectrogram window (in seconds).
        % spg_fs:
        %   Desired sampling frequency of the spectrogram.
        % spg_freq_bounds:
        %   2-element vector containing the lowest and the highest frequencies
        %   considered in the spectrogram (if empty, all frequency bins are kept).
        
        window_duration = 200e-3;
        spg_fs = 50;
        spg_freq_bounds = [0 1000];
        
        obj = computeSpectrograms(obj,...
            window_duration, spg_fs,spg_freq_bounds);
        
        %% Compute spectrogram correlations between the audio and the brain data
        
        obj = computeSpectrogramCorrelations(obj);
        
        %% Display the spectrogram correlations and save the figures
        %
        % disp_freqs_bounds:
        %   2-element vector containing the lowest and the highest frequencies
        %   displayed in the spectrogram (if empty, all frequency bins are kept).
        % display_channels:
        %   'index' or 'id' of the channels to be displayed.
        % colormap_limits:
        %   2-element vector containing the lowest and the limits of the colormap
        %   displaying the z-scored spectrograls.
        %
        % Can return figure handles.
        
        display_channels = [];
        disp_freqs_bounds = [];
        colormap_limits = [0 5];
        
        displayCorrelations(obj, disp_freqs_bounds, display_channels, colormap_limits);
        
        %% Compute spectrogram cross-correlations between the audio and the brain data
        %
        % max_time_lag:
        %   Maximum absolute time lag in seconds considered when applying positive
        %   and negative delays to the audio spectrogram.
        
        max_time_lag = 0.5;
        
        obj = computeSpectrogramCrossCorrelations(obj, max_time_lag);
        
        
        %% Display cross-correlations
        %
        % crosscorr_min_max_freqs:
        %   2-element vector containing the lowest and the highest
        %   frequencies to be considered (if empty, all frequency bins are kept).
        % top_corr_ratio:
        %   Ratio of the highest cross-correlograms to display. 0.01 means that the
        %   1% of cross-correlograms reaching the highest values will be displayed.
        
        crosscorr_min_max_freqs = [75 1000]; % frequency range considered
        top_corr_ratio = 0.01; % ratio of the highest correlations to display
        
        displayCrossCorrelations(obj, crosscorr_min_max_freqs, top_corr_ratio)
        
        
        %% Compute statistical criterion P
        %
        % criterion_min_max_freqs:
        %   2-element vector containing the lowest and the highest
        %   frequencies to be considered (if empty, all frequency bins are kept).
        
        criterion_min_max_freqs = [75 Inf];
        
        obj = computeStatisticalCriterion(obj, criterion_min_max_freqs);
        
        %% Display statistical criterion P
        
        displayStatisticalCriterion(obj);
        
        %% Clean up large intermediate files
        % スペクトログラムファイルと移動平均処理済みファイルを削除
        
        fprintf('Cleaning up intermediate files...\n');
        
        % 移動平均処理済みの音声データを削除
        if exist(obj.centered_audio_matfile_path, 'file')
            delete(obj.centered_audio_matfile_path);
            fprintf('  Deleted: %s\n', obj.centered_audio_matfile_path);
        end
        
        % 移動平均処理済みの脳電データを削除
        if exist(obj.centered_brain_matfile_path, 'file')
            delete(obj.centered_brain_matfile_path);
            fprintf('  Deleted: %s\n', obj.centered_brain_matfile_path);
        end
        
        % 音声スペクトログラムを削除
        if exist(obj.audio_spg_path, 'file')
            delete(obj.audio_spg_path);
            fprintf('  Deleted: %s\n', obj.audio_spg_path);
        end
        
        % 脳電スペクトログラムを削除
        if exist(obj.brain_spg_path, 'file')
            delete(obj.brain_spg_path);
            fprintf('  Deleted: %s\n', obj.brain_spg_path);
        end
        
        fprintf('Cleanup complete.\n');
        
        %% Reset for next run
        % 次のランに影響を与えないようにリセット
        clear obj;
        
        % 図を安全に閉じる
        try
            drawnow;  % 描画キューを処理
            pause(0.5);  % 少し待機
            close all force;  % 全ての図を強制的に閉じる
            fprintf('All figures closed.\n');
        catch ME
            fprintf('Warning: Could not close all figures: %s\n', ME.message);
        end
    
    end % run loop
    
    fprintf('\n########################################\n');
    fprintf('### Completed Subject: %s ###\n', subject_id);
    fprintf('########################################\n\n');
    
end % subject loop

fprintf('\n========================================\n');
fprintf('All analyses completed for all subjects!\n');
fprintf('========================================\n');

