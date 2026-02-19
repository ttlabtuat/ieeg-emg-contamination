%% EXAMPLE OF DATA FORMATTING

% This script shows how to create files should be MAT-files with a
% defined structure called RecordingMatfiles (see files in 'Test data'
% folder).

%% Add path to class and functions
addpath(genpath('../Toolbox/'));

%% Output file path

output_path = '../Data/Formatting example/fake_recording.mat';

%% Prepare fake recording data

fs = 100; % sampling frequency (Hz)
sample_nb = 60 * fs;
channel_nb = 10;

data = rand(sample_nb, channel_nb);

%% SOLUTION 1
% if the full data can be loaded in MATLAB

createRecordingMatfile(output_path, data, fs)

%% SOLUTION 2
% if the full data cannot be loaded in MATLAB, it is possible to initialize
% the RecordingMatfile and to fill it chunk by chunk

init_value = 0;
matfile_handle = createRecordingMatfile(output_path, init_value, fs,...
    'init_dims', [sample_nb, channel_nb]);

chunk_size = 10;
s1 = 1; % first sample of the chunk

while s1 <= sample_nb
    s2 = min(sample_nb, s1 + chunk_size - 1); % last sample of the chunk
    
    chunk = data(s1:s2, :); % here: read a chunk in the source file
    
    matfile_handle.values(s1:s2, :) = chunk;
    
    s1 = s2 + 1;
    
end
