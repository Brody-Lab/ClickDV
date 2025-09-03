% TEST_EXTRACTION - Test the extract_source_data function
% 
% This script tests the MATLAB data extraction function with an example
% source file to verify the extraction works correctly.

clear; clc;

fprintf('=== Testing MATLAB Data Extraction ===\n\n');

% Define paths to test files
source_file = 'data/raw/A324/2023-07-21/A324_2023_07_21_Cells_source_2.mat';
output_file = 'data/raw/A324/2023-07-21/A324_pycells_20230721_matlab_extracted.mat';

% Check if source file exists
if ~exist(source_file, 'file')
    fprintf('Source file not found: %s\n', source_file);
    fprintf('Looking for available source files...\n');
    
    % Look for any files with 'source' in the name
    source_pattern = 'data/raw/*/*source*.mat';
    source_files = dir(source_pattern);
    
    if ~isempty(source_files)
        fprintf('Found source files:\n');
        for i = 1:length(source_files)
            full_path = fullfile(source_files(i).folder, source_files(i).name);
            fprintf('  %d: %s\n', i, full_path);
        end
        
        % Use the first one found
        source_file = fullfile(source_files(1).folder, source_files(1).name);
        fprintf('\nUsing: %s\n\n', source_file);
    else
        fprintf('No source files found! Please check the data directory.\n');
        return;
    end
end

% Test the extraction
try
    fprintf('Testing extraction function...\n');
    extracted_data = extract_source_data(source_file, output_file);
    
    fprintf('\n=== EXTRACTION TEST COMPLETED ===\n');
    
    % Additional analysis of region/hemisphere data
    if isfield(extracted_data, 'region') && isfield(extracted_data, 'hemisphere')
        fprintf('\n=== REGION/HEMISPHERE ANALYSIS ===\n');
        
        regions = extracted_data.region;
        hemispheres = extracted_data.hemisphere;
        
        if iscell(regions)
            unique_regions = unique(regions(~cellfun(@isempty, regions)));
            fprintf('Unique regions found: %s\n', strjoin(unique_regions, ', '));
        end
        
        if iscell(hemispheres)
            unique_hemispheres = unique(hemispheres(~cellfun(@isempty, hemispheres)));
            fprintf('Unique hemispheres found: %s\n', strjoin(unique_hemispheres, ', '));
        end
    end
    
catch ME
    fprintf('\nERROR during extraction:\n');
    fprintf('Message: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end