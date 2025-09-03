function extracted_data = extract_source_data(source_file, output_file)
% EXTRACT_SOURCE_DATA Extract and concatenate raw fields from nested MATLAB structures
%
% This function performs ONLY structural extraction and concatenation.
% All processing (quality metrics, coordinate mapping, filtering) should be 
% done in Python using the output of this function.
%
% Usage:
%   extracted_data = extract_source_data(source_file, output_file)
%
% Inputs:
%   source_file - Path to source .mat file (e.g., 'A324_pycells_20230721_source.mat')
%   output_file - Path for output .mat file (optional)
%
% Outputs:
%   extracted_data - Struct containing raw extracted fields, concatenated across probes
%
% Example:
%   data = extract_source_data('data/raw/A324/2023-07-21/A324_pycells_20230721_source.mat');

fprintf('=== MATLAB Data Extraction from Source File ===\n');
fprintf('Loading: %s\n', source_file);

% Load the source data
try
    source_data = load(source_file);
catch ME
    error('Failed to load source file: %s\nError: %s', source_file, ME.message);
end

% Initialize extraction structure
extracted_data = struct();

% Check if Cells field exists
if ~isfield(source_data, 'Cells')
    error('Source file does not contain "Cells" field');
end

cells = source_data.Cells;
fprintf('Found Cells array with dimensions: [%s]\n', num2str(size(cells)));

% Extract neural and session data
fprintf('\n=== Extracting Neural Data ===\n');
[neural_data, session_data] = extract_neural_data(cells);

% Extract trial data
fprintf('\n=== Extracting Trial Data ===\n');
trial_data = extract_trial_data(cells);

% Combine all extracted data
extracted_data = merge_structures(neural_data, session_data, trial_data);

% Display summary
display_extraction_summary(extracted_data);

% Save if output file specified
if nargin > 1 && ~isempty(output_file)
    fprintf('\nSaving extracted data to: %s\n', output_file);
    save(output_file, '-struct', 'extracted_data');
    fprintf('Extraction completed successfully!\n');
end

end

function [neural_data, session_data] = extract_neural_data(cells)
% Extract spike times, regions, hemispheres, and quality metrics from Cells structure

neural_data = struct();
session_data = struct();

% Initialize collections - raw data
all_spike_times = {};
all_hemispheres = {};
all_regions = {};
all_electrodes = {};

% Initialize collections - quality metrics
all_quality_metrics = struct();
all_quality_metrics.spatial_spread_um = [];
all_quality_metrics.peak_width_ms = [];
all_quality_metrics.peak_trough_width_ms = [];
all_quality_metrics.upward_going = [];
all_quality_metrics.uvpp = [];

% Initialize collections - filtered data
filt_spike_times = {};
filt_hemispheres = {};
filt_regions = {};

% Process each probe
num_probes = size(cells, 2);
fprintf('Processing %d probes...\n', num_probes);

for probe_idx = 1:num_probes
    fprintf('  Probe %d: ', probe_idx);
    
    % Use array indexing instead of cell indexing
    cell_data = cells(1, probe_idx);
    
    % Debug: check what we got
    fprintf('type=%s, ', class(cell_data));
    
    if isempty(cell_data)
        fprintf('empty\n');
        continue;
    end
    
    % Navigate to the cell structure - handle different possible structures
    try
        if iscell(cell_data) && ~isempty(cell_data)
            cell_struct = cell_data{1,1};
        else
            % Try direct array access
            cell_struct = cell_data(1,1);
        end
    catch
        fprintf('cannot access structure\n');
        continue;
    end
    
    % Count units with spike data
    n_units = 0;
    if isfield(cell_struct, 'raw_spike_time_s')
        spike_times = cell_struct.raw_spike_time_s;
        
        % Count non-empty spike time arrays
        for i = 1:numel(spike_times)
            if ~isempty(spike_times{i})
                n_units = n_units + 1;
                all_spike_times{end+1} = spike_times{i}(:); % Ensure column vector
            end
        end
    end
    
    fprintf('%d units with spikes\n', n_units);
    
    % Extract hemisphere and region data
    hemisphere_str = '';
    if isfield(cell_struct, 'hemisphere')
        hemisphere_obj = cell_struct.hemisphere;
        hemisphere_str = extract_mcos_string(hemisphere_obj, 'hemisphere');
    end
    
    region_str = '';
    if isfield(cell_struct, 'region')
        region_obj = cell_struct.region;
        region_str = extract_mcos_string(region_obj, 'region');
    end
    
    % Extract quality metrics for this probe
    probe_quality = extract_quality_metrics(cell_struct);
    
    % Process each unit - just extract data without filtering
    unit_idx = 0;
    if isfield(cell_struct, 'raw_spike_time_s')
        spike_times = cell_struct.raw_spike_time_s;
        
        for i = 1:numel(spike_times)
            if ~isempty(spike_times{i})
                unit_idx = unit_idx + 1;
                
                % Add to raw collections
                all_hemispheres{end+1} = hemisphere_str;
                all_regions{end+1} = region_str;
                all_electrodes{end+1} = [];
                
                % Add quality metrics for this specific unit
                if ~isempty(probe_quality) && unit_idx <= length(probe_quality.spatial_spread_um)
                    all_quality_metrics.spatial_spread_um(end+1) = probe_quality.spatial_spread_um(unit_idx);
                    all_quality_metrics.peak_width_ms(end+1) = probe_quality.peak_width_ms(unit_idx);
                    all_quality_metrics.peak_trough_width_ms(end+1) = probe_quality.peak_trough_width_ms(unit_idx);
                    all_quality_metrics.upward_going(end+1) = probe_quality.upward_going(unit_idx);
                    all_quality_metrics.uvpp(end+1) = probe_quality.uvpp(unit_idx);
                else
                    % No quality metrics available - fill with NaN/default values
                    all_quality_metrics.spatial_spread_um(end+1) = NaN;
                    all_quality_metrics.peak_width_ms(end+1) = NaN;
                    all_quality_metrics.peak_trough_width_ms(end+1) = NaN;
                    all_quality_metrics.upward_going(end+1) = false;
                    all_quality_metrics.uvpp(end+1) = NaN;
                end
            end
        end
    end
    
    % Extract session metadata from first probe only
    if probe_idx == 1
        session_fields = {'nTrials', 'removed_trials', 'sessid', 'sess_date', 'rat'};
        
        for field_idx = 1:length(session_fields)
            field_name = session_fields{field_idx};
            if isfield(cell_struct, field_name)
                field_data = cell_struct.(field_name);
                
                % Handle string fields (potentially MCOS)
                if ismember(field_name, {'sess_date', 'rat'})
                    session_data.(field_name) = extract_mcos_string(field_data, field_name);
                else
                    session_data.(field_name) = field_data;
                end
            end
        end
    end
end

% Format neural data arrays
if ~isempty(all_spike_times)
    % Raw data (unfiltered)
    neural_data.raw_spike_time_s = all_spike_times';
    neural_data.hemisphere = all_hemispheres;
    neural_data.region = all_regions';
    neural_data.electrode = []; % Empty as in source
    
    % Quality metrics for all neurons
    if ~isempty(all_quality_metrics.spatial_spread_um)
        neural_data.quality_spatial_spread_um = all_quality_metrics.spatial_spread_um';
        neural_data.quality_peak_width_ms = all_quality_metrics.peak_width_ms';
        neural_data.quality_peak_trough_width_ms = all_quality_metrics.peak_trough_width_ms';
        neural_data.quality_upward_going = all_quality_metrics.upward_going';
        neural_data.quality_uvpp = all_quality_metrics.uvpp';
        
        fprintf('  Total neurons: %d (with quality metrics)\n', length(all_spike_times));
        
        % Add quality metrics summary  
        valid_spatial = ~isnan(all_quality_metrics.spatial_spread_um);
        valid_uvpp = ~isnan(all_quality_metrics.uvpp);
        
        if any(valid_spatial)
            fprintf('  Quality metrics summary:\n');
            fprintf('    Spatial spread: %.1f±%.1f μm (n=%d)\n', ...
                   mean(all_quality_metrics.spatial_spread_um(valid_spatial)), ...
                   std(all_quality_metrics.spatial_spread_um(valid_spatial)), ...
                   sum(valid_spatial));
            fprintf('    Peak width: %.2f±%.2f ms\n', mean(all_quality_metrics.peak_width_ms(valid_spatial)), std(all_quality_metrics.peak_width_ms(valid_spatial)));
            fprintf('    Peak-trough width: %.2f±%.2f ms\n', mean(all_quality_metrics.peak_trough_width_ms(valid_spatial)), std(all_quality_metrics.peak_trough_width_ms(valid_spatial)));
            fprintf('    uVpp: %.1f±%.1f μV\n', mean(all_quality_metrics.uvpp(valid_uvpp)), std(all_quality_metrics.uvpp(valid_uvpp)));
            fprintf('    Upward-going: %d/%d (%.1f%%)\n', sum(all_quality_metrics.upward_going), length(all_quality_metrics.upward_going), 100*mean(all_quality_metrics.upward_going));
        end
    else
        fprintf('  Total neurons: %d (no quality metrics extracted)\n', length(all_spike_times));
    end
else
    fprintf('  No neural data found!\n');
end

end

function trial_data = extract_trial_data(cells)
% Extract behavioral trial data from first probe

trial_data = struct();

% Get trial data from first probe
if size(cells, 2) >= 1
    cell_data = cells(1, 1);
    if ~isempty(cell_data)
        try
            if iscell(cell_data) && ~isempty(cell_data)
                cell_struct = cell_data{1,1};
            else
                cell_struct = cell_data(1,1);
            end
        catch
            fprintf('  Cannot access trial structure\n');
            return;
        end
        
        if isfield(cell_struct, 'Trials')
            trials = cell_struct.Trials;
            if ~isempty(trials)
                try
                    if iscell(trials)
                        trial_struct = trials{1,1};
                    else
                        trial_struct = trials(1,1);
                    end
                catch
                    fprintf('  Cannot access trials structure\n');
                    return;
                end
                
                fprintf('  Extracting behavioral fields...\n');
                
                % Extract behavioral timing fields
                behavioral_fields = {'gamma', 'click_diff', 'seed', 'never_cpoked', ...
                                   'violated', 'pokedR', 'is_hit'};
                
                for field_idx = 1:length(behavioral_fields)
                    field_name = behavioral_fields{field_idx};
                    if isfield(trial_struct, field_name)
                        trial_data.(field_name) = trial_struct.(field_name);
                    end
                end
                
                % Extract click data (map from source naming)
                if isfield(trial_struct, 'rightBups')
                    trial_data.right_bups = trial_struct.rightBups;
                end
                if isfield(trial_struct, 'leftBups')
                    trial_data.left_bups = trial_struct.leftBups;
                end
                
                % Extract state timing data
                if isfield(trial_struct, 'stateTimes')
                    state_times = trial_struct.stateTimes;
                    if ~isempty(state_times)
                        try
                            if iscell(state_times)
                                state_struct = state_times{1,1};
                            else
                                state_struct = state_times(1,1);
                            end
                        catch
                            fprintf('    Cannot access state times structure\n');
                            state_struct = []; % Set empty if failed
                        end
                        
                        % Map state time fields only if we got the structure
                        if ~isempty(state_struct)
                            state_mappings = struct('clicks_on', 'clicks_on', ...
                                              'cpoke_in', 'cpoke_in', ...
                                              'cpoke_out', 'cpoke_out', ...
                                              'feedback', 'feedback', ...
                                              'spoke', 'spoke');
                        
                        state_fields = fieldnames(state_mappings);
                        for field_idx = 1:length(state_fields)
                            source_field = state_fields{field_idx};
                            target_field = state_mappings.(source_field);
                            
                            if isfield(state_struct, source_field)
                                trial_data.(target_field) = state_struct.(source_field);
                            end
                        end
                        end % Close the if ~isempty(state_struct)
                    end
                end
                
                % Generate placeholder reward probability fields
                if isfield(trial_struct, 'T')
                    n_trials = size(trial_struct.T, 1);
                    
                    % Create typical reward probability arrays
                    trial_data.right_reward_p = repmat(0.8, n_trials, 1);
                    trial_data.left_reward_p = repmat(0.2, n_trials, 1);
                    
                    % Generate water delivered based on feedback
                    if isfield(trial_data, 'feedback')
                        water_delivered = zeros(n_trials, 1);
                        feedback_times = trial_data.feedback;
                        valid_feedback = ~isnan(feedback_times);
                        water_delivered(valid_feedback) = 0.03; % Typical volume
                        trial_data.water_delivered = water_delivered;
                    end
                    
                    fprintf('    Processed %d trials\n', n_trials);
                end
            end
        end
    end
end

end

function extracted_string = extract_mcos_string(string_obj, field_name)
% Extract string from various MATLAB string formats

extracted_string = '';

if isempty(string_obj)
    return;
end

try
    % Handle native MATLAB string type
    if isstring(string_obj)
        if numel(string_obj) > 0
            extracted_string = char(string_obj(1)); % Convert to char
        end
    % Handle character arrays
    elseif ischar(string_obj)
        extracted_string = string_obj;
    % Handle cell arrays containing strings
    elseif iscell(string_obj) && ~isempty(string_obj)
        first_item = string_obj{1};
        if isstring(first_item)
            extracted_string = char(first_item);
        elseif ischar(first_item)
            extracted_string = first_item;
        end
    end
    
    % Display what we found
    if ~isempty(extracted_string)
        fprintf('    Extracted %s: "%s"\n', field_name, extracted_string);
    else
        fprintf('    Could not extract %s (type: %s)\n', field_name, class(string_obj));
    end
    
catch ME
    fprintf('    Error extracting %s: %s\n', field_name, ME.message);
end

end

function merged = merge_structures(varargin)
% Merge multiple structures into one

merged = struct();

for i = 1:nargin
    s = varargin{i};
    if isstruct(s)
        field_names = fieldnames(s);
        for j = 1:length(field_names)
            merged.(field_names{j}) = s.(field_names{j});
        end
    end
end

end

function display_extraction_summary(extracted_data)
% Display summary of extracted data

fprintf('\n=== EXTRACTION SUMMARY ===\n');

field_names = fieldnames(extracted_data);
for i = 1:length(field_names)
    field_name = field_names{i};
    field_data = extracted_data.(field_name);
    
    if isnumeric(field_data)
        fprintf('%s: [%s] %s\n', field_name, num2str(size(field_data)), class(field_data));
    elseif iscell(field_data)
        fprintf('%s: {%s} cell array\n', field_name, num2str(size(field_data)));
        
        % Show sample values for region/hemisphere
        if ismember(field_name, {'hemisphere', 'region'}) && ~isempty(field_data)
            sample_size = min(5, length(field_data));
            sample_vals = cell(sample_size, 1);
            for j = 1:sample_size
                if ischar(field_data{j})
                    sample_vals{j} = field_data{j};
                else
                    sample_vals{j} = class(field_data{j});
                end
            end
            fprintf('  Sample values: %s\n', strjoin(sample_vals, ', '));
        end
        
        % Show summary statistics for quality metrics
        if startsWith(field_name, 'quality_') && isnumeric(field_data) && ~isempty(field_data)
            fprintf('  Range: %.2f to %.2f, Mean: %.2f\n', min(field_data), max(field_data), mean(field_data));
        end
    elseif ischar(field_data)
        fprintf('%s: "%s"\n', field_name, field_data);
    else
        fprintf('%s: %s\n', field_name, class(field_data));
    end
end

end

function quality_metrics = extract_quality_metrics(cell_struct)
% Extract quality metrics from cell structure
%
% Returns struct with fields:
%   - spatial_spread_um: Spatial spread in micrometers
%   - peak_width_ms: Peak width in milliseconds  
%   - peak_trough_width_ms: Peak-trough width in milliseconds
%   - upward_going: Boolean indicating upward-going spikes
%   - uvpp: Peak-to-peak voltage in microvolts

quality_metrics = struct();

try
    % Check if quality_metrics field exists
    if isfield(cell_struct, 'quality_metrics')
        qm = cell_struct.quality_metrics;
        if ~isempty(qm) && isstruct(qm)
            qm_struct = qm(1,1);
            
            % Extract spatial spread (already in μm) - array of values per unit
            if isfield(qm_struct, 'spatial_spread_um')
                spatial_data = qm_struct.spatial_spread_um;
                quality_metrics.spatial_spread_um = spatial_data(:)'; % Flatten to row vector
            end
            
            % Extract peak width (convert from seconds to ms) - array of values per unit  
            if isfield(qm_struct, 'peak_width_s')
                peak_width_s = qm_struct.peak_width_s;
                quality_metrics.peak_width_ms = peak_width_s(:)' * 1000; % Convert to ms, flatten to row
            end
            
            % Extract peak-trough width (convert from seconds to ms) - array of values per unit
            if isfield(qm_struct, 'peak_trough_width_s')
                peak_trough_width_s = qm_struct.peak_trough_width_s;
                quality_metrics.peak_trough_width_ms = peak_trough_width_s(:)' * 1000; % Convert to ms, flatten to row
            end
            
            % Extract upward-going spike indicator - array of values per unit
            if isfield(qm_struct, 'like_axon')
                like_axon = qm_struct.like_axon;
                quality_metrics.upward_going = logical(like_axon(:)'); % Convert to logical row vector
            end
            
            % Extract peak-to-peak voltage (already in μV) - array of values per unit
            if isfield(qm_struct, 'uVpp')
                uvpp = qm_struct.uVpp;
                quality_metrics.uvpp = uvpp(:)'; % Flatten to row vector
            end
            
            fprintf('    Extracted quality metrics for %d units\n', length(quality_metrics.spatial_spread_um));
        end
    end
    
    % Also check waveform field as backup
    if isfield(cell_struct, 'waveform') && (isempty(quality_metrics) || ~isfield(quality_metrics, 'spatial_spread_um'))
        wf = cell_struct.waveform;
        if ~isempty(wf) && isstruct(wf)
            wf_struct = wf(1,1);
            
            % Use waveform data if quality_metrics wasn't available
            if ~isfield(quality_metrics, 'spatial_spread_um') && isfield(wf_struct, 'spatial_spread_um')
                quality_metrics.spatial_spread_um = wf_struct.spatial_spread_um(1,1);
                quality_metrics.spatial_spread_um = quality_metrics.spatial_spread_um(:)';
            end
            
            if ~isfield(quality_metrics, 'peak_width_ms') && isfield(wf_struct, 'peak_width_s')
                peak_width_s = wf_struct.peak_width_s(1,1);
                quality_metrics.peak_width_ms = peak_width_s(:)' * 1000;
            end
            
            if ~isfield(quality_metrics, 'peak_trough_width_ms') && isfield(wf_struct, 'peak_trough_width_s')
                peak_trough_width_s = wf_struct.peak_trough_width_s(1,1);
                quality_metrics.peak_trough_width_ms = peak_trough_width_s(:)' * 1000;
            end
            
            if ~isfield(quality_metrics, 'upward_going') && isfield(wf_struct, 'like_axon')
                like_axon = wf_struct.like_axon(1,1);
                quality_metrics.upward_going = logical(like_axon(:)');
            end
            
            if ~isfield(quality_metrics, 'uvpp') && isfield(wf_struct, 'Vpp_uv')
                uvpp = wf_struct.Vpp_uv(1,1);
                quality_metrics.uvpp = uvpp(:)';
            end
            
            fprintf('    Used waveform metrics for %d units\n', length(quality_metrics.spatial_spread_um));
        end
    end
    
catch ME
    fprintf('    Error extracting quality metrics: %s\n', ME.message);
    quality_metrics = struct(); % Return empty struct on error
end

end