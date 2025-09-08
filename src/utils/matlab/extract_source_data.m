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

% Extract and concatenate raw data across all probes
fprintf('\n=== Extracting and Concatenating Raw Data ===\n');
extracted_data = extract_and_concatenate_raw_data(cells);

% Display summary
display_extraction_summary(extracted_data);

% Save if output file specified
if nargin > 1 && ~isempty(output_file)
    fprintf('\nSaving extracted data to: %s\n', output_file);
    save(output_file, '-struct', 'extracted_data');
    fprintf('Extraction completed successfully!\n');
end

end

function extracted_data = extract_and_concatenate_raw_data(cells)
% Extract and concatenate all raw fields across probes without processing

extracted_data = struct();

% Initialize concatenation arrays
all_spike_times = {};
all_hemispheres = {};
all_regions = {};
all_raw_quality_metrics = {};
all_coordinates = [];
all_other_unit_fields = containers.Map();

% Process each probe
num_probes = size(cells, 2);
fprintf('Processing %d probes...\n', num_probes);

for probe_idx = 1:num_probes
    fprintf('  Probe %d: ', probe_idx);
    
    cell_data = cells(1, probe_idx);
    
    if isempty(cell_data)
        fprintf('empty\n');
        continue;
    end
    
    % Navigate to the cell structure
    try
        if iscell(cell_data) && ~isempty(cell_data)
            cell_struct = cell_data{1,1};
        else
            cell_struct = cell_data(1,1);
        end
    catch
        fprintf('cannot access structure\n');
        continue;
    end
    
    % Count units and concatenate spike times
    n_units = 0;
    if isfield(cell_struct, 'raw_spike_time_s')
        spike_times = cell_struct.raw_spike_time_s;
        for i = 1:numel(spike_times)
            if ~isempty(spike_times{i})
                n_units = n_units + 1;
                all_spike_times{end+1} = spike_times{i}(:);
            end
        end
    end
    
    fprintf('%d units\n', n_units);
    
    % Concatenate metadata fields for each unit
    unit_level_fields = {'hemisphere', 'region', 'electrode', 'quality_metrics', ...
                        'waveform', 'distance_from_tip', 'included_units', ...
                        'frac_spikes_removed'};
    
    for field_name = unit_level_fields
        field = field_name{1};
        if isfield(cell_struct, field)
            field_data = cell_struct.(field);
            
            % Store raw field data (replicated for each unit if needed)
            if strcmp(field, 'hemisphere') || strcmp(field, 'region')
                % String fields - extract and replicate
                str_val = extract_mcos_string(field_data, field);
                for u = 1:n_units
                    if strcmp(field, 'hemisphere')
                        all_hemispheres{end+1} = str_val;
                    elseif strcmp(field, 'region')
                        all_regions{end+1} = str_val;
                    end
                end
            else
                % Store raw data for Python processing
                if ~all_other_unit_fields.isKey(field)
                    all_other_unit_fields(field) = {};
                end
                unit_data = all_other_unit_fields(field);
                unit_data{end+1} = field_data; % Store probe-level data
                all_other_unit_fields(field) = unit_data;
            end
        end
    end
    
    % Store coordinates (probe-level)
    probe_coords = struct();
    coord_fields = {'AP', 'ML', 'DV'};
    for coord_field = coord_fields
        field = coord_field{1};
        if isfield(cell_struct, field)
            probe_coords.(field) = cell_struct.(field);
        end
    end
    all_coordinates = [all_coordinates; probe_coords];
    
    % Extract session metadata from first probe only
    if probe_idx == 1
        session_fields = {'nTrials', 'removed_trials', 'sessid', 'sess_date', 'rat', ...
                         'bank', 'penetration', 'rec', 'shank', 'probe_serial'};
        
        for field_name = session_fields
            field = field_name{1};
            if isfield(cell_struct, field)
                field_data = cell_struct.(field);
                
                % Handle string fields
                if ismember(field, {'sess_date', 'rat', 'probe_serial'})
                    extracted_data.(field) = extract_mcos_string(field_data, field);
                else
                    extracted_data.(field) = field_data;
                end
            end
        end
        
        % Extract trial data from first probe
        if isfield(cell_struct, 'Trials')
            trials = cell_struct.Trials;
            if ~isempty(trials)
                fprintf('  Extracting trial data...\n');
                extracted_data = extract_raw_trial_data(trials, extracted_data);
            end
        end
    end
end

% Format concatenated neural data
if ~isempty(all_spike_times)
    extracted_data.raw_spike_time_s = all_spike_times';
    extracted_data.hemisphere = all_hemispheres;
    extracted_data.region = all_regions';
    extracted_data.electrode = []; % Empty placeholder
    
    % Store other raw unit-level data for Python processing
    unit_field_keys = keys(all_other_unit_fields);
    for i = 1:length(unit_field_keys)
        field = unit_field_keys{i};
        extracted_data.(field) = all_other_unit_fields(field);
    end
    
    fprintf('  Total concatenated neurons: %d\n', length(all_spike_times));
end

% Store probe coordinates
extracted_data.probe_coordinates = all_coordinates;

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