"""
Extract and list every field found in the source MATLAB file
"""
import scipy.io as sio
import numpy as np

def collect_all_fields(source_data):
    """Collect every field name and basic info from the nested structure"""
    
    cells = source_data['Cells']
    all_fields = {}
    
    print(f"Analyzing {cells.shape[1]} probes...")
    
    # Process each probe to collect field information
    for probe_idx in range(cells.shape[1]):
        cell_data = cells[0, probe_idx]
        
        if cell_data.size == 0:
            continue
            
        cell_struct = cell_data[0, 0]
        
        if not hasattr(cell_struct, 'dtype') or not hasattr(cell_struct.dtype, 'names'):
            continue
        
        # Count units in this probe
        n_units = 0
        if 'raw_spike_time_s' in cell_struct.dtype.names:
            spike_times = cell_struct['raw_spike_time_s']
            if spike_times.size > 0 and spike_times.dtype == 'object':
                n_units = sum(1 for x in spike_times.flat if x.size > 0)
        
        # Process each field
        for field_name in cell_struct.dtype.names:
            field_data = cell_struct[field_name]
            
            if field_name not in all_fields:
                all_fields[field_name] = {
                    'found_in_probes': [],
                    'shapes': [],
                    'dtypes': [],
                    'has_data': [],
                    'sample_content': None,
                    'is_neuron_level': False,
                    'is_trial_level': False,
                    'is_session_level': False
                }
            
            info = all_fields[field_name]
            info['found_in_probes'].append(probe_idx)
            info['shapes'].append(field_data.shape if hasattr(field_data, 'shape') else 'no_shape')
            info['dtypes'].append(str(field_data.dtype) if hasattr(field_data, 'dtype') else str(type(field_data)))
            info['has_data'].append(field_data.size > 0 if hasattr(field_data, 'size') else True)
            
            # Determine field category
            if field_name == 'raw_spike_time_s' or (field_data.size > 0 and field_data.dtype == 'object' and n_units > 0):
                info['is_neuron_level'] = True
            elif field_name == 'Trials':
                info['is_trial_level'] = True
            else:
                info['is_session_level'] = True
            
            # Get sample content (first time we see this field with data)
            if info['sample_content'] is None and field_data.size > 0:
                try:
                    if field_data.dtype == 'object':
                        sample = str(field_data.flat[0])[:100] if field_data.size > 0 else "empty"
                    else:
                        sample = str(field_data).replace('\n', ' ')[:100]
                    info['sample_content'] = sample
                except:
                    info['sample_content'] = "could_not_extract"
    
    # Also extract trial subfields
    trial_fields = {}
    if 'Trials' in all_fields:
        # Get trials data from first probe
        for probe_idx in range(cells.shape[1]):
            cell_data = cells[0, probe_idx]
            if cell_data.size == 0:
                continue
            cell_struct = cell_data[0, 0]
            if 'Trials' in cell_struct.dtype.names:
                trials_data = cell_struct['Trials']
                if trials_data.size > 0:
                    trial_struct = trials_data[0, 0]
                    if hasattr(trial_struct, 'dtype') and hasattr(trial_struct.dtype, 'names'):
                        for trial_field in trial_struct.dtype.names:
                            trial_field_data = trial_struct[trial_field]
                            trial_fields[f'Trials.{trial_field}'] = {
                                'shape': trial_field_data.shape if hasattr(trial_field_data, 'shape') else 'no_shape',
                                'dtype': str(trial_field_data.dtype) if hasattr(trial_field_data, 'dtype') else str(type(trial_field_data)),
                                'has_data': trial_field_data.size > 0 if hasattr(trial_field_data, 'size') else True,
                                'sample_content': str(trial_field_data).replace('\n', ' ')[:100] if hasattr(trial_field_data, 'size') and trial_field_data.size > 0 else "empty"
                            }
                            
                            # Check for nested fields like stateTimes
                            if trial_field == 'stateTimes' and trial_field_data.size > 0:
                                try:
                                    state_struct = trial_field_data[0, 0]
                                    if hasattr(state_struct, 'dtype') and hasattr(state_struct.dtype, 'names'):
                                        for state_field in state_struct.dtype.names:
                                            state_field_data = state_struct[state_field]
                                            trial_fields[f'Trials.stateTimes.{state_field}'] = {
                                                'shape': state_field_data.shape if hasattr(state_field_data, 'shape') else 'no_shape',
                                                'dtype': str(state_field_data.dtype) if hasattr(state_field_data, 'dtype') else str(type(state_field_data)),
                                                'has_data': state_field_data.size > 0 if hasattr(state_field_data, 'size') else True,
                                                'sample_content': str(state_field_data).replace('\n', ' ')[:100] if hasattr(state_field_data, 'size') and state_field_data.size > 0 else "empty"
                                            }
                                except:
                                    pass
                break
    
    return all_fields, trial_fields

# Load and analyze
source_data = sio.loadmat('/home/ham/SF/Personal/Education/07 Princeton/Rotations/Brody-Daw/ClickDV/data/raw/A324/2023-07-21/A324_pycells_20230721.mat')
main_fields, trial_fields = collect_all_fields(source_data)

# Write comprehensive field list
output_file = '/home/ham/SF/Personal/Education/07 Princeton/Rotations/Brody-Daw/ClickDV/data/raw/A324/2023-07-21/all_available_fields.txt'

with open(output_file, 'w') as f:
    f.write("=== ALL AVAILABLE FIELDS IN A324_pycells_20230721.mat ===\n\n")
    f.write(f"Total main fields: {len(main_fields)}\n")
    f.write(f"Total trial subfields: {len(trial_fields)}\n\n")
    
    f.write("=== MAIN PROBE FIELDS ===\n")
    for field_name, info in sorted(main_fields.items()):
        f.write(f"\nFIELD: {field_name}\n")
        f.write(f"  Found in probes: {list(set(info['found_in_probes']))}\n")
        f.write(f"  Shapes: {list(set(str(s) for s in info['shapes']))}\n")
        f.write(f"  Data types: {list(set(info['dtypes']))}\n")
        f.write(f"  Has data: {any(info['has_data'])}\n")
        f.write(f"  Category: {'NEURON' if info['is_neuron_level'] else 'TRIAL' if info['is_trial_level'] else 'SESSION'}\n")
        f.write(f"  Sample: {info['sample_content']}\n")
    
    f.write(f"\n\n=== TRIAL SUBFIELDS ===\n")
    for field_name, info in sorted(trial_fields.items()):
        f.write(f"\nFIELD: {field_name}\n")
        f.write(f"  Shape: {info['shape']}\n")
        f.write(f"  Data type: {info['dtype']}\n")
        f.write(f"  Has data: {info['has_data']}\n")
        f.write(f"  Sample: {info['sample_content']}\n")

print(f"Field analysis complete! Results saved to: {output_file}")
print(f"Found {len(main_fields)} main fields and {len(trial_fields)} trial subfields")