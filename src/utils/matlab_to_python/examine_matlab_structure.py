"""
Script to convert A324_pycells_20230721.mat from nested structure to flat dictionary format
matching A324_pycells_20230727.mat structure
"""
import scipy.io as sio
import numpy as np
from typing import Dict, Any

def extract_all_fields_from_nested_structure(source_data: Dict) -> Dict:
    """Extract ALL fields from nested MATLAB structure and flatten to dictionary format"""
    
    cells = source_data['Cells']  # Shape (1, 8) - 8 probes
    extracted = {}
    
    print(f"Processing {cells.shape[1]} probes...")
    
    # Initialize collections for neuron-level data (concatenate across probes)
    neuron_level_collections = {}
    
    # Process each probe
    for probe_idx in range(cells.shape[1]):
        print(f"Processing probe {probe_idx}...")
        cell_data = cells[0, probe_idx]
        
        if cell_data.size == 0:
            print(f"  Probe {probe_idx}: empty")
            continue
            
        # Access the structured array fields
        cell_struct = cell_data[0, 0]
        
        if not hasattr(cell_struct, 'dtype') or not hasattr(cell_struct.dtype, 'names'):
            print(f"  Probe {probe_idx}: no structured fields")
            continue
            
        print(f"  Probe {probe_idx}: {len(cell_struct.dtype.names)} fields available")
        
        # Determine number of units in this probe
        n_units = 0
        if 'raw_spike_time_s' in cell_struct.dtype.names:
            spike_times = cell_struct['raw_spike_time_s']
            if spike_times.size > 0 and spike_times.dtype == 'object':
                n_units = sum(1 for x in spike_times.flat if x.size > 0)
        print(f"  Probe {probe_idx}: {n_units} units with spikes")
        
        # Process every field in this probe's structure
        for field_name in cell_struct.dtype.names:
            field_data = cell_struct[field_name]
            
            if field_data.size == 0:
                continue
                
            # Handle special cases for data that should be concatenated across units
            if field_name == 'raw_spike_time_s' and field_data.dtype == 'object':
                # This is spike times - one per unit
                if field_name not in neuron_level_collections:
                    neuron_level_collections[field_name] = []
                for unit_spikes in field_data.flat:
                    if unit_spikes.size > 0:
                        neuron_level_collections[field_name].append(unit_spikes.reshape(-1, 1))
                        
            elif field_name in ['hemisphere', 'region', 'electrode', 'waveform', 'distance_from_tip', 
                              'penetration', 'probe_serial', 'ML', 'AP', 'DV', 'shank', 'included_units',
                              'quality_metrics', 'waveformSim', 'clusterNotes', 'frac_spikes_removed']:
                # These are neuron metadata - need one entry per unit
                if field_name not in neuron_level_collections:
                    neuron_level_collections[field_name] = []
                
                if field_data.dtype == 'object' and hasattr(field_data, 'flat'):
                    # Array of unit-specific values
                    for unit_data in field_data.flat:
                        neuron_level_collections[field_name].append(unit_data)
                else:
                    # Single value for all units on this probe
                    for _ in range(n_units):
                        neuron_level_collections[field_name].append(field_data)
                        
            elif field_name == 'Trials':
                # Trial data - extract from first probe only
                if probe_idx == 0:
                    print("  Extracting trial data...")
                    extract_trials_data(field_data, extracted)
                    
            else:
                # Session/recording metadata - take from first probe or store probe-specific
                if probe_idx == 0:
                    extracted[field_name] = field_data
    
    # Format neuron-level collections into output arrays
    print("Formatting neuron-level data...")
    for field_name, field_list in neuron_level_collections.items():
        if field_list:
            if field_name == 'raw_spike_time_s':
                extracted[field_name] = np.array(field_list, dtype=object).reshape(-1, 1)
            elif field_name == 'hemisphere':
                extracted[field_name] = np.array(field_list, dtype=object).reshape(1, -1)
            else:
                extracted[field_name] = np.array(field_list, dtype=object).reshape(-1, 1)
            print(f"  {field_name}: {extracted[field_name].shape}")
    
    return extracted

def extract_trials_data(trials_data, extracted):
    """Extract all fields from Trials structure"""
    if trials_data.size == 0:
        return
        
    trial_struct = trials_data[0, 0]
    
    if not hasattr(trial_struct, 'dtype') or not hasattr(trial_struct.dtype, 'names'):
        return
    
    # Extract all trial fields
    for field_name in trial_struct.dtype.names:
        field_data = trial_struct[field_name]
        if field_data.size > 0:
            # Handle nested structures like stateTimes
            if field_name == 'stateTimes':
                extract_state_times(field_data, extracted)
            else:
                extracted[field_name] = field_data

def extract_state_times(state_times_data, extracted):
    """Extract state timing fields and rename appropriately"""
    if state_times_data.size == 0:
        return
        
    state_struct = state_times_data[0, 0] if hasattr(state_times_data, 'flat') else state_times_data
    
    if not hasattr(state_struct, 'dtype') or not hasattr(state_struct.dtype, 'names'):
        return
        
    # Map state time fields to target names
    state_time_mapping = {
        'clicks_on': 'clicks_on',
        'cpoke_in': 'cpoke_in', 
        'cpoke_out': 'cpoke_out',
        'feedback': 'feedback',
        'spoke': 'spoke'
    }
    
    for source_field, target_field in state_time_mapping.items():
        if source_field in state_struct.dtype.names:
            field_data = state_struct[source_field]
            if field_data.size > 0:
                extracted[target_field] = field_data
    
    # Also extract other state times with descriptive names
    for field_name in state_struct.dtype.names:
        if field_name not in state_time_mapping:
            field_data = state_struct[field_name]
            if field_data.size > 0:
                extracted[f'state_{field_name}'] = field_data

def save_converted_data(source_path: str, output_path: str) -> None:
    """Convert and save the data"""
    
    # Load source data
    print(f"Loading source data from {source_path}")
    source_data = sio.loadmat(source_path)
    
    # Extract and convert
    print("Converting nested structure to flat format...")
    converted_data = extract_all_fields_from_nested_structure(source_data)
    
    # Save converted data - use HDF5 format for large files
    print(f"Saving converted data to {output_path}")
    try:
        sio.savemat(output_path, converted_data, oned_as='column')
    except Exception as e:
        print(f"Standard .mat format failed: {e}")
        print("Trying HDF5 format...")
        h5_path = output_path.replace('.mat', '.h5')
        import h5py
        with h5py.File(h5_path, 'w') as f:
            for key, value in converted_data.items():
                if isinstance(value, np.ndarray):
                    if value.dtype == 'object':
                        # Handle object arrays by converting to strings where possible
                        try:
                            str_array = np.array([str(x) for x in value.flat]).reshape(value.shape)
                            f.create_dataset(key, data=str_array)
                        except:
                            print(f"Skipping {key} - cannot convert object array")
                    else:
                        f.create_dataset(key, data=value)
                else:
                    f.create_dataset(key, data=value)
        print(f"Saved as HDF5: {h5_path}")
    
    print("Conversion completed!")
    
    # Print summary
    print("\n=== CONVERTED DATA SUMMARY ===")
    for key, value in converted_data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {type(value)}")

if __name__ == "__main__":
    source_file = '/home/ham/SF/Personal/Education/07 Princeton/Rotations/Brody-Daw/ClickDV/data/raw/A324/2023-07-21/A324_pycells_20230721.mat'
    output_file = '/home/ham/SF/Personal/Education/07 Princeton/Rotations/Brody-Daw/ClickDV/data/raw/A324/2023-07-21/A324_pycells_20230721_flattened.mat'
    
    save_converted_data(source_file, output_file)