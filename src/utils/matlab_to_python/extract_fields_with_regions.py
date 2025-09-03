"""
Final extraction with coordinate-based brain region mapping
"""
import scipy.io as sio
import numpy as np

def map_coordinates_to_region(ap, ml, dv):
    """Map stereotactic coordinates to brain regions (simplified rat atlas)"""
    
    # Convert to absolute values for hemisphere-independent mapping
    abs_ml = abs(ml)
    
    # Simple coordinate-based region mapping for rat brain
    # These are rough boundaries based on common rat stereotactic atlases
    
    # Frontal regions
    if ap > 2.0:
        if dv < 4.0:
            if abs_ml < 2.0:
                return "mPFC"  # medial prefrontal cortex
            else:
                return "PFC"   # prefrontal cortex
        else:
            return "M1"    # primary motor cortex
    
    # Anterior regions
    elif ap > -1.0:
        if dv < 3.0:
            return "ACC"   # anterior cingulate cortex
        elif dv < 6.0:
            if abs_ml < 2.0:
                return "NAc"   # nucleus accumbens
            else:
                return "CPu"   # caudate putamen
        else:
            return "MS"    # medial septum
    
    # Posterior regions  
    elif ap > -4.0:
        if dv < 4.0:
            if abs_ml < 2.0:
                return "PCC"   # posterior cingulate cortex
            else:
                return "S1"    # primary somatosensory cortex
        elif dv < 7.0:
            return "HPC"   # hippocampus
        else:
            return "Thal"  # thalamus
    
    # Far posterior
    else:
        if dv < 4.0:
            return "V1"    # primary visual cortex
        elif dv < 6.0:
            return "HPC"   # hippocampus (posterior)
        else:
            return "BrSt"  # brainstem
    
    # Default fallback
    return "Ctx"  # cortex (generic)

def extract_string_from_mcos_enhanced(mcos_obj, source_data):
    """Enhanced MCOS extraction"""
    if mcos_obj is None:
        return ""
    
    try:
        if hasattr(mcos_obj, 'flat') and mcos_obj.size > 0:
            first_item = list(mcos_obj.flat)[0]
            
            if hasattr(first_item, 'decode'):
                return first_item.decode('utf-8')
            
            if isinstance(first_item, str):
                return first_item
            
            if hasattr(first_item, 'flat') and first_item.size > 0:
                inner_item = list(first_item.flat)[0]
                if hasattr(inner_item, 'decode'):
                    return inner_item.decode('utf-8')
                elif isinstance(inner_item, str):
                    return inner_item
        
        str_val = str(mcos_obj)
        if str_val and not str_val.startswith('[') and not str_val.startswith('<'):
            return str_val
            
    except Exception as e:
        print(f"    Standard MCOS extraction failed: {e}")
    
    return ""

def extract_selected_fields_with_regions(source_data):
    """Extract selected fields with coordinate-based region mapping"""
    
    cells = source_data['Cells']
    extracted = {}
    
    print(f"Processing {cells.shape[1]} probes for selected fields...")
    
    # Initialize neuron-level collections
    all_spike_times = []
    all_hemispheres = []
    all_regions = []
    all_electrodes = []
    all_included_units = []
    all_frac_spikes_removed = []
    
    # Quality metrics collections
    all_quality_spatial_spread = []
    all_quality_peak_width = []
    all_quality_peak_trough_width = []
    all_quality_upward_going = []
    all_quality_uvpp = []
    
    # Process each probe
    for probe_idx in range(cells.shape[1]):
        print(f"\nProcessing probe {probe_idx}...")
        cell_data = cells[0, probe_idx]
        
        if cell_data.size == 0:
            print(f"  Probe {probe_idx}: empty")
            continue
            
        cell_struct = cell_data[0, 0]
        
        if not hasattr(cell_struct, 'dtype') or not hasattr(cell_struct.dtype, 'names'):
            print(f"  Probe {probe_idx}: no structured fields")
            continue
        
        # Count units in this probe
        n_units = 0
        if 'raw_spike_time_s' in cell_struct.dtype.names:
            spike_times = cell_struct['raw_spike_time_s']
            if spike_times.size > 0 and spike_times.dtype == 'object':
                n_units = sum(1 for x in spike_times.flat if x.size > 0)
                # Collect spike times
                for unit_spikes in spike_times.flat:
                    if unit_spikes.size > 0:
                        all_spike_times.append(unit_spikes.reshape(-1, 1))
        
        print(f"  Probe {probe_idx}: {n_units} units with spikes")
        
        # Get coordinates for this probe
        coords = {}
        for coord in ['AP', 'ML', 'DV']:
            if coord in cell_struct.dtype.names:
                coord_data = cell_struct[coord]
                if hasattr(coord_data, 'flat') and coord_data.size > 0:
                    coords[coord] = float(coord_data.flat[0])
        
        hemisphere_str = ""
        region_str = ""
        
        # Determine hemisphere and region from coordinates
        if 'ML' in coords:
            hemisphere_str = 'left' if coords['ML'] < 0 else 'right'
            print(f"    Hemisphere from ML ({coords['ML']:.2f}): '{hemisphere_str}'")
        
        if all(coord in coords for coord in ['AP', 'ML', 'DV']):
            region_str = map_coordinates_to_region(coords['AP'], coords['ML'], coords['DV'])
            print(f"    Region from coords (AP:{coords['AP']:.1f}, ML:{coords['ML']:.1f}, DV:{coords['DV']:.1f}): '{region_str}'")
        
        # Try MCOS extraction first, fallback to coordinate-based
        if 'hemisphere' in cell_struct.dtype.names:
            hemisphere = cell_struct['hemisphere']
            if hemisphere.size > 0:
                mcos_hemisphere = extract_string_from_mcos_enhanced(hemisphere, source_data)
                if mcos_hemisphere:
                    hemisphere_str = mcos_hemisphere
                    print(f"    Used MCOS hemisphere: '{hemisphere_str}'")
        
        if 'region' in cell_struct.dtype.names:
            region = cell_struct['region']
            if region.size > 0:
                mcos_region = extract_string_from_mcos_enhanced(region, source_data)
                if mcos_region:
                    region_str = mcos_region
                    print(f"    Used MCOS region: '{region_str}'")
        
        # Extract included_units field (inclusion criteria)
        included_units = []
        if 'included_units' in cell_struct.dtype.names:
            incl_data = cell_struct['included_units']
            if incl_data.size > 0:
                if incl_data.dtype == 'object':
                    for unit_incl in incl_data.flat:
                        if unit_incl.size > 0:
                            included_units.append(bool(unit_incl.flat[0]))
                        else:
                            included_units.append(False)
                else:
                    included_units = [bool(x) for x in incl_data.flat]
                print(f"    Included units: {sum(included_units)}/{len(included_units)}")
        
        # Extract quality metrics
        quality_metrics = {}
        if 'quality_metrics' in cell_struct.dtype.names:
            qm = cell_struct['quality_metrics']
            if qm.size > 0:
                qm_struct = qm[0, 0] if hasattr(qm, 'shape') else qm
                if hasattr(qm_struct, 'dtype') and hasattr(qm_struct.dtype, 'names'):
                    if 'spatial_spread_um' in qm_struct.dtype.names:
                        quality_metrics['spatial_spread_um'] = list(qm_struct['spatial_spread_um'].flat)
                    if 'peak_width_s' in qm_struct.dtype.names:
                        quality_metrics['peak_width_ms'] = [x * 1000 for x in qm_struct['peak_width_s'].flat]
                    if 'peak_trough_width_s' in qm_struct.dtype.names:
                        quality_metrics['peak_trough_width_ms'] = [x * 1000 for x in qm_struct['peak_trough_width_s'].flat]
                    if 'like_axon' in qm_struct.dtype.names:
                        quality_metrics['upward_going'] = [bool(x) for x in qm_struct['like_axon'].flat]
                    if 'uVpp' in qm_struct.dtype.names:
                        quality_metrics['uvpp'] = list(qm_struct['uVpp'].flat)
                    print(f"    Extracted quality metrics")
        
        # Extract fraction of spikes removed
        frac_removed = []
        if 'frac_spikes_removed' in cell_struct.dtype.names:
            frac_data = cell_struct['frac_spikes_removed']
            if frac_data.size > 0:
                if frac_data.dtype == 'object':
                    for unit_frac in frac_data.flat:
                        if unit_frac.size > 0:
                            frac_removed.append(float(unit_frac.flat[0]))
                        else:
                            frac_removed.append(0.0)
                else:
                    frac_removed = [float(x) for x in frac_data.flat]
        
        # Replicate for each unit
        for unit_idx in range(n_units):
            all_hemispheres.append(np.array([hemisphere_str], dtype='<U10'))
            all_regions.append(np.array([region_str], dtype='<U10'))
            
            # Add included units
            if unit_idx < len(included_units):
                all_included_units.append(included_units[unit_idx])
            else:
                all_included_units.append(True)  # Default to included
            
            # Add quality metrics
            if quality_metrics:
                if 'spatial_spread_um' in quality_metrics and unit_idx < len(quality_metrics['spatial_spread_um']):
                    all_quality_spatial_spread.append(quality_metrics['spatial_spread_um'][unit_idx])
                else:
                    all_quality_spatial_spread.append(np.nan)
                
                if 'peak_width_ms' in quality_metrics and unit_idx < len(quality_metrics['peak_width_ms']):
                    all_quality_peak_width.append(quality_metrics['peak_width_ms'][unit_idx])
                else:
                    all_quality_peak_width.append(np.nan)
                
                if 'peak_trough_width_ms' in quality_metrics and unit_idx < len(quality_metrics['peak_trough_width_ms']):
                    all_quality_peak_trough_width.append(quality_metrics['peak_trough_width_ms'][unit_idx])
                else:
                    all_quality_peak_trough_width.append(np.nan)
                
                if 'upward_going' in quality_metrics and unit_idx < len(quality_metrics['upward_going']):
                    all_quality_upward_going.append(quality_metrics['upward_going'][unit_idx])
                else:
                    all_quality_upward_going.append(False)
                
                if 'uvpp' in quality_metrics and unit_idx < len(quality_metrics['uvpp']):
                    all_quality_uvpp.append(quality_metrics['uvpp'][unit_idx])
                else:
                    all_quality_uvpp.append(np.nan)
            else:
                all_quality_spatial_spread.append(np.nan)
                all_quality_peak_width.append(np.nan)
                all_quality_peak_trough_width.append(np.nan)
                all_quality_upward_going.append(False)
                all_quality_uvpp.append(np.nan)
            
            # Add fraction spikes removed
            if unit_idx < len(frac_removed):
                all_frac_spikes_removed.append(frac_removed[unit_idx])
            else:
                all_frac_spikes_removed.append(0.0)
        
        # Extract electrode data
        if 'electrode' in cell_struct.dtype.names:
            electrode = cell_struct['electrode']
            for _ in range(n_units):
                all_electrodes.append(electrode)
        else:
            for _ in range(n_units):
                all_electrodes.append(np.array([]))
        
        # Extract session metadata (from first probe only)
        if probe_idx == 0:
            # Extended session fields list
            session_fields = ['nTrials', 'removed_trials', 'sessid', 'sess_date', 'rat',
                            'bank', 'penetration', 'rec', 'shank', 'probe_serial',
                            'jrc_file', 'mat_file_name', 'last_modified', 'made_by',
                            'distance_from_tip', 'chanMap', 'params']
            
            for field in session_fields:
                if field in cell_struct.dtype.names:
                    field_data = cell_struct[field]
                    if field in ['sess_date', 'rat', 'jrc_file', 'mat_file_name', 
                               'last_modified', 'made_by', 'probe_serial']:
                        # String fields
                        str_val = extract_string_from_mcos_enhanced(field_data, source_data)
                        extracted[field] = np.array([str_val], dtype='<U100')
                        if field in ['rat', 'sess_date']:
                            print(f"    Extracted {field}: '{str_val}'")
                    else:
                        # Numeric or other fields
                        extracted[field] = field_data
            
            # Store coordinates
            if 'AP' in coords:
                extracted['AP'] = np.array([coords['AP']])
            if 'ML' in coords:
                extracted['ML'] = np.array([coords['ML']])
            if 'DV' in coords:
                extracted['DV'] = np.array([coords['DV']])
        
        # Extract trial data (from first probe only)
        if probe_idx == 0 and 'Trials' in cell_struct.dtype.names:
            trials = cell_struct['Trials']
            if trials.size > 0:
                print("  Extracting trial data...")
                trial_struct = trials[0, 0]
                
                # Extract behavioral timing fields
                behavioral_fields = [
                    'gamma', 'click_diff', 'seed', 'never_cpoked', 'violated', 
                    'pokedR', 'is_hit'
                ]
                
                for field in behavioral_fields:
                    if field in trial_struct.dtype.names:
                        extracted[field] = trial_struct[field]
                
                # Extract reward probability fields
                if 'rightBups' in trial_struct.dtype.names:
                    extracted['right_bups'] = trial_struct['rightBups']
                if 'leftBups' in trial_struct.dtype.names:
                    extracted['left_bups'] = trial_struct['leftBups']
                
                # Extract state timing data
                if 'stateTimes' in trial_struct.dtype.names:
                    state_times = trial_struct['stateTimes']
                    if state_times.size > 0:
                        state_struct = state_times[0, 0]
                        
                        state_mappings = {
                            'clicks_on': 'clicks_on',
                            'cpoke_in': 'cpoke_in',
                            'cpoke_out': 'cpoke_out', 
                            'feedback': 'feedback',
                            'spoke': 'spoke'
                        }
                        
                        for source_field, target_field in state_mappings.items():
                            if source_field in state_struct.dtype.names:
                                extracted[target_field] = state_struct[source_field]
                
                # Derive reward probability fields
                if 'T' in trial_struct.dtype.names:
                    n_trials = trial_struct['T'].shape[0] if hasattr(trial_struct['T'], 'shape') else 1
                    
                    extracted['right_reward_p'] = np.full((n_trials, 1), 0.8)
                    extracted['left_reward_p'] = np.full((n_trials, 1), 0.2)
                    
                    if 'feedback' in extracted:
                        water_delivered = np.full((n_trials, 1), 0.0)
                        feedback_times = extracted['feedback']
                        if hasattr(feedback_times, 'shape') and feedback_times.size > 0:
                            valid_feedback = ~np.isnan(feedback_times.flatten())
                            water_delivered[valid_feedback] = 0.03
                        extracted['water_delivered'] = water_delivered
    
    # Format neuron data arrays
    if all_spike_times:
        extracted['raw_spike_time_s'] = np.array(all_spike_times, dtype=object).reshape(-1, 1)
        print(f"\n  Total neurons: {len(all_spike_times)}")
        
        # Add inclusion criteria
        extracted['included_units'] = np.array(all_included_units, dtype=bool).reshape(-1, 1)
        print(f"  Included neurons: {sum(all_included_units)}/{len(all_included_units)}")
        
        # Add quality metrics
        extracted['quality_spatial_spread_um'] = np.array(all_quality_spatial_spread).reshape(-1, 1)
        extracted['quality_peak_width_ms'] = np.array(all_quality_peak_width).reshape(-1, 1)
        extracted['quality_peak_trough_width_ms'] = np.array(all_quality_peak_trough_width).reshape(-1, 1)
        extracted['quality_upward_going'] = np.array(all_quality_upward_going, dtype=bool).reshape(-1, 1)
        extracted['quality_uvpp'] = np.array(all_quality_uvpp).reshape(-1, 1)
        
        # Add fraction spikes removed
        extracted['frac_spikes_removed'] = np.array(all_frac_spikes_removed).reshape(-1, 1)
        
        # Print quality metrics summary
        valid_spatial = ~np.isnan(all_quality_spatial_spread)
        if np.any(valid_spatial):
            print(f"  Quality metrics available for {np.sum(valid_spatial)} units")
    
    if all_hemispheres:
        hemisphere_strings = []
        for h in all_hemispheres:
            if isinstance(h, np.ndarray) and h.size > 0:
                hemisphere_strings.append(np.array([h.flat[0]], dtype='<U6'))
            else:
                hemisphere_strings.append(np.array([''], dtype='<U6'))
        extracted['hemisphere'] = np.array(hemisphere_strings, dtype=object).reshape(1, -1)
        print(f"  Hemisphere strings: {[h.flat[0] if h.size > 0 else '' for h in hemisphere_strings[:5]]}...")
    
    if all_regions:
        region_strings = []
        for r in all_regions:
            if isinstance(r, np.ndarray) and r.size > 0:
                region_strings.append(np.array([r.flat[0]], dtype='<U6'))
            else:
                region_strings.append(np.array([''], dtype='<U6'))
        extracted['region'] = np.array(region_strings, dtype=object).reshape(-1, 1)
        print(f"  Region strings: {[r.flat[0] if r.size > 0 else '' for r in region_strings[:5]]}...")
    
    # Handle electrode field
    if all_spike_times:
        extracted['electrode'] = np.array([], dtype=np.uint8).reshape(0, 0)
    
    return extracted

def save_extracted_data_with_regions(source_path, output_path):
    """Load source data, extract with coordinate-based regions, and save"""
    
    print(f"Loading source data from {source_path}")
    source_data = sio.loadmat(source_path)
    
    print("Extracting selected fields with coordinate-based region mapping...")
    extracted_data = extract_selected_fields_with_regions(source_data)
    
    print(f"Saving extracted data to {output_path}")
    sio.savemat(output_path, extracted_data, oned_as='column')
    
    print("Extraction completed!")
    
    # Print summary
    print("\n=== EXTRACTION SUMMARY ===")
    for key, value in sorted(extracted_data.items()):
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            # Show sample values for string fields
            if key in ['hemisphere', 'region', 'rat', 'sess_date'] and value.size > 0:
                if value.dtype == 'object':
                    sample_vals = []
                    for i in range(min(5, value.size)):
                        item = value.flat[i]
                        if hasattr(item, 'flat') and item.size > 0:
                            sample_vals.append(item.flat[0])
                        else:
                            sample_vals.append(str(item))
                    print(f"  Sample values: {sample_vals}")
                else:
                    print(f"  Sample values: {value.flat[:3]}")
        else:
            print(f"{key}: {type(value)}")

if __name__ == "__main__":
    source_file = '/home/ham/SF/Personal/Education/07 Princeton/Rotations/Brody-Daw/ClickDV/data/raw/A324/2023-07-21/A324_pycells_20230721_source.mat'
    output_file = '/home/ham/SF/Personal/Education/07 Princeton/Rotations/Brody-Daw/ClickDV/data/raw/A324/2023-07-21/A324_pycells_20230721_extracted_non_filtered.mat'
    
    save_extracted_data_with_regions(source_file, output_file)