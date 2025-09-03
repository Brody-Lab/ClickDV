"""
Quality filtering utilities for neural data

Applies quality criteria to filter neurons based on waveform characteristics:
- Spatial spread < 150 μm
- Peak width < 1 ms
- Peak-trough width < 1 ms
- Upward-going spike = FALSE
- uVpp > 50 μV
"""

import numpy as np
import scipy.io as sio
from typing import Dict, Tuple, Optional


def apply_quality_filter(data: Dict, 
                        spatial_spread_max: float = 150.0,
                        peak_width_max: float = 1.0,
                        peak_trough_width_max: float = 1.0,
                        allow_upward_going: bool = False,
                        uvpp_min: float = 50.0,
                        verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Apply quality filtering criteria to neural data.
    
    If 'included_units' field exists, uses that as the primary criterion.
    Otherwise falls back to quality metrics filtering.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing neural data with quality metrics
    spatial_spread_max : float
        Maximum allowed spatial spread in μm (default: 150)
    peak_width_max : float
        Maximum allowed peak width in ms (default: 1.0)
    peak_trough_width_max : float
        Maximum allowed peak-trough width in ms (default: 1.0)
    allow_upward_going : bool
        Whether to allow upward-going spikes (default: False)
    uvpp_min : float
        Minimum required peak-to-peak voltage in μV (default: 50)
    verbose : bool
        Print filtering statistics (default: True)
        
    Returns:
    --------
    quality_mask : np.ndarray
        Boolean mask indicating which neurons pass quality criteria
    filter_stats : dict
        Dictionary with filtering statistics and breakdown
    """
    
    # Check if included_units field exists
    if 'included_units' in data:
        # Use pre-computed inclusion criteria
        quality_mask = data['included_units'].flatten().astype(bool)
        n_total = len(quality_mask)
        n_pass = np.sum(quality_mask)
        
        # Compile statistics
        filter_stats = {
            'n_total': n_total,
            'n_pass': n_pass,
            'pass_rate': n_pass / n_total,
            'method': 'included_units',
            'criteria': {
                'included_units': {
                    'threshold': 'Pre-computed inclusion',
                    'n_pass': n_pass,
                    'pass_rate': n_pass / n_total
                }
            }
        }
        
        if verbose:
            print(f"=== QUALITY FILTERING RESULTS ===")
            print(f"Using pre-computed 'included_units' field")
            print(f"Total neurons: {n_total}")
            print(f"Included neurons: {n_pass} ({n_pass/n_total:.1%})")
            print(f"Excluded neurons: {n_total - n_pass} ({(n_total - n_pass)/n_total:.1%})")
        
        return quality_mask, filter_stats
    
    # Fall back to quality metrics filtering
    # Extract quality metrics
    spatial_spread = data['quality_spatial_spread_um'].flatten()
    peak_width = data['quality_peak_width_ms'].flatten()
    peak_trough_width = data['quality_peak_trough_width_ms'].flatten()
    upward_going = data['quality_upward_going'].flatten().astype(bool)
    uvpp = data['quality_uvpp'].flatten()
    
    n_total = len(spatial_spread)
    
    # Apply individual criteria
    filter_spatial = spatial_spread < spatial_spread_max
    filter_peak_width = peak_width < peak_width_max
    filter_peak_trough = peak_trough_width < peak_trough_width_max
    filter_not_upward = ~upward_going if not allow_upward_going else np.ones(n_total, dtype=bool)
    filter_voltage = uvpp > uvpp_min
    
    # Handle NaN values (fail NaN values)
    filter_spatial = filter_spatial & ~np.isnan(spatial_spread)
    filter_peak_width = filter_peak_width & ~np.isnan(peak_width)
    filter_peak_trough = filter_peak_trough & ~np.isnan(peak_trough_width)
    filter_voltage = filter_voltage & ~np.isnan(uvpp)
    
    # Combined filter - all criteria must be met
    quality_mask = (filter_spatial & 
                   filter_peak_width & 
                   filter_peak_trough & 
                   filter_not_upward & 
                   filter_voltage)
    
    n_pass = np.sum(quality_mask)
    
    # Compile statistics
    filter_stats = {
        'n_total': n_total,
        'n_pass': n_pass,
        'pass_rate': n_pass / n_total,
        'method': 'quality_metrics',
        'criteria': {
            'spatial_spread': {
                'threshold': f'<{spatial_spread_max} μm',
                'n_pass': np.sum(filter_spatial),
                'pass_rate': np.sum(filter_spatial) / n_total
            },
            'peak_width': {
                'threshold': f'<{peak_width_max} ms',
                'n_pass': np.sum(filter_peak_width),
                'pass_rate': np.sum(filter_peak_width) / n_total
            },
            'peak_trough_width': {
                'threshold': f'<{peak_trough_width_max} ms',
                'n_pass': np.sum(filter_peak_trough),
                'pass_rate': np.sum(filter_peak_trough) / n_total
            },
            'not_upward_going': {
                'threshold': 'FALSE' if not allow_upward_going else 'Any',
                'n_pass': np.sum(filter_not_upward),
                'pass_rate': np.sum(filter_not_upward) / n_total
            },
            'voltage': {
                'threshold': f'>{uvpp_min} μV',
                'n_pass': np.sum(filter_voltage),
                'pass_rate': np.sum(filter_voltage) / n_total
            }
        }
    }
    
    if verbose:
        print(f"=== QUALITY FILTERING RESULTS ===")
        print(f"Using quality metrics filtering")
        print(f"Total neurons: {n_total}")
        print(f"Pass quality filter: {n_pass} ({n_pass/n_total:.1%})")
        print(f"Filtered out: {n_total - n_pass} ({(n_total - n_pass)/n_total:.1%})")
        print()
        print("Individual criteria:")
        for criterion, stats in filter_stats['criteria'].items():
            print(f"  {criterion} {stats['threshold']}: "
                  f"{stats['n_pass']}/{n_total} ({stats['pass_rate']:.1%})")
    
    return quality_mask, filter_stats


def create_filtered_dataset(data: Dict, quality_mask: np.ndarray, 
                           verbose: bool = True) -> Dict:
    """
    Create dataset with original data plus filtered fields.
    
    Parameters:
    -----------
    data : dict
        Original neural data dictionary
    quality_mask : np.ndarray
        Boolean mask indicating which neurons to keep
    verbose : bool
        Print dataset statistics
        
    Returns:
    --------
    filtered_data : dict
        Dictionary with all original data plus filtered fields
    """
    
    # Start with a copy of all original data
    filtered_data = data.copy()
    
    n_original = len(quality_mask)
    n_filtered = np.sum(quality_mask)
    
    # Create filtered versions of neural data fields
    neural_fields = ['raw_spike_time_s', 'hemisphere', 'region']
    
    for field in neural_fields:
        if field in data:
            original_data = data[field]
            
            # Check both dimensions for neuron-level data
            # hemisphere and region might be shaped as (1, n_neurons) or (n_neurons, 1)
            if original_data.shape[0] == len(quality_mask):
                # Shape is (n_neurons, ...)
                filt_field_name = f'filt_{field.replace("raw_", "").replace("_s", "")}'
                filtered_data[filt_field_name] = original_data[quality_mask]
            elif original_data.shape[1] == len(quality_mask):
                # Shape is (1, n_neurons) - need to transpose, filter, then transpose back
                filt_field_name = f'filt_{field.replace("raw_", "").replace("_s", "")}'
                filtered_data[filt_field_name] = original_data[:, quality_mask]
            else:
                # Not neuron-level data, skip filtering
                pass
    
    if verbose:
        print(f"\n=== FILTERED DATASET CREATED ===")
        print(f"Original neurons: {n_original}")
        print(f"Filtered neurons: {n_filtered}")
        print(f"Reduction: {(n_original - n_filtered)/n_original:.1%}")
        
        # Show region breakdown for filtered data
        if 'filt_region' in filtered_data:
            regions = []
            for r in filtered_data['filt_region'].flatten():
                if hasattr(r, 'size') and r.size > 0:
                    regions.append(str(r.flat[0] if hasattr(r, 'flat') else r))
                elif isinstance(r, str):
                    regions.append(r)
            
            if regions:
                unique_regions, counts = np.unique(regions, return_counts=True)
                print("Filtered regions:")
                for region, count in zip(unique_regions, counts):
                    print(f"  {region}: {count} neurons")
    
    return filtered_data


def filter_and_save(input_file: str, output_file: str, **filter_kwargs) -> Dict:
    """
    Load data, apply quality filtering, and save filtered dataset.
    
    Parameters:
    -----------
    input_file : str
        Path to input .mat file with quality metrics
    output_file : str
        Path for output filtered .mat file
    **filter_kwargs : 
        Additional arguments for apply_quality_filter()
        
    Returns:
    --------
    filter_stats : dict
        Filtering statistics
    """
    
    print(f"Loading: {input_file}")
    data = sio.loadmat(input_file)
    
    print("Applying quality filtering...")
    quality_mask, filter_stats = apply_quality_filter(data, **filter_kwargs)
    
    print("Creating filtered dataset...")
    filtered_data = create_filtered_dataset(data, quality_mask)
    
    print(f"Saving: {output_file}")
    sio.savemat(output_file, filtered_data, oned_as='column')
    
    print("✓ Quality filtering completed!")
    
    return filter_stats


if __name__ == "__main__":
    # Example usage - will use included_units if available, otherwise quality metrics
    input_file = "/home/ham/SF/Personal/Education/07 Princeton/Rotations/Brody-Daw/ClickDV/data/raw/A324/2023-07-21/A324_pycells_20230721_extracted_non_filtered.mat"
    output_file = "/home/ham/SF/Personal/Education/07 Princeton/Rotations/Brody-Daw/ClickDV/data/raw/A324/2023-07-21/A324_pycells_20230721.mat"
    
    # Apply filtering - will automatically use included_units if present
    stats = filter_and_save(input_file, output_file)