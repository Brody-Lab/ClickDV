#!/usr/bin/env python3
"""
Plot comparison between log-odds variable from MATLAB file and DV from HDF5 file.
Displays the first 240 trials only, similar to individual plots in the decision_variables_visualization notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.io import loadmat
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Data paths
MATLAB_FILE = Path("data/raw/A324/2023-07-27/DVs_A324_2023_07_27.mat")
HDF5_FILE = Path("data/processed/A324/2023-07-27/A324_2023-07-27_session_data.h5")

def load_matlab_dvs(mat_file_path, max_trials=240):
    """
    Load log-odds decision variables from MATLAB file.
    
    Returns:
    - Dictionary with log-odds data and trial information
    """
    print(f"Loading MATLAB file: {mat_file_path}")
    mat_data = loadmat(mat_file_path)
    
    # Print available keys for debugging
    available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    print(f"  Available keys: {available_keys[:10]}...")  # Show first 10 keys
    
    # Extract log-odds data - this is the key variable we want
    log_odds = None
    if 'log_odds' in mat_data:
        log_odds = mat_data['log_odds']
        print(f"  Found log_odds with shape: {log_odds.shape}")
    elif 'log_odds_same_lambda' in mat_data:
        log_odds = mat_data['log_odds_same_lambda']
        print(f"  Found log_odds_same_lambda with shape: {log_odds.shape}")
    
    # Get trial information
    trials_data = mat_data.get('Trials', None)
    
    # Limit to first max_trials
    if log_odds is not None and log_odds.shape[1] > max_trials:
        log_odds = log_odds[:, :max_trials]
        print(f"  Limited to first {max_trials} trials")
    
    return {'log_odds': log_odds, 'trials': trials_data}

def load_session_data(filepath):
    """
    Load all data from HDF5 file using the same function as in the notebook.
    
    Parameters:
    - filepath: HDF5 file path
    
    Returns:
    - Dictionary with all loaded data
    """
    with pd.HDFStore(filepath, mode='r') as store:
        trial_df = store.get('trials')
        click_df = store.get('clicks')
        dv_df = store.get('decision_variables')
        
        # Reconstruct neural data
        neural_flat = store.get('neural_data').values
        neural_shape = store.get('neural_data_shape')['shape'].iloc[0]
        neural_data = neural_flat.reshape(neural_shape)
        
        # Load metadata and parse JSON
        metadata_df = store.get('metadata')
        metadata = metadata_df.iloc[0].to_dict()

        # Parse the JSON string back to dictionary
        if 'analysis_params' in metadata and isinstance(metadata['analysis_params'], str):
            metadata['analysis_params'] = json.loads(metadata['analysis_params'])
        if 'data_shapes' in metadata and isinstance(metadata['data_shapes'], str):
            metadata['data_shapes'] = json.loads(metadata['data_shapes'])

    return {
        'trial_df': trial_df,
        'click_df': click_df, 
        'dv_df': dv_df,
        'neural_data': neural_data,
        'metadata': metadata
    }

def load_hdf5_dvs(h5_file_path, max_trials=240):
    """
    Load decision variables from HDF5 file (from GLM processing).
    
    Returns:
    - DataFrame with DV data for first max_trials
    """
    print(f"Loading HDF5 file: {h5_file_path}")
    
    # Use the load_session_data function
    loaded_data = load_session_data(h5_file_path)
    
    # Extract the DataFrames we need
    dv_df = loaded_data['dv_df']
    trial_df = loaded_data['trial_df']
    
    # Merge to get original trial numbers
    dv_with_orig = dv_df.merge(trial_df[['trial_id', 'original_trial_num']], on='trial_id')
    
    # Filter to first max_trials based on original trial numbers
    dv_filtered = dv_with_orig[dv_with_orig['original_trial_num'] <= max_trials].copy()
    
    print(f"  Loaded {len(dv_filtered)} timepoints for {dv_filtered['trial_id'].nunique()} trials")
    print(f"  Original trial range: {dv_filtered['original_trial_num'].min()} to {dv_filtered['original_trial_num'].max()}")
    
    return dv_filtered

def plot_dv_comparison(matlab_data, hdf5_data):
    """
    Create comparison plot of MATLAB log-odds and HDF5 DV trajectories.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: HDF5 Decision Variables (from GLM)
    # Separate by choice
    left_trials = hdf5_data[hdf5_data['choice'] == 0]
    right_trials = hdf5_data[hdf5_data['choice'] == 1]
    
    # Calculate mean trajectories
    left_mean = left_trials.groupby('time_bin')['decision_variable'].agg(['mean', 'sem'])
    right_mean = right_trials.groupby('time_bin')['decision_variable'].agg(['mean', 'sem'])
    
    # Plot HDF5 data
    ax1.plot(left_mean.index, left_mean['mean'], 'r-', label=f'Left (n={left_trials["trial_id"].nunique()})', linewidth=2)
    ax1.fill_between(left_mean.index, 
                     left_mean['mean'] - left_mean['sem'], 
                     left_mean['mean'] + left_mean['sem'], 
                     alpha=0.3, color='red')
    
    ax1.plot(right_mean.index, right_mean['mean'], 'b-', label=f'Right (n={right_trials["trial_id"].nunique()})', linewidth=2)
    ax1.fill_between(right_mean.index, 
                     right_mean['mean'] - right_mean['sem'], 
                     right_mean['mean'] + right_mean['sem'], 
                     alpha=0.3, color='blue')
    
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='black', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Time from first click (s)')
    ax1.set_ylabel('Decision Variable')
    ax1.set_title('GLM-extracted DV (HDF5 file)\nFirst 240 trials')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MATLAB Log-Odds
    if matlab_data['log_odds'] is not None:
        log_odds_array = matlab_data['log_odds']
        
        # log_odds should be time_bins x trials
        # Create time axis (assuming same time bins as HDF5 data)
        time_bins = sorted(hdf5_data['time_bin'].unique())
        
        # Calculate mean across trials
        if len(log_odds_array.shape) > 1:
            mean_log_odds = np.nanmean(log_odds_array, axis=1)
            sem_log_odds = np.nanstd(log_odds_array, axis=1) / np.sqrt(np.sum(~np.isnan(log_odds_array), axis=1))
            
            ax2.plot(time_bins[:len(mean_log_odds)], mean_log_odds, 'g-', 
                    label=f'Log-odds (n={log_odds_array.shape[1]} trials)', linewidth=2)
            ax2.fill_between(time_bins[:len(mean_log_odds)], 
                            mean_log_odds - sem_log_odds, 
                            mean_log_odds + sem_log_odds, 
                            alpha=0.3, color='green')
        else:
            ax2.plot(time_bins[:len(log_odds_array)], log_odds_array, 'g-', label='Log-odds', linewidth=2)
        
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(0, color='black', linestyle=':', alpha=0.7)
        ax2.set_xlabel('Time from first click (s)')
        ax2.set_ylabel('Log-Odds')
        ax2.set_title('MATLAB Log-Odds (DVs file)\nFirst 240 trials')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'MATLAB data not available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # Overall title
    fig.suptitle('Decision Variable Comparison: GLM-extracted vs MATLAB Log-Odds\nA324 Session 2023-07-27 (First 240 trials)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """Main execution function."""
    print("=" * 60)
    print("Decision Variable Comparison Plot")
    print("=" * 60)
    
    # Check if files exist
    if not MATLAB_FILE.exists():
        print(f"ERROR: MATLAB file not found: {MATLAB_FILE}")
        return
    
    if not HDF5_FILE.exists():
        print(f"ERROR: HDF5 file not found: {HDF5_FILE}")
        return
    
    # Load data
    print("\nLoading data...")
    matlab_data = load_matlab_dvs(MATLAB_FILE, max_trials=240)
    hdf5_data = load_hdf5_dvs(HDF5_FILE, max_trials=240)
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    fig = plot_dv_comparison(matlab_data, hdf5_data)
    
    # Save figure
    output_path = Path("figures/dv_comparison_240trials.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    # Show plot
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()