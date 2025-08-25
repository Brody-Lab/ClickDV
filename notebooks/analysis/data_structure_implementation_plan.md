# Data Structure Implementation Plan
## ClickDV Project - Session Data Organization

### Overview

This document outlines the implementation of a three-DataFrame approach for organizing ClickDV session data. The approach separates concerns while maintaining referential integrity through trial_id keys:

1. **Trial DataFrame** (`trial_df`) - One row per trial with behavioral summary data
2. **Click DataFrame** (`click_df`) - One row per click event with detailed timing
3. **Decision Variables DataFrame** (`dv_df`) - One row per (trial, timepoint) with DV time series

### Data Structure Specifications

#### 1. Trial DataFrame Schema

```python
trial_df = pd.DataFrame({
    # Core identifiers
    'trial_id': int,                    # 0-based index: 0, 1, 2, ..., n_trials-1
    'original_trial_num': int,          # Original trial number before filtering
    
    # Choice and outcome
    'choice': int,                      # 0=left, 1=right
    'rewarded': bool,                   # True if trial was rewarded
    'violated': bool,                   # True if trial rule was violated
    
    # Timing events (all in seconds, absolute time)
    'cpoke_in': float,                  # Center poke entry time
    'cpoke_out': float,                 # Center poke exit time  
    'clicks_on': float,                 # Stimulus onset time
    'first_click': float,               # First click time (absolute)
    'last_click': float,                # Last click time (absolute)
    
    # Derived timing features
    'trial_duration': float,            # cpoke_out - cpoke_in
    'click_duration': float,            # last_click - first_click
    'time_to_first_click': float,       # first_click - clicks_on
    'decision_time': float,             # cpoke_out - last_click
    
    # Click summary statistics
    'n_left_clicks': int,               # Number of left clicks
    'n_right_clicks': int,              # Number of right clicks
    'total_clicks': int,                # Total clicks in trial
    'click_rate': float,                # Clicks per second during stimulus
    'click_asymmetry': float,           # (right - left) / (right + left)
})
```

#### 2. Click DataFrame Schema

```python
click_df = pd.DataFrame({
    # Link to trial data
    'trial_id': int,                    # Foreign key to trial_df.trial_id
    
    # Click identification
    'click_side': str,                  # 'left' or 'right'
    'click_number': int,                # Sequential number within trial (1, 2, 3, ...)
    'click_number_side': int,           # Sequential number within side (1st left, 2nd left, etc.)
    
    # Timing (absolute and relative)
    'click_time': float,                # Absolute click time
    'time_from_clicks_on': float,       # click_time - clicks_on
    'time_from_first_click': float,     # click_time - first_click_time  
    'time_from_cpoke_in': float,        # click_time - cpoke_in
    'time_to_cpoke_out': float,         # cpoke_out - click_time
    
    # Context information
    'choice': int,                      # Trial choice (for easy filtering)
    'click_side_matches_choice': bool,  # True if click side matches final choice
})
```

#### 3. Decision Variables DataFrame Schema

```python
dv_df = pd.DataFrame({
    # Link to trial data
    'trial_id': int,                    # Foreign key to trial_df.trial_id
    
    # Time information
    'time_bin_idx': int,                # Index into time_bins array (0, 1, 2, ...)
    'time_bin': float,                  # Time relative to clicks_on (-0.025, 0.025, ...)
    'time_absolute': float,             # Absolute time (clicks_on + time_bin)
    
    # Decision variable data
    'decision_variable': float,         # DV value (may be NaN)
    'is_valid': bool,                   # False if DV is NaN
    'model_accuracy': float,            # Cross-validation accuracy at this timepoint
    
    # Trial context (denormalized for easy analysis)
    'choice': int,                      # Trial choice
    'rewarded': bool,                   # Trial outcome
    'time_since_first_click': float,    # time_bin - (first_click - clicks_on)
    'time_since_last_click': float,     # time_bin - (last_click - clicks_on)
})
```

### Implementation Functions

#### Function 1: `create_trial_dataframe()`

```python
def create_trial_dataframe(data, valid_trial_mask, max_trials=240):
    """
    Create trial-level DataFrame with behavioral summary data.
    
    Parameters:
    - data: Loaded .mat file data dictionary
    - valid_trial_mask: Boolean array indicating valid trials
    - max_trials: Maximum number of trials to process
    
    Returns:
    - pd.DataFrame with trial-level data
    """
    # Extract basic trial data
    n_total = min(max_trials, len(data['cpoke_in'].flatten()))
    trial_ids = np.arange(np.sum(valid_trial_mask))
    
    # Timing data (already filtered to valid trials)
    cpoke_in = data['cpoke_in'].flatten()[:n_total][valid_trial_mask]
    cpoke_out = data['cpoke_out'].flatten()[:n_total][valid_trial_mask]
    clicks_on = np.array([clicks[0] for clicks in data['clicks_on'][:n_total]])[valid_trial_mask]
    first_clicks = np.array(data['first_clicks']).flatten()[:n_total][valid_trial_mask]
    last_clicks = np.array(data['last_clicks']).flatten()[:n_total][valid_trial_mask]
    
    # Choice and outcome data
    choices = data['pokedR'].flatten()[:n_total][valid_trial_mask].astype(int)
    rewarded = data['is_hit'].flatten()[:n_total][valid_trial_mask].astype(bool)
    violated = data['violated'].flatten()[:n_total][valid_trial_mask].astype(bool)
    
    # Calculate click counts per trial
    left_bups = data['left_bups'][:n_total][valid_trial_mask]
    right_bups = data['right_bups'][:n_total][valid_trial_mask]
    
    n_left_clicks = [len(clicks[0]) if clicks[0].size > 0 else 0 for clicks in left_bups]
    n_right_clicks = [len(clicks[0]) if clicks[0].size > 0 else 0 for clicks in right_bups]
    total_clicks = np.array(n_left_clicks) + np.array(n_right_clicks)
    
    # Derived features
    trial_duration = cpoke_out - cpoke_in
    click_duration = last_clicks - first_clicks
    time_to_first_click = first_clicks - clicks_on
    decision_time = cpoke_out - last_clicks
    
    # Click rate during stimulus period
    click_rate = total_clicks / click_duration
    click_rate[click_duration == 0] = 0  # Handle zero-duration trials
    
    # Click asymmetry: (right - left) / (right + left)
    click_asymmetry = (np.array(n_right_clicks) - np.array(n_left_clicks)) / total_clicks
    click_asymmetry[total_clicks == 0] = 0  # Handle trials with no clicks
    
    return pd.DataFrame({
        'trial_id': trial_ids,
        'original_trial_num': np.where(valid_trial_mask)[0],
        'choice': choices,
        'rewarded': rewarded,
        'violated': violated,
        'cpoke_in': cpoke_in,
        'cpoke_out': cpoke_out,
        'clicks_on': clicks_on,
        'first_click': first_clicks,
        'last_click': last_clicks,
        'trial_duration': trial_duration,
        'click_duration': click_duration,
        'time_to_first_click': time_to_first_click,
        'decision_time': decision_time,
        'n_left_clicks': n_left_clicks,
        'n_right_clicks': n_right_clicks,
        'total_clicks': total_clicks,
        'click_rate': click_rate,
        'click_asymmetry': click_asymmetry
    })
```

#### Function 2: `create_click_dataframe()`

```python
def create_click_dataframe(data, valid_trial_mask, max_trials=240):
    """
    Create click events DataFrame with one row per click.
    
    Parameters:
    - data: Loaded .mat file data dictionary  
    - valid_trial_mask: Boolean array indicating valid trials
    - max_trials: Maximum number of trials to process
    
    Returns:
    - pd.DataFrame with click event data
    """
    click_records = []
    valid_trial_indices = np.where(valid_trial_mask)[0]
    
    # Extract trial timing for context
    cpoke_in = data['cpoke_in'].flatten()[:max_trials][valid_trial_mask]
    cpoke_out = data['cpoke_out'].flatten()[:max_trials][valid_trial_mask]
    clicks_on = np.array([clicks[0] for clicks in data['clicks_on'][:max_trials]])[valid_trial_mask]
    first_clicks = np.array(data['first_clicks']).flatten()[:max_trials][valid_trial_mask]
    choices = data['pokedR'].flatten()[:max_trials][valid_trial_mask].astype(int)
    
    for trial_idx, original_trial in enumerate(valid_trial_indices):
        if original_trial >= max_trials:
            continue
            
        # Get clicks for this trial
        left_clicks = data['left_bups'][original_trial]
        right_clicks = data['right_bups'][original_trial]
        
        # Process left clicks
        if left_clicks[0].size > 0:
            left_times = left_clicks[0] + clicks_on[trial_idx]  # Convert to absolute time
            for click_num, click_time in enumerate(left_times):
                click_records.append({
                    'trial_id': trial_idx,
                    'click_side': 'left',
                    'click_number': None,  # Will fill after processing both sides
                    'click_number_side': click_num + 1,
                    'click_time': click_time,
                    'time_from_clicks_on': click_time - clicks_on[trial_idx],
                    'time_from_first_click': click_time - first_clicks[trial_idx],
                    'time_from_cpoke_in': click_time - cpoke_in[trial_idx],
                    'time_to_cpoke_out': cpoke_out[trial_idx] - click_time,
                    'choice': choices[trial_idx],
                    'click_side_matches_choice': (choices[trial_idx] == 0)  # 0=left
                })
        
        # Process right clicks
        if right_clicks[0].size > 0:
            right_times = right_clicks[0] + clicks_on[trial_idx]  # Convert to absolute time
            for click_num, click_time in enumerate(right_times):
                click_records.append({
                    'trial_id': trial_idx,
                    'click_side': 'right',
                    'click_number': None,  # Will fill after processing both sides
                    'click_number_side': click_num + 1,
                    'click_time': click_time,
                    'time_from_clicks_on': click_time - clicks_on[trial_idx],
                    'time_from_first_click': click_time - first_clicks[trial_idx],
                    'time_from_cpoke_in': click_time - cpoke_in[trial_idx],
                    'time_to_cpoke_out': cpoke_out[trial_idx] - click_time,
                    'choice': choices[trial_idx],
                    'click_side_matches_choice': (choices[trial_idx] == 1)  # 1=right
                })
    
    # Convert to DataFrame
    click_df = pd.DataFrame(click_records)
    
    # Add overall click number (sorted by time within each trial)
    if len(click_df) > 0:
        click_df = click_df.sort_values(['trial_id', 'click_time'])
        click_df['click_number'] = click_df.groupby('trial_id').cumcount() + 1
        click_df = click_df.reset_index(drop=True)
    
    return click_df
```

#### Function 3: `create_dv_dataframe()`

```python
def create_dv_dataframe(DVs, valid_choices, time_bins, accuracies, trial_df):
    """
    Create decision variables DataFrame in long format.
    
    Parameters:
    - DVs: Decision variables array (n_timepoints, n_trials)
    - valid_choices: Binary choices array
    - time_bins: Time bin definitions
    - accuracies: Model accuracies per timepoint
    - trial_df: Trial DataFrame for additional context
    
    Returns:
    - pd.DataFrame with decision variable time series
    """
    n_timepoints, n_trials = DVs.shape
    
    # Create base arrays
    trial_ids = np.repeat(range(n_trials), n_timepoints)
    time_bin_indices = np.tile(range(n_timepoints), n_trials)
    time_bins_repeated = np.tile(time_bins, n_trials)
    dv_values = DVs.T.flatten()  # Transpose then flatten to get correct order
    choices_repeated = np.repeat(valid_choices, n_timepoints)
    
    # Calculate absolute times
    clicks_on_repeated = np.repeat(trial_df['clicks_on'].values, n_timepoints)
    time_absolute = clicks_on_repeated + time_bins_repeated
    
    # Calculate relative times to click events
    first_clicks_repeated = np.repeat(trial_df['first_click'].values, n_timepoints)
    last_clicks_repeated = np.repeat(trial_df['last_click'].values, n_timepoints)
    time_since_first_click = time_bins_repeated - (first_clicks_repeated - clicks_on_repeated)
    time_since_last_click = time_bins_repeated - (last_clicks_repeated - clicks_on_repeated)
    
    # Model accuracy (repeated for each trial at each timepoint)
    accuracy_repeated = np.tile(accuracies, n_trials)
    
    # Create DataFrame
    dv_df = pd.DataFrame({
        'trial_id': trial_ids,
        'time_bin_idx': time_bin_indices,
        'time_bin': time_bins_repeated,
        'time_absolute': time_absolute,
        'decision_variable': dv_values,
        'is_valid': ~np.isnan(dv_values),
        'model_accuracy': accuracy_repeated,
        'choice': choices_repeated,
        'rewarded': np.repeat(trial_df['rewarded'].values, n_timepoints),
        'time_since_first_click': time_since_first_click,
        'time_since_last_click': time_since_last_click
    })
    
    return dv_df
```

#### Function 4: `save_session_data()`

```python
def save_session_data(trial_df, click_df, dv_df, neural_data, metadata, filepath):
    """
    Save all DataFrames and arrays to HDF5 file.
    
    Parameters:
    - trial_df, click_df, dv_df: DataFrames to save
    - neural_data: Neural firing rates array
    - metadata: Dictionary with session metadata
    - filepath: Output HDF5 file path
    """
    with pd.HDFStore(filepath, mode='w') as store:
        # Save DataFrames
        store.put('trials', trial_df, format='table')
        store.put('clicks', click_df, format='table')  
        store.put('decision_variables', dv_df, format='table')
        
        # Save neural data as array
        store.put('neural_data', pd.DataFrame(neural_data.reshape(neural_data.shape[0], -1)))
        
        # Save metadata
        metadata_df = pd.DataFrame([metadata])  # Convert dict to single-row DataFrame
        store.put('metadata', metadata_df, format='table')
        
        # Store neural data shape separately for reconstruction
        shape_df = pd.DataFrame([{'shape': neural_data.shape}])
        store.put('neural_data_shape', shape_df)
```

#### Function 5: `load_session_data()`

```python
def load_session_data(filepath):
    """
    Load all data from HDF5 file.
    
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
        
        # Load metadata
        metadata = store.get('metadata').iloc[0].to_dict()
    
    return {
        'trial_df': trial_df,
        'click_df': click_df, 
        'dv_df': dv_df,
        'neural_data': neural_data,
        'metadata': metadata
    }
```

### Integration with Existing Notebook

#### Step 1: Add at End of Current Notebook

Add new cells at the end of `decision_variables_extraction_v003.ipynb`:

```python
# =============================================================================
# 5. Data Structure Creation
# =============================================================================

print("Creating structured DataFrames for analysis...")

# Create trial-level DataFrame
trial_df = create_trial_dataframe(data, valid_trial_mask, max_trials=240)
print(f"✓ Trial DataFrame: {len(trial_df)} trials × {len(trial_df.columns)} columns")

# Create click events DataFrame  
click_df = create_click_dataframe(data, valid_trial_mask, max_trials=240)
print(f"✓ Click DataFrame: {len(click_df)} clicks × {len(click_df.columns)} columns")

# Create decision variables DataFrame
dv_df = create_dv_dataframe(DVs_2, valid_choices, time_bins, accuracies_2, trial_df)
print(f"✓ DV DataFrame: {len(dv_df)} timepoints × {len(dv_df.columns)} columns")

# Display summary statistics
print("\nData Structure Summary:")
print(f"  Trials: {len(trial_df)}")
print(f"  Total clicks: {len(click_df)}")
print(f"  DV timepoints: {len(dv_df)}")
print(f"  Neural data shape: {X.shape}")
```

#### Step 2: Add Data Validation

```python
# Data integrity checks
print("\nValidating data integrity...")

# Check trial_id alignment
assert len(trial_df) == len(valid_choices), "Trial DataFrame length mismatch"
assert set(click_df['trial_id']) <= set(trial_df['trial_id']), "Click trials not in trial_df"
assert set(dv_df['trial_id']) <= set(trial_df['trial_id']), "DV trials not in trial_df"

# Check data consistency
assert len(trial_df) == X.shape[2], "Neural data trial count mismatch"
assert len(time_bins) == X.shape[1], "Time bins count mismatch"
assert DVs_2.shape == (len(time_bins), len(trial_df)), "DV array shape mismatch"

print("✓ All data integrity checks passed")
```

#### Step 3: Add Usage Examples

```python
# Example analyses with new data structure

print("\n=== Example Analyses ===")

# 1. Trial-level analysis
print("1. Choice distribution:")
print(trial_df['choice'].value_counts())

print("\n2. Click rate by choice:")
print(trial_df.groupby('choice')['click_rate'].agg(['mean', 'std']))

# 2. Click-level analysis  
print("\n3. Click timing by side:")
click_timing = click_df.groupby('click_side')['time_from_clicks_on'].agg(['mean', 'std'])
print(click_timing)

print("\n4. Early vs late clicks:")
click_df['is_early'] = click_df['click_number'] <= 3
early_vs_late = click_df.groupby(['click_side', 'is_early'])['time_from_clicks_on'].mean()
print(early_vs_late)

# 3. Decision variable analysis
print("\n5. DV trajectory by choice:")
dv_summary = dv_df.groupby(['choice', 'time_bin'])['decision_variable'].mean().reset_index()
left_choice_dv = dv_summary[dv_summary['choice'] == 0]['decision_variable'].iloc[10]  # Mid-trial
right_choice_dv = dv_summary[dv_summary['choice'] == 1]['decision_variable'].iloc[10]
print(f"Mid-trial DV: Left={left_choice_dv:.3f}, Right={right_choice_dv:.3f}")
```

### Testing and Validation Procedures

#### Automated Checks

```python
def validate_data_structure(trial_df, click_df, dv_df, neural_data, original_data):
    """
    Comprehensive validation of created data structures.
    """
    checks_passed = 0
    total_checks = 0
    
    # 1. Basic structure checks
    total_checks += 1
    if len(trial_df.columns) >= 15:  # Expected minimum columns
        checks_passed += 1
        print("✓ Trial DataFrame has expected columns")
    else:
        print("✗ Trial DataFrame missing columns")
    
    # 2. Data completeness
    total_checks += 1
    if not trial_df.isnull().any().any():
        checks_passed += 1
        print("✓ No missing values in trial data")
    else:
        print("✗ Missing values found in trial data")
    
    # 3. Referential integrity
    total_checks += 1
    if click_df['trial_id'].isin(trial_df['trial_id']).all():
        checks_passed += 1
        print("✓ All click events link to valid trials")
    else:
        print("✗ Orphaned click events found")
    
    # 4. Data consistency
    total_checks += 1
    expected_dv_rows = len(trial_df) * len(neural_data.shape[1] if len(neural_data.shape) > 1 else 1)
    if len(dv_df) == expected_dv_rows:
        checks_passed += 1
        print("✓ DV DataFrame has expected number of rows")
    else:
        print(f"✗ DV DataFrame size mismatch: got {len(dv_df)}, expected {expected_dv_rows}")
    
    # 5. Value ranges
    total_checks += 1
    if trial_df['choice'].isin([0, 1]).all():
        checks_passed += 1
        print("✓ All choices are binary (0/1)")
    else:
        print("✗ Invalid choice values found")
    
    print(f"\nValidation Summary: {checks_passed}/{total_checks} checks passed")
    return checks_passed == total_checks
```

### File I/O Strategy

#### Recommended File Structure

```
data/processed/session_data/
├── A324_2023-07-27_session_data.h5     # Main data file
├── A324_2023-07-27_metadata.json       # Human-readable metadata
└── A324_2023-07-27_summary.txt         # Analysis summary
```

#### Save Function Usage

```python
# Prepare metadata
session_metadata = {
    'session_id': SESSION_ID,
    'session_date': SESSION_DATE,
    'n_trials': len(trial_df),
    'n_neurons': X.shape[0],
    'n_timepoints': len(time_bins),
    'analysis_params': {
        'gaussian_sigma_ms': GAUSSIAN_SIGMA_MS,
        'time_window': TIME_WINDOW,
        'cv_folds': CV_FOLDS
    },
    'creation_timestamp': pd.Timestamp.now().isoformat(),
    'notebook_version': 'decision_variables_extraction_v003'
}

# Save structured data
output_file = OUTPUT_DIR / f"{SESSION_ID}_{SESSION_DATE}_session_data.h5"
save_session_data(trial_df, click_df, dv_df, X, session_metadata, output_file)
print(f"✓ Session data saved to: {output_file}")
```

### Usage Examples

#### Analysis Workflows

**1. Trial-level behavioral analysis:**
```python
# Response time analysis
trial_df['response_time'] = trial_df['cpoke_out'] - trial_df['cpoke_in']
sns.boxplot(data=trial_df, x='choice', y='response_time')

# Click strategy analysis  
trial_df['click_bias'] = (trial_df['n_right_clicks'] - trial_df['n_left_clicks']) / trial_df['total_clicks']
correlation = trial_df['click_bias'].corr(trial_df['choice'])
print(f"Click bias vs choice correlation: {correlation:.3f}")
```

**2. Click dynamics analysis:**
```python
# Inter-click intervals
click_df['prev_click_time'] = click_df.groupby('trial_id')['click_time'].shift(1)
click_df['inter_click_interval'] = click_df['click_time'] - click_df['prev_click_time']

# Click rate evolution
click_df['click_epoch'] = pd.cut(click_df['time_from_clicks_on'], bins=5, labels=['early', 'mid-early', 'mid', 'mid-late', 'late'])
click_rate_evolution = click_df.groupby(['click_epoch', 'choice']).size()
```

**3. Decision variable analysis:**
```python
# DV trajectories with confidence intervals
dv_summary = dv_df.groupby(['choice', 'time_bin']).agg({
    'decision_variable': ['mean', 'std', 'count']
}).reset_index()

# Join with trial outcomes for richer analysis
enriched_dv = dv_df.merge(trial_df[['trial_id', 'rewarded', 'click_rate']], on='trial_id')
```

**4. Cross-structure analysis:**
```python
# How do early clicks predict later decision variables?
early_clicks = click_df[click_df['click_number'] <= 3].groupby('trial_id').agg({
    'click_side': lambda x: (x == 'right').mean(),  # Proportion right clicks
    'time_from_clicks_on': 'mean'  # Average early click timing
}).rename(columns={'click_side': 'early_right_prop'})

# Merge with DV data at decision time
decision_time_dv = dv_df[dv_df['time_bin'] >= 0.5].groupby('trial_id')['decision_variable'].mean()
click_dv_analysis = early_clicks.merge(decision_time_dv.to_frame('late_dv'), left_index=True, right_index=True)
```

This implementation plan provides a comprehensive, production-ready approach to restructuring the ClickDV session data into analyzable DataFrames while maintaining data integrity and enabling flexible analysis workflows.