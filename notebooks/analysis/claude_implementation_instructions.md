# Claude Implementation Instructions
## Data Structure Creation for ClickDV Project

### Context

You are implementing a data structure reorganization for the ClickDV computational neuroscience project. The current analysis notebook has extracted decision variables from neural data, and now needs to organize this data into structured DataFrames for easier analysis.

### Required Files to Read

**FIRST, read these files to understand the task:**

1. **Implementation Plan**: `notebooks/analysis/data_structure_implementation_plan.md`
   - Contains complete technical specifications 
   - Includes all function definitions you need to implement
   - Review this file thoroughly before proceeding

2. **Current Notebook**: `notebooks/analysis/decision_variables_extraction_v003.ipynb`
   - Contains all the processed data variables you'll work with
   - Key variables: `X`, `DVs_2`, `valid_choices`, `data`, `valid_trial_mask`, `time_bins`, `accuracies_2`
   - Shows the current state of data processing

3. **Project Context**: `CLAUDE.md` in the project root
   - Provides project background and coding standards

### Implementation Steps

#### Step 1: Verify Current State

1. Open the notebook `notebooks/analysis/decision_variables_extraction_v003.ipynb`
2. Verify that these key variables exist and have expected shapes:
   - `X`: Neural data array, should be shape (1150, 22, 186) 
   - `DVs_2`: Decision variables, should be shape (22, 186)
   - `valid_choices`: Choice array, should be length 186
   - `data`: Original .mat file data dictionary
   - `valid_trial_mask`: Boolean mask for valid trials
   - `time_bins`: Time bin definitions, should be length 22
   - `accuracies_2`: Model accuracies per timepoint

#### Step 2: Add Implementation Functions

**Location**: Add these as new cells after the current analysis (around cell 56+)

Copy the following functions from the implementation plan document:

1. `create_trial_dataframe()` - Creates trial-level summary DataFrame
2. `create_click_dataframe()` - Creates click events DataFrame  
3. `create_dv_dataframe()` - Creates decision variables time series DataFrame
4. `save_session_data()` - Saves all data to HDF5 format
5. `load_session_data()` - Loads data back from HDF5
6. `validate_data_structure()` - Validates data integrity

**Important**: Copy these functions exactly as written in the implementation plan. They are designed to work with the specific variable names and data structures in the current notebook.

#### Step 3: Create Data Structures

**Add a new cell** with this code:

```python
# =============================================================================
# 5. Data Structure Creation
# =============================================================================

print("Creating structured DataFrames for analysis...")

# Extract the max_trials value used in preprocessing (should be 240)
max_trials = 240

# Create trial-level DataFrame
print("Creating trial DataFrame...")
trial_df = create_trial_dataframe(data, valid_trial_mask, max_trials)
print(f"✓ Trial DataFrame: {len(trial_df)} trials × {len(trial_df.columns)} columns")

# Create click events DataFrame  
print("Creating click DataFrame...")
click_df = create_click_dataframe(data, valid_trial_mask, max_trials)
print(f"✓ Click DataFrame: {len(click_df)} clicks × {len(click_df.columns)} columns")

# Create decision variables DataFrame
print("Creating DV DataFrame...")
dv_df = create_dv_dataframe(DVs_2, valid_choices, time_bins, accuracies_2, trial_df)
print(f"✓ DV DataFrame: {len(dv_df)} timepoints × {len(dv_df.columns)} columns")

# Display basic info
print("\nData Structure Summary:")
print(f"  Trials: {len(trial_df)}")
print(f"  Total clicks: {len(click_df)}")  
print(f"  DV timepoints: {len(dv_df)}")
print(f"  Neural data shape: {X.shape}")

# Display first few rows of each DataFrame
print("\nTrial DataFrame preview:")
print(trial_df.head(3))

print("\nClick DataFrame preview:")  
print(click_df.head(3))

print("\nDV DataFrame preview:")
print(dv_df.head(3))
```

#### Step 4: Add Data Validation

**Add another new cell**:

```python
# Data integrity validation
print("Validating data structures...")

# Run comprehensive validation
validation_passed = validate_data_structure(trial_df, click_df, dv_df, X, data)

if validation_passed:
    print("\n✅ All validation checks passed - data structures are ready for analysis")
else:
    print("\n❌ Validation failed - check the errors above before proceeding")

# Additional basic checks
print("\nBasic consistency checks:")
print(f"✓ Trial DataFrame shape: {trial_df.shape}")
print(f"✓ Click DataFrame shape: {click_df.shape}")
print(f"✓ DV DataFrame shape: {dv_df.shape}")
print(f"✓ Unique trials in click_df: {click_df['trial_id'].nunique()}")
print(f"✓ Unique trials in dv_df: {dv_df['trial_id'].nunique()}")
print(f"✓ Expected trials: {len(trial_df)}")
```

#### Step 5: Create Usage Examples

**Add another new cell**:

```python
print("=== Example Analyses with New Data Structures ===")

# 1. Trial-level analysis
print("\n1. Choice distribution:")
choice_dist = trial_df['choice'].value_counts()
print(f"   Left (0): {choice_dist.get(0, 0)} trials")
print(f"   Right (1): {choice_dist.get(1, 0)} trials")

print("\n2. Click rate by choice:")
click_rate_by_choice = trial_df.groupby('choice')['click_rate'].agg(['mean', 'std', 'count'])
print(click_rate_by_choice)

# 2. Click-level analysis
print("\n3. Click distribution by side:")
click_dist = click_df['click_side'].value_counts()
print(click_dist)

print("\n4. Average click timing:")
click_timing = click_df.groupby('click_side')['time_from_clicks_on'].agg(['mean', 'std'])
print(click_timing)

# 3. Decision variable analysis
print("\n5. DV summary by choice:")
dv_summary = dv_df.groupby('choice')['decision_variable'].agg(['mean', 'std', 'count'])
print(dv_summary)

print("\n6. DV trajectory shape:")
dv_pivot = dv_df.pivot_table(values='decision_variable', 
                           index='time_bin', 
                           columns='choice', 
                           aggfunc='mean')
print(f"   DV trajectory table shape: {dv_pivot.shape}")
print(f"   Time range: {dv_pivot.index.min():.3f} to {dv_pivot.index.max():.3f}s")
```

#### Step 6: Save the Data

**Add final cell**:

```python
# Save structured data to HDF5 file
print("Saving structured session data...")

# Prepare metadata
session_metadata = {
    'session_id': SESSION_ID,
    'session_date': SESSION_DATE,
    'n_trials': len(trial_df),
    'n_neurons': X.shape[0],
    'n_timepoints': len(time_bins),
    'n_clicks': len(click_df),
    'analysis_params': {
        'gaussian_sigma_ms': GAUSSIAN_SIGMA_MS,
        'time_window': TIME_WINDOW,
        'cv_folds': CV_FOLDS,
        'sampling_interval_ms': SAMPLING_INTERVAL_MS
    },
    'data_structure_version': '1.0',
    'creation_timestamp': pd.Timestamp.now().isoformat(),
    'notebook_version': 'decision_variables_extraction_v003'
}

# Define output file path
output_file = OUTPUT_DIR / f"{SESSION_ID}_{SESSION_DATE}_structured_data.h5"
print(f"Output file: {output_file}")

# Save all data
save_session_data(trial_df, click_df, dv_df, X, session_metadata, str(output_file))

print(f"✅ Session data saved successfully!")
print(f"   File: {output_file}")
print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

# Test loading the data back
print("\nTesting data reload...")
loaded_data = load_session_data(str(output_file))

print("✅ Data loaded successfully!")
print(f"   Trial DataFrame: {loaded_data['trial_df'].shape}")
print(f"   Click DataFrame: {loaded_data['click_df'].shape}")
print(f"   DV DataFrame: {loaded_data['dv_df'].shape}")
print(f"   Neural data: {loaded_data['neural_data'].shape}")
print(f"   Metadata keys: {list(loaded_data['metadata'].keys())}")

print("\n🎉 Data structure implementation complete!")
```

### Critical Implementation Notes

#### Variable Name Mapping
Make sure these variables from the notebook are used correctly:

- `data` → Original .mat file data (contains 'left_bups', 'right_bups', etc.)
- `valid_trial_mask` → Boolean array for filtering valid trials  
- `valid_choices` → Filtered choices array (0=left, 1=right)
- `DVs_2` → Decision variables array, shape (timepoints, trials)
- `X` → Neural firing rates, shape (neurons, timepoints, trials)
- `time_bins` → Time bin definitions
- `accuracies_2` → Model accuracies per timepoint

#### Expected Variable Values
Before implementation, verify:
- `len(valid_choices)` should be 186 (number of valid trials)
- `DVs_2.shape` should be (22, 186) 
- `X.shape` should be (1150, 22, 186)
- `SESSION_ID` should be 'A324'
- `SESSION_DATE` should be '2023-07-27'

#### Error Handling
If you encounter errors:

1. **"Variable not found"** → Check that you're running cells in order and all preprocessing has completed
2. **"Shape mismatch"** → Verify the data dimensions match expected values above
3. **"Invalid trial_id"** → Check that `valid_trial_mask` is being applied correctly
4. **"Save failed"** → Ensure `OUTPUT_DIR` exists and is writable

### Testing Your Implementation

#### After completing all steps, verify:

1. **Three DataFrames created successfully**
   - `trial_df`: ~186 rows, 19 columns
   - `click_df`: Variable rows (total clicks), 12 columns  
   - `dv_df`: ~4092 rows (186 trials × 22 timepoints), 12 columns

2. **Data integrity maintained**
   - All validation checks pass
   - Trial IDs align across DataFrames
   - No unexpected NaN values (except in DV data after last clicks)

3. **HDF5 file created and loadable**
   - File exists at expected location
   - Can be loaded back successfully
   - Contains all expected data structures

4. **Example analyses work**
   - Can group by choice, time, click side
   - Can calculate summary statistics
   - Can create pivot tables for visualization

### Expected Output Files

After successful implementation:

- **HDF5 Data File**: `notebooks/analysis/outputs/A324_2023-07-27_structured_data.h5`
  - Contains all DataFrames and neural data
  - ~100-200 MB file size
  - Loadable with the `load_session_data()` function

### Success Criteria

✅ **Implementation successful if:**
- All cells execute without errors
- All validation checks pass  
- HDF5 file is created and can be reloaded
- Example analyses produce reasonable results
- Data dimensions match expected values

🚨 **Stop and debug if:**
- Any validation check fails
- DataFrame shapes are unexpected
- Cannot save or reload data
- Example analyses produce errors

### Final Notes

This implementation preserves all existing analysis while adding structured data access. The original variables (`X`, `DVs_2`, etc.) remain unchanged, and the new DataFrames provide alternative views of the same data optimized for different types of analysis.