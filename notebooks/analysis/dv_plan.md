# Decision Variables Extraction Implementation Plan

## Section 1: Setup & Data Loading
- Import libraries (numpy, scipy, sklearn, matplotlib, seaborn)
- Load A324_pycells_20230727.mat file
- Examine data structure and extract key fields
- Display essential statistics (n_trials, n_neurons, session duration)

## Section 2: Data Preprocessing & Quality Control
- Apply neuron quality filters (spatial spread, peak width, presence ratio)
- Extract behavioral choices and apply trial exclusion criteria
- Implement 50ms Gaussian smoothing for firing rate calculation using scipy.ndimage.gaussian_filter1d
- Create time series aligned to cpoke_in (-0.5s to +1.5s, 50ms sampling)
- Format as (n_neurons, n_timepoints, n_trials) floating-point array in Hz
- Session validation: ≥300 trials, ≤8% lapse rate, choice balance check

## Section 3: Logistic Regression Model Training
- Implement 10-fold stratified cross-validation
- Find optimal regularization parameter for each time point
- Calculate geometric mean of regularization parameters
- Refit models using constant regularization across time
- Output essential performance metrics only

## Section 4: Decision Variables Calculation & Visualization
- Extract decision variables using fitted logistic regression models
- Calculate choice prediction accuracy over time
- Apply sign correction for visualization
- Generate key plots: accuracy evolution, DV trajectories by choice, model weights heatmap
- Minimal text output focusing on data shapes and key statistics

## Section 5: Validation & Results Export
- Compare results against paper benchmarks
- Validate temporal evolution patterns and check for overfitting
- Export processed decision variables and performance summary
- Document essential findings for GLM analysis

## Implementation Guidelines

### Output Philosophy
- Minimize print statements to essential information only (data shapes, key statistics, validation results)
- Focus on code execution over extensive documentation
- Brief, technical markdown explanations for methodology

### Gaussian Smoothing Implementation
- Use 50ms Gaussian kernel following Uberphys methodology
- High-resolution (1ms) spike trains convolved with Gaussian kernel, then downsampled
- Output continuous firing rates in Hz with temporal correlations
- Maintain consistency with published approach

### Code Structure
- Direct data examination over generic patterns
- Simple, exploratory code appropriate for research notebook
- Complete functions only (no incomplete signatures)
- Professional formatting without decorative elements