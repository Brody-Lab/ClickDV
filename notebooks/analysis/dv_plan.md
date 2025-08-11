# Decision Variables Extraction Notebook Plan

## Structure Overview
Create a Jupyter notebook: `decision_variables_extraction.ipynb` with the following sections:

### 1. Introduction & Setup (Markdown + Code)
- Project context and GLM objectives
- Import necessary libraries (numpy, scipy, sklearn, matplotlib, seaborn)
- Set random seeds for reproducibility
- Define file paths and parameters

### 2. Data Loading & Exploration (Code + Markdown)
- Load A324_pycells_20230727.mat file
- Examine data structure and fields
- Extract spike times, behavioral choices, trial timing
- Display basic statistics (n_trials, n_neurons, session duration)

### 3. Data Preprocessing (Code + Markdown)
- Apply neuron quality filters (spatial spread, peak width, presence ratio, etc.)
- Calculate smoothed firing rates using 50ms Gaussian kernel
- Align data to stimulus onset with time bins (-0.5s to +1.5s, 50ms steps)
- Format data as (n_neurons, n_timepoints, n_trials) array

### 4. Session Quality Control (Code + Markdown)
- Validate session meets paper criteria (≥300 trials, ≤8% lapse rate)
- Check choice balance and behavioral performance
- Visualize trial structure and choice distribution

### 5. Model Training & Cross-Validation (Code + Markdown)
- Implement 10-fold stratified cross-validation
- Find optimal regularization parameter for each time point
- Calculate geometric mean of regularization parameters
- Refit models using constant regularization across time

### 6. Decision Variables Calculation (Code + Markdown)
- Extract decision variables using fitted logistic regression models
- Calculate choice prediction accuracy over time
- Apply sign correction for visualization purposes
- Store results in structured format

### 7. Results Visualization (Code + Markdown)
- Plot choice prediction accuracy evolution over time
- Visualize average DV trajectories by choice type
- Show single-trial examples
- Display model weights heatmap over time

### 8. Validation & Quality Assessment (Code + Markdown)
- Compare results against paper benchmarks
- Validate temporal evolution patterns
- Check for overfitting indicators
- Generate performance summary statistics

### 9. Export & Summary (Code + Markdown)
- Save processed decision variables to file
- Create summary report of key findings
- Document next steps for GLM analysis

## Notebook Structure Guidelines

### Formatting Requirements (Sections 2-9)
- **Separate heading cells**: Each heading level (H1, H2, H3) goes in its own markdown cell
- **Section numbering**: Number all sections (2., 2.1, 2.2, etc.)
- **Concise explanations**: Reduce markdown text - focus on essential information only
- **No incomplete functions**: Only include complete, implemented functions (no signatures)
- **No completion summaries**: Avoid "section complete" or similar ending paragraphs
- **Brief session details**: Keep result summaries concise but informative

### Implementation Notes
- Focus on code execution over extensive documentation
- Essential parameters and methodology references only
- Clean, navigable structure with minimal verbose explanations

## Key Features
- Proper error handling and data validation
- Publication-quality visualizations
- Modular functions following the methodology document
- Performance metrics and validation checks
- Clear documentation for reproducibility