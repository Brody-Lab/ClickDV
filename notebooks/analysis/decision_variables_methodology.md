# Neural Decision Variables: Complete Implementation Guide

## Paper Reference
**Title**: "Brain-wide coordination of decision formation and commitment"  
**Authors**: Adrian G. Bondy*, Julie A. Charlton*, Thomas Zhihao Luo*, et al.  
**Date**: July 25, 2025 Draft  
**File Location**: `/home/ham/SF/Personal/Education/07 Princeton/Zotero/Princeton/Brody Lab/20250725 Draft - Brain-wide coordination of decision formation and commitment.pdf`  
**Key Section**: "Neural decoding of choice and calculation of 'decision variables'" (page 34)  
**Additional References**: 
- Methods section (page 30): "Subjects" - behavioral task exclusion criteria
- Page 34: Glmnet package usage and geometric mean regularization
- Figure 1f-h (page 3-4): Example results and expected patterns

## Overview
Decision variables (DVs) are 1-dimensional summaries of neural population activity that capture how the brain's representation of an upcoming behavioral choice evolves over time. This methodology uses logistic regression to decode choice from population firing rates and extract time-varying decision signals from neural data.

## Theoretical Background

### What are Decision Variables?
- **Definition**: A scalar time series DV(t,i) representing the neural population's "confidence" in making a particular choice on trial i at time t
- **Interpretation**: More positive values indicate stronger evidence for one choice (e.g., "go right"), more negative values indicate stronger evidence for the alternative choice (e.g., "go left")
- **Units**: Log-odds of choice probability, equivalent to `log(p_right / p_left)`

### Why Use This Method?
1. **Dimensionality Reduction**: Summarizes activity of hundreds of neurons into single time-varying signal
2. **Trial-by-trial Resolution**: Captures moment-to-moment decision formation process
3. **Cross-regional Comparison**: Enables comparison of decision signals across brain regions
4. **Internally Generated Signals**: Can reveal decision dynamics independent of stimulus or motor planning

## Detailed Methodology

### Step 1: Neural Data Preprocessing

#### Input Requirements
- **Spike times**: List of spike times for each neuron across all trials
- **Trial structure**: Start/end times for each trial, aligned to task events
- **Behavioral data**: Choice made by subject on each trial (binary: left/right, A/B, etc.)
- **Session quality**: Exclude sessions with <300 trials or >8% lapse rate (from Methods, page 30)

#### Firing Rate Calculation
```python
# Detailed implementation for Gaussian convolution
import numpy as np
from scipy import ndimage

def calculate_firing_rates(spike_times, trial_times, time_bins, sigma_ms=50):
    """
    Convolve spike times with Gaussian kernel and sample at regular intervals
    
    Parameters:
    - spike_times: dict of {neuron_id: [spike_times_in_seconds]}
    - trial_times: array of trial start times
    - time_bins: time points relative to trial start for sampling
    - sigma_ms: standard deviation of Gaussian kernel in milliseconds
    """
    from scipy import ndimage
    import numpy as np
    
    dt = 0.001  # 1ms resolution for convolution
    sigma_samples = sigma_ms / 1000 / dt  # Convert to samples
    n_neurons = len(spike_times)
    n_trials = len(trial_times)
    n_timepoints = len(time_bins)
    
    # Initialize firing rate array
    firing_rates = np.zeros((n_neurons, n_timepoints, n_trials))
    
    for neuron_idx, (neuron_id, spikes) in enumerate(spike_times.items()):
        for trial_idx, trial_start in enumerate(trial_times):
            
            # Define time window for this trial (extend beyond time_bins for edge effects)
            trial_duration = time_bins[-1] - time_bins[0] + 0.5  # Add 500ms buffer
            trial_start_extended = trial_start + time_bins[0] - 0.25
            trial_end_extended = trial_start + time_bins[-1] + 0.25
            
            # Create high-resolution time axis
            time_axis = np.arange(trial_start_extended, trial_end_extended, dt)
            
            # Create spike train (1 where spike occurs, 0 elsewhere)
            spike_train = np.zeros(len(time_axis))
            trial_spikes = spikes[(spikes >= trial_start_extended) & 
                                (spikes < trial_end_extended)]
            
            for spike_time in trial_spikes:
                spike_idx = int((spike_time - trial_start_extended) / dt)
                if 0 <= spike_idx < len(spike_train):
                    spike_train[spike_idx] = 1.0 / dt  # Convert to rate (Hz)
            
            # Apply Gaussian smoothing
            smoothed = ndimage.gaussian_filter1d(spike_train, sigma=sigma_samples)
            
            # Sample at desired time points
            for time_idx, rel_time in enumerate(time_bins):
                abs_time = trial_start + rel_time
                sample_idx = int((abs_time - trial_start_extended) / dt)
                
                if 0 <= sample_idx < len(smoothed):
                    firing_rates[neuron_idx, time_idx, trial_idx] = smoothed[sample_idx]
    
    return firing_rates

# Result: X(n,t,i) = firing rate of neuron n at time t on trial i
```

**Key Parameters**:
- **Smoothing kernel**: 50ms standard deviation symmetric Gaussian
- **Sampling interval**: 50ms (paper uses this consistently)
- **Time alignment**: Align to stimulus onset (most common), can also align to movement or commitment
- **Temporal correlation**: Adjacent time points will be correlated due to 50ms smoothing

#### Data Structure and Reshaping
```python
# X should be a 3D array: [neurons, time_points, trials]
# choices should be 1D array: [trials] with binary values (0/1 or -1/+1)
X.shape = (n_neurons, n_timepoints, n_trials)
choices.shape = (n_trials,)

# For sklearn, need to reshape to 2D for each time point:
# X_t.shape = (n_trials, n_neurons) for time point t
X_t = X[:, t, :].T  # Transpose to get trials x neurons
```

### Step 2: Quality Control and Neuron Selection

#### Inclusion Criteria (from paper Table 2, page 33)
Apply these filters to exclude low-quality units:

1. **Spatial spread** < 150 μm (spatial decay of waveform energy)
2. **Peak width** < 1 ms (width of main deflection at half height)
3. **Peak-trough width** < 1 ms (time from trough to peak)
4. **No upward-going spikes** (exclude positive deflections)
5. **Peak-to-peak voltage** > 50 μV
6. **Presence ratio** > 0.5 (fires ≥1 spike on ≥50% of trials)

#### Additional Filters
- Exclude obvious artifacts (noise, movement-related)
- Consider excluding neurons with extremely low or high baseline firing rates
- ~65% of Kilosort-detected units typically pass these criteria

### Step 3: Logistic Regression Model

#### Mathematical Formulation
For each time point t, fit a logistic regression model:

```
p(t,i)(R) = f(X(t,i) · β(t) + α(t))
```

Where:
- `p(t,i)(R)` = probability of rightward choice at time t on trial i
- `f` = sigmoid/logistic function: `f(x) = 1/(1 + exp(-x))`
- `X(t,i)` = vector of firing rates for all neurons at time t on trial i
- `β(t)` = weight vector for time t (learned parameters)
- `α(t)` = bias term for time t (learned parameter)

#### Objective Function
Maximize log-likelihood with L1 regularization across all trials:
```
argmax Σ(i=1 to N_trials) log p(C(i) | X(t,i), β(t), α(t)) - λ||β(t)||₁
```

Where:
- `C(i)` = actual choice on trial i (binary: 0 or 1)
- `λ` = L1 regularization parameter (found via cross-validation)
- This optimization is performed separately for each time point t
- The paper uses Glmnet package (MATLAB) equivalent to sklearn LogisticRegressionCV in Python

### Step 4: Cross-Validation and Regularization

#### Cross-Validation Procedure
1. **Type**: 10-fold stratified cross-validation
2. **Stratification**: Ensure balanced representation of both choice types in each fold
3. **Parameter search**: Find optimal λ for each time point independently
4. **Critical Step**: Use geometric mean of optimal λ values across all time points for final models
5. **Final fitting**: Refit all models using this constant λ value (crucial for temporal consistency)

#### Implementation Details
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import numpy as np

def find_optimal_regularization(X, choices):
    """
    Find optimal regularization parameter using geometric mean approach from paper
    """
    n_timepoints = X.shape[1]
    optimal_lambdas = []
    
    # Step 1: Find optimal λ for each time point
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for t in range(n_timepoints):
        X_t = X[:, t, :].T  # Reshape to (trials, neurons)
        
        model = LogisticRegressionCV(
            Cs=np.logspace(-4, 2, 50),  # Regularization range (C = 1/λ)
            cv=cv,
            penalty='l1',
            solver='liblinear',
            scoring='balanced_accuracy',
            max_iter=1000
        )
        model.fit(X_t, choices)
        optimal_lambdas.append(1.0 / model.C_[0])  # Convert C back to λ
    
    # Step 2: Calculate geometric mean (critical step from paper)
    geometric_mean_lambda = np.exp(np.mean(np.log(optimal_lambdas)))
    final_C = 1.0 / geometric_mean_lambda
    
    return final_C, optimal_lambdas

# Usage:
final_C, individual_lambdas = find_optimal_regularization(X, choices)
print(f"Using constant C = {final_C:.4f} across all time points")
```

### Step 5: Decision Variable Calculation

#### Core Formula
```
DV(t,i) = X(t,i) · β(t) + α(t)
```

This is mathematically equivalent to:
```
DV(t,i) = log(p(t,i)(R) / p(t,i)(L)) = logit(p(t,i)(R))
```

#### Complete Implementation
```python
def calculate_decision_variables(X, choices, final_C):
    """
    Calculate decision variables using fitted logistic regression models
    
    Parameters:
    - X: neural data (n_neurons, n_timepoints, n_trials)
    - choices: behavioral choices (n_trials,) binary 0/1
    - final_C: regularization parameter from geometric mean
    
    Returns:
    - DVs: decision variables (n_timepoints, n_trials)
    - models: fitted models for each time point
    - accuracies: prediction accuracy for each time point
    """
    n_neurons, n_timepoints, n_trials = X.shape
    DVs = np.zeros((n_timepoints, n_trials))
    models = []
    accuracies = []
    
    # Fit model for each time point using constant regularization
    for t in range(n_timepoints):
        X_t = X[:, t, :].T  # Shape: (trials, neurons)
        
        # Fit final model with constant C
        model = LogisticRegression(
            C=final_C, 
            penalty='l1', 
            solver='liblinear',
            max_iter=1000
        )
        model.fit(X_t, choices)
        models.append(model)
        
        # Calculate decision variables (log-odds)
        DVs[t, :] = model.decision_function(X_t)
        
        # Calculate prediction accuracy
        predictions = model.predict(X_t)
        from sklearn.metrics import balanced_accuracy_score
        accuracies.append(balanced_accuracy_score(choices, predictions))
    
    return DVs, models, np.array(accuracies)
```

#### Properties of DV
- **Positive values**: Favor one choice (e.g., rightward/choice=1)
- **Negative values**: Favor opposite choice (e.g., leftward/choice=0)  
- **Magnitude**: Confidence/strength of evidence for that choice
- **Zero crossing**: Point of indecision between choices
- **Scale**: Log-odds units, unbounded real numbers

#### Sign Correction for Visualization
```python
# For pooling left and right choice trials in visualization:
def sign_correct_DVs(DVs, choices):
    """
    Flip sign of DVs on left-choice trials for visualization
    This makes all 'correct' DVs positive regardless of choice direction
    """
    DVs_corrected = DVs.copy()
    left_trials = (choices == 0)
    DVs_corrected[:, left_trials] *= -1
    return DVs_corrected
```

### Step 6: Performance Metrics

#### Choice Prediction Accuracy
Use **balanced accuracy** to handle class imbalance (paper uses this consistently):
```python
# Balanced accuracy is the average of sensitivity and specificity
balanced_accuracy = 0.5 * (TPR + TNR)
                  = 0.5 * (TP/(TP+FN) + TN/(TN+FP))
                  
# Where:
# TPR = True Positive Rate (Sensitivity) = correct right choices / total right choices
# TNR = True Negative Rate (Specificity) = correct left choices / total left choices
```

#### Expected Performance Patterns (from Figure 1h, page 4)
- **Baseline**: ~0.5 (chance level) at trial start
- **Peak accuracy**: ~0.6-0.8 near choice execution (varies by brain region)
- **Temporal evolution**: Gradual monotonic increase over time
- **Regional differences**: Some regions (M1, dmFC, ADS) show higher peak accuracy

#### Cross-Validation Assessment
- Assess performance on held-out test sets within the cross-validation framework
- Report accuracy as function of time to visualize temporal evolution
- Validate that accuracy increases smoothly without sudden jumps (indicates proper regularization)

## Complete Implementation Workflow

### Master Function
```python
def calculate_decision_variables_full_pipeline(spike_times, choices, trial_times, 
                                             time_bins=None, sigma_ms=50):
    """
    Complete pipeline for decision variable calculation following paper methodology
    
    Parameters:
    - spike_times: dict of {neuron_id: [spike_times_in_seconds]}
    - choices: array of binary choices (0/1) for each trial
    - trial_times: array of trial start times
    - time_bins: time points relative to trial start (default: -0.5 to 1.5s)
    - sigma_ms: Gaussian smoothing kernel std (default: 50ms from paper)
    
    Returns:
    - DVs: decision variables (n_timepoints, n_trials)
    - accuracies: prediction accuracy over time
    - models: fitted logistic regression models
    """
    # Step 1: Quality control (implement neuron filtering here)
    filtered_neurons = apply_quality_filters(spike_times)
    
    # Step 2: Calculate firing rates
    if time_bins is None:
        time_bins = np.arange(-0.5, 1.51, 0.05)  # -500ms to +1500ms, 50ms steps
    
    X = calculate_firing_rates(filtered_neurons, trial_times, time_bins, sigma_ms)
    
    # Step 3: Find optimal regularization
    final_C, individual_lambdas = find_optimal_regularization(X, choices)
    
    # Step 4: Calculate decision variables
    DVs, models, accuracies = calculate_decision_variables(X, choices, final_C)
    
    return {
        'decision_variables': DVs,
        'prediction_accuracy': accuracies,
        'models': models,
        'regularization_C': final_C,
        'time_bins': time_bins,
        'n_neurons': X.shape[0],
        'n_trials': X.shape[2]
    }
```

## Implementation Considerations

### Computational Requirements
- **Memory**: Store firing rate arrays (can be large: ~GB for hundreds of neurons, thousands of trials)
- **Processing**: Fit separate model for each time point (~10-50 models depending on trial duration)
- **Parallelization**: Time points can be processed independently using joblib or multiprocessing
- **Typical runtime**: Minutes to hours depending on data size and number of neurons

### Session Quality Control (Critical - from Methods page 30)
```python
def validate_session_quality(choices, n_trials, lapse_rate=None):
    """
    Apply session-level exclusion criteria from paper
    """
    # Exclude sessions with < 300 trials
    if n_trials < 300:
        return False, "Insufficient trials"
    
    # Exclude sessions with lapse rate > 8%
    if lapse_rate is not None and lapse_rate > 0.08:
        return False, "High lapse rate"
    
    return True, "Session passes quality control"
```

### Common Issues and Solutions

#### 1. Class Imbalance
- **Problem**: Unequal number of left vs right choices
- **Solution**: Use balanced accuracy, stratified CV, can also use class_weight='balanced' in LogisticRegression

#### 2. Overfitting
- **Problem**: Perfect training accuracy, poor generalization
- **Solution**: The geometric mean regularization approach specifically addresses this
- **Warning signs**: Accuracy > 0.95, or sudden jumps in accuracy over time

#### 3. Temporal Dependencies
- **Problem**: Adjacent time points are not independent due to 50ms smoothing
- **Solution**: This is expected and intended behavior, use consistent regularization across time

#### 4. Poor Accuracy Evolution
- **Problem**: Accuracy doesn't increase over time or shows non-monotonic behavior  
- **Solution**: Check neuron quality, increase regularization, verify trial alignment

#### 5. Missing Data
- **Problem**: Some trials have missing neural data or behavioral responses
- **Solution**: Exclude incomplete trials rather than imputation (preserves statistical properties)

### Parameter Sensitivity Analysis

#### Smoothing Kernel Effects
```python
# Test different smoothing kernels (paper validates this approach)
def test_smoothing_sensitivity(X_raw, choices, sigma_values=[25, 50, 75, 100]):
    """
    Test effect of different Gaussian smoothing kernels on DV performance
    """
    results = {}
    for sigma in sigma_values:
        X_smoothed = apply_gaussian_smoothing(X_raw, sigma)
        final_C, _ = find_optimal_regularization(X_smoothed, choices)
        DVs, _, accuracies = calculate_decision_variables(X_smoothed, choices, final_C)
        results[sigma] = {
            'peak_accuracy': np.max(accuracies),
            'accuracy_evolution': accuracies
        }
    return results
```

#### Sampling Interval Effects
- **Paper uses**: 50ms intervals consistently
- **Sensitivity**: Smaller intervals (25ms) increase temporal resolution but also computational cost
- **Larger intervals** (100ms): May miss rapid dynamics but reduce overfitting

#### Regularization Strength Impact
- **Too weak**: Overfitting, perfect training accuracy, poor generalization
- **Too strong**: Underfitting, accuracy barely above chance
- **Geometric mean approach**: Balances these extremes automatically

### Validation Against Paper Results

#### Expected Performance Benchmarks (from Figure 1h)
```python
def validate_against_paper_benchmarks(accuracies, DVs, choices):
    """
    Check if results match expected patterns from paper
    """
    checks = {
        'baseline_accuracy': np.mean(accuracies[:5]) > 0.5,  # Above chance at start
        'peak_accuracy': np.max(accuracies) > 0.6,  # Should reach 60%+ 
        'monotonic_increase': np.all(np.diff(accuracies) >= -0.02),  # Allow small decreases
        'dv_magnitude': np.std(DVs) > 0.5,  # DVs should have substantial variance
        'choice_consistency': np.corrcoef(np.mean(DVs, axis=0), choices)[0,1] > 0.3
    }
    return checks

# Expected ranges from paper:
# - Peak accuracy: 0.6-0.8 (varies by brain region)
# - DV standard deviation: 0.5-2.0 (log-odds units)
# - Temporal evolution: Smooth increase over 1-2 seconds
```

#### Regional Performance Differences
- **Strongest regions** (paper): M1, dmFC, ADS (peak accuracy ~0.75-0.8)
- **Moderate regions**: mPFC, S1 (peak accuracy ~0.65-0.7)  
- **Weaker regions**: HPC, BLA (peak accuracy ~0.55-0.65)

### Expected Results

#### Typical Patterns
1. **Accuracy Evolution**: Choice prediction accuracy should gradually increase from chance (~0.5) toward choice execution
2. **DV Trajectories**: Average DV should ramp toward correct choice over time
3. **Single-trial Variability**: Individual trials show substantial fluctuations around mean (magnitude comparable to mean)
4. **Regional Differences**: Some brain regions may show stronger/earlier choice signals

#### Validation Checks
- Accuracy significantly above chance (p < 0.05, permutation test)
- Smooth temporal evolution of accuracy (no sudden jumps)
- DV trajectories consistent with behavioral choice patterns
- Cross-validation performance stable across folds (std < 0.05)

## Key References in Paper

### Primary Methodology Section
- **Page 34**: "Neural decoding of choice and calculation of 'decision variables'"
- **Methods section**: Complete implementation details

### Quality Control
- **Page 33, Table 2**: "Waveform-shape-based unit inclusion criteria"
- **Page 33**: "Neuronal selection" section

### Validation and Results
- **Figure 1f-h**: Example DV calculation and choice prediction accuracy
- **Page 4**: Description of choice prediction accuracy patterns

### Mathematical Details
- **Page 34**: Complete equations for logistic regression model
- **Page 34**: Cross-validation and regularization procedures

## Software Dependencies

### Required Python Packages
```python
import numpy as np
import scipy as sp
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
```

### Optional but Recommended
```python
import pandas as pd  # Data organization
import seaborn as sns  # Visualization
from joblib import Parallel, delayed  # Parallelization
```

## Output Format

### Decision Variables
- **Shape**: `(n_timepoints, n_trials)` array
- **Values**: Real numbers (log-odds scale)
- **Interpretation**: Positive = evidence for choice 1, negative = evidence for choice 0

### Model Parameters
- **β(t)**: Weight vectors for each time point `(n_neurons, n_timepoints)`  
- **α(t)**: Bias terms for each time point `(n_timepoints,)`
- **Accuracy**: Choice prediction accuracy over time `(n_timepoints,)`

### Visualization Examples

#### 1. Choice Prediction Accuracy Over Time
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy_evolution(time_bins, accuracies, region_name="Region"):
    """
    Plot choice prediction accuracy over time (reproduces Figure 1h pattern)
    """
    plt.figure(figsize=(8, 5))
    plt.plot(time_bins, accuracies, 'b-', linewidth=2, label=f'{region_name}')
    plt.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Chance')
    plt.xlabel('Time from stimulus onset (s)')
    plt.ylabel('Choice prediction accuracy')
    plt.title(f'Decision Variable Performance: {region_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.45, 0.85])
    return plt.gcf()
```

#### 2. Average DV Trajectories by Choice
```python
def plot_dv_trajectories(time_bins, DVs, choices, region_name="Region"):
    """
    Plot average DV evolution for left vs right choices
    """
    left_trials = (choices == 0)
    right_trials = (choices == 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot mean ± SEM for each choice
    dv_left_mean = np.mean(DVs[:, left_trials], axis=1)
    dv_left_sem = np.std(DVs[:, left_trials], axis=1) / np.sqrt(np.sum(left_trials))
    
    dv_right_mean = np.mean(DVs[:, right_trials], axis=1)
    dv_right_sem = np.std(DVs[:, right_trials], axis=1) / np.sqrt(np.sum(right_trials))
    
    plt.plot(time_bins, dv_left_mean, 'r-', linewidth=2, label='Left choice')
    plt.fill_between(time_bins, dv_left_mean - dv_left_sem, 
                     dv_left_mean + dv_left_sem, alpha=0.3, color='red')
    
    plt.plot(time_bins, dv_right_mean, 'b-', linewidth=2, label='Right choice')
    plt.fill_between(time_bins, dv_right_mean - dv_right_sem, 
                     dv_right_mean + dv_right_sem, alpha=0.3, color='blue')
    
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time from stimulus onset (s)')
    plt.ylabel('Decision variable (log-odds)')
    plt.title(f'Decision Variable Trajectories: {region_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()
```

#### 3. Single-Trial Examples
```python
def plot_single_trial_examples(time_bins, DVs, choices, n_examples=5):
    """
    Plot individual trial DV trajectories (reproduces Figure 1i style)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Select random trials of each choice type
    left_trials = np.where(choices == 0)[0]
    right_trials = np.where(choices == 1)[0]
    
    example_trials = np.concatenate([
        np.random.choice(left_trials, n_examples//2 + 1, replace=False),
        np.random.choice(right_trials, n_examples//2, replace=False)
    ])
    
    for i, trial_idx in enumerate(example_trials):
        if i >= len(axes):
            break
            
        ax = axes[i]
        choice_label = "Left" if choices[trial_idx] == 0 else "Right"
        color = 'red' if choices[trial_idx] == 0 else 'blue'
        
        ax.plot(time_bins, DVs[:, trial_idx], color=color, linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_title(f'Trial {trial_idx} ({choice_label} choice)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('DV')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(example_trials), len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    return fig
```

#### 4. Model Weights Visualization
```python
def plot_model_weights(models, time_bins, neuron_ids=None):
    """
    Visualize logistic regression weights over time
    """
    n_timepoints = len(models)
    n_neurons = len(models[0].coef_[0])
    
    # Extract weights across time
    weights = np.zeros((n_neurons, n_timepoints))
    for t, model in enumerate(models):
        weights[:, t] = model.coef_[0]
    
    plt.figure(figsize=(12, 8))
    
    # Plot heatmap of weights
    im = plt.imshow(weights, aspect='auto', cmap='RdBu_r', 
                   extent=[time_bins[0], time_bins[-1], 0, n_neurons])
    plt.colorbar(im, label='Logistic regression weight')
    plt.xlabel('Time from stimulus onset (s)')
    plt.ylabel('Neuron index')
    plt.title('Evolution of Logistic Regression Weights')
    
    # Add neurons with strongest weights
    max_weights = np.max(np.abs(weights), axis=1)
    top_neurons = np.argsort(max_weights)[-5:]  # Top 5 neurons
    
    for neuron_idx in top_neurons:
        plt.axhline(neuron_idx, color='white', linestyle='--', alpha=0.7, linewidth=1)
        if neuron_ids:
            plt.text(time_bins[-1]*0.02, neuron_idx, f'N{neuron_ids[neuron_idx]}', 
                    color='white', fontsize=8)
    
    return plt.gcf()
```

## Important Caveats and Considerations

### Temporal Smoothing Effects (Critical)
- **50ms Gaussian kernel** creates temporal correlations between adjacent time points
- **Expected behavior**: DVs at nearby time points will be similar due to smoothing
- **Not a bug**: This is intended to capture gradual evolution of decision signals
- **Analysis implication**: Don't treat adjacent time points as independent observations

### Movement and Motor Artifacts
- **Problem**: Decision variables may reflect motor planning rather than decision formation
- **Paper's approach**: Uses "uninstructed movements" as potential confound (mentioned in Discussion)
- **Mitigation**: Compare DV timing to movement onset, use early trial periods
- **Alternative alignment**: Align to stimulus rather than movement when possible

### Interpretation of DV Magnitude
- **Scale**: Log-odds units, theoretically unbounded
- **Typical range**: -3 to +3 (corresponding to ~95% confidence in choice)
- **Zero crossing**: Not necessarily the "moment of decision" - could reflect noise
- **Comparison across regions**: Magnitudes may differ due to population size, not decision strength

### Trial-by-Trial Variability (Key Finding)
- **Paper finding**: DV residual fluctuations have "magnitudes comparable to the mean itself" (page 5)
- **Implication**: Single trials can show dramatically different trajectories even with same stimulus
- **Biological significance**: Reflects internally-generated decision dynamics
- **Analysis consideration**: Don't expect clean, stereotyped trajectories on individual trials

### Cross-Regional Comparisons
- **Population size effects**: Larger populations may show artificially higher accuracy
- **Paper control**: "choice prediction accuracy differences between regions were not trivially due to population size differences" (page 4)
- **Recommendation**: Subsample to equal population sizes or use statistical controls

### Session Quality and Behavioral Performance
- **Critical filters**: <300 trials or >8% lapse rate sessions excluded
- **Behavioral consistency**: Method requires reliable choice behavior to extract decision signals
- **Poor performance**: If accuracy doesn't exceed chance, check behavioral task performance first

### Regularization Sensitivity
- **Geometric mean approach**: Specifically chosen to balance temporal consistency with performance
- **Alternative approaches**: Using different λ for each time point can lead to overfitting
- **Validation**: Cross-validation performance should be stable across folds

### Statistical Dependencies
- **Pseudo-replication**: Same neurons may be recorded across sessions
- **Temporal structure**: 50ms sampling with 50ms smoothing creates expected correlations
- **Cross-validation**: Properly handles trial-level dependencies but not neuron-level

This methodology provides a principled approach to extract decision-related signals from population neural activity, enabling analysis of how decisions unfold over time in the brain. However, careful attention to these caveats is essential for proper interpretation of results.