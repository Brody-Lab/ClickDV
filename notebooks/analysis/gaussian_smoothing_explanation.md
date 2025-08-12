# Gaussian Smoothing for Neural Decision Variables: Detailed Explanation

## Overview

This document explains the Gaussian smoothing methodology implemented in Section 3.3 of the decision variables extraction notebook. Gaussian smoothing is the most critical preprocessing step that transforms discrete neural spike events into smooth, continuous firing rates suitable for decision variable analysis.

## Why Gaussian Smoothing is Essential

### The Problem with Raw Spike Times

- **Discrete events**: Neural spikes are binary events - a neuron either fires (1) or doesn't (0) at any millisecond
- **Extreme noise**: Raw spike data appears as random 0s and 1000s (when converted to Hz)
- **Sparse data**: Most time points have no spikes, making analysis difficult
- **Decision formation**: Biological decision-making is a gradual process, but raw spikes look like random noise
- **Analysis failure**: Logistic regression on raw spikes would fit to noise rather than meaningful decision signals

### The Solution: Gaussian Smoothing

- **Continuous firing rates**: Converts discrete spike events into smooth, continuous firing rate signals
- **Temporal correlations**: Creates biologically realistic correlations between adjacent time points
- **Population dynamics**: Allows detection of coordinated activity across neural populations
- **Decision signals**: Enables capture of gradual decision formation over time

## The Algorithm Step-by-Step

### 1. Parameters Setup

```python
dt = 0.001  # 1ms resolution for convolution
sigma_samples = sigma_ms / 1000 / dt  # Convert 50ms to 50 samples
```

**Parameter choices:**
- **1ms resolution**: High enough to avoid aliasing, fine enough for accurate convolution
- **50ms sigma**: From Uberphys paper methodology - captures neural population dynamics at optimal timescale

### 2. For Each Neuron and Each Trial

#### Step 2a: Create Extended Time Window

```python
trial_start_extended = trial_start + time_bins[0] - 0.25  # 250ms buffer before
trial_end_extended = trial_start + time_bins[-1] + 0.25   # 250ms buffer after
```

**Why buffers are critical:**
- Gaussian kernel has extended tails (±3 standard deviations)
- Without buffers, edge effects would distort data at trial boundaries
- 250ms buffer ensures ~5 standard deviations (5×50ms) of Gaussian are included

#### Step 2b: Build High-Resolution Spike Train

```python
time_axis = np.arange(trial_start_extended, trial_end_extended, dt)  # 1ms resolution
spike_train = np.zeros(len(time_axis))  # Initialize to zeros

for spike_time in trial_spikes:
    spike_idx = int((spike_time - trial_start_extended) / dt)
    if 0 <= spike_idx < len(spike_train):
        spike_train[spike_idx] = 1.0 / dt  # = 1000 Hz
```

**Process explanation:**
- Create 1ms-resolution time axis for extended trial window
- Initialize all time points to 0 Hz
- For each actual spike: set that millisecond to 1000 Hz (converting "1 spike in 1ms" to rate)
- Result: sparse binary array with 1000 Hz spikes, 0 Hz elsewhere

### 3. Apply Gaussian Smoothing (The Critical Step)

```python
smoothed = ndimage.gaussian_filter1d(spike_train, sigma=sigma_samples)
```

**What happens during convolution:**
- Input: Noisy spike train `[0, 0, 1000, 0, 0, 1000, 0, ...]`
- Gaussian kernel: Bell curve centered on each point
- Output: Smooth, continuous firing rate curve

**Gaussian kernel properties:**
- Each spike gets "spread out" over ±150ms (3 standard deviations)
- Multiple nearby spikes add together constructively  
- Result: smooth curve that peaks where spikes are dense, low where spikes are sparse

### 4. Sample at Analysis Time Points

```python
for time_idx, rel_time in enumerate(time_bins):
    abs_time = trial_start + rel_time
    sample_idx = int((abs_time - trial_start_extended) / dt)
    firing_rates[neuron_idx, time_idx, trial_idx] = smoothed[sample_idx]
```

- Sample the smooth curve at analysis time points (-0.5s to +1.5s, every 50ms)
- Store in final array format: (neurons, timepoints, trials)
- Discard the high-resolution intermediate data

## Mathematical Details

### Gaussian Kernel Formula

```
G(t) = (1/√(2πσ²)) × exp(-t²/(2σ²))
```

Where:
- σ = 50ms standard deviation
- t = time offset from spike
- **Effect**: Each spike influences firing rate for ~150ms around it (±3σ)

### Convolution Operation

Mathematical representation:
```
firing_rate(t) = Σ(spikes(τ) × G(t-τ))
```

**Physical interpretation:**
- For each time point t, firing rate is weighted sum of nearby spikes
- Weights follow Gaussian distribution: highest for nearby spikes, lower for distant
- Creates smooth temporal profile around each spike event

### Temporal Correlation Structure

After smoothing, adjacent time points are correlated:
```
Correlation(t, t+Δt) ≈ exp(-Δt²/(2σ²))
```

This creates the smooth temporal evolution essential for decision variable analysis.

## Parameter Justification

### 50ms Standard Deviation

**Why not smaller (e.g., 10ms)?**
- Still too noisy for population analysis
- Doesn't capture neural population coordination timescales
- Logistic regression would still fit to noise

**Why not larger (e.g., 200ms)?**
- Over-smoothed, loses temporal precision of decision formation
- Can't resolve rapid changes in decision signals
- Reduces statistical power for detecting dynamic effects

**Why 50ms is optimal:**
- Matches timescale of neural population dynamics from literature
- Balances noise reduction with temporal resolution
- Validated in Uberphys paper across multiple brain regions

### 1ms Convolution Resolution

- **Nyquist requirement**: Must be much finer than 50ms smoothing kernel
- **Accuracy**: Ensures accurate convolution without aliasing artifacts
- **Computational**: Fine enough for precision, not so fine as to be wasteful

### 50ms Sampling Interval

- **Nyquist compliance**: With 50ms smoothing, can sample at 50ms without information loss
- **Statistical power**: Balances temporal resolution with statistical robustness
- **Computational efficiency**: Reduces final data size while preserving all information

## Before/After Transformation Examples

### Raw Spike Train Example
```
Time (ms):    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
Spikes:       0   0   1   0   0   0   1   0   0   0   0   1   0   0   0   0
Rate (Hz):    0   0 1000  0   0   0 1000  0   0   0   0 1000  0   0   0   0
```

### After Gaussian Smoothing
```
Time (ms):    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
Rate (Hz):    5  15  45  35  25  35  55  45  35  45  35  65  45  35  25  15
```

### Key Properties of Smoothed Data

1. **Smooth transitions**: No sudden jumps from 0 to 1000 Hz
2. **Temporal correlations**: Adjacent time points have similar values
3. **Realistic magnitudes**: Typical firing rates 1-50 Hz instead of 0/1000 Hz
4. **Preserved timing**: Peaks still occur near original spike times
5. **Population coordination**: Multiple neurons can show correlated smooth changes

## Connection to Decision Variables

### How Smoothing Enables Decision Variable Analysis

**Gradual Decision Formation:**
- **Raw spikes**: Random-looking pattern of 0s and 1000s
- **Smoothed rates**: Gradual changes that can track decision formation over time
- **Logistic regression**: Can detect patterns in smooth temporal evolution

**Population Coordination:**
- Multiple neurons with correlated smooth firing rates
- Allows detection of coordinated population activity during decisions
- Essential for finding decision-related signals that span neural populations

**Temporal Consistency:**
- 50ms smoothing creates biologically realistic temporal correlations
- Decision variables evolve smoothly over time (as expected from neurobiology)
- Prevents overfitting to random spike timing fluctuations

### Statistical Requirements

**Why continuous variables are essential:**
- **Logistic regression assumption**: Requires continuous predictor variables
- **Cross-validation stability**: Smooth signals provide stable model fits across folds
- **Generalization**: Smooth patterns generalize better than noise fits

**Population decoding:**
- Decision variables represent population-level signals
- Individual spike timing is noisy, but population firing rates are informative
- Gaussian smoothing extracts the population signal while suppressing individual neuron noise

## Implementation Notes

### Computational Considerations

- **Memory usage**: High-resolution convolution requires significant memory
- **Processing time**: Most computationally expensive step in preprocessing
- **Parallelization**: Could be parallelized across neurons for speed improvement

### Quality Control

**Validation checks after smoothing:**
- Firing rates should be positive (Gaussian smoothing preserves this)
- Typical values should be 0.1-50 Hz (biologically realistic)
- No NaN or infinite values
- Smooth temporal evolution without sudden jumps

### Edge Case Handling

- **No spikes in window**: Results in near-zero firing rates (correct)
- **Very high firing rates**: Gaussian smoothing naturally handles without saturation
- **Irregular trial lengths**: Extended buffers ensure consistent smoothing at edges

## Scientific Validation

### Literature Support

This methodology follows the exact approach from:
- **Bondy et al. (2025)**: "Brain-wide coordination of decision formation and commitment"
- **Standard practice**: Used across computational neuroscience literature
- **Cross-species validation**: Effective in rodents, primates, and humans

### Expected Results

After proper Gaussian smoothing:
- **Temporal evolution**: Smooth increase/decrease in firing rates over trials
- **Population coordination**: Multiple neurons showing correlated changes
- **Decision prediction**: Firing rate patterns should predict behavioral choices
- **Cross-validation stability**: Consistent results across different data subsets

## Conclusion

Gaussian smoothing with a 50ms kernel is the critical preprocessing step that transforms noisy, discrete neural spike data into the smooth, continuous firing rate signals required for decision variable analysis. This transformation:

1. **Preserves biological information** while reducing noise
2. **Creates temporal correlations** that match neural population dynamics
3. **Enables population decoding** through logistic regression
4. **Provides statistical stability** for cross-validation and model fitting

Without proper Gaussian smoothing, decision variable extraction would fail because logistic regression would fit to spike timing noise rather than meaningful decision-related population signals. The 50ms kernel represents the optimal balance between noise reduction and temporal resolution for capturing neural decision formation dynamics.