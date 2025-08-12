# Decision Variables Extraction - Section 5 Implementation Plan

## Overview

This document outlines the implementation plan for Section 5 of the decision variables extraction notebook (`decision_variables_extraction.ipynb`). Section 5 completes the "Validation & Results Export" phase as specified in the original `dv_plan.md`.

## Section 5: Advanced Analysis & GLM Preparation

### 5.1 Extended Benchmark Validation
- **Objective**: Implement comprehensive comparison against paper benchmarks
- **Tasks**:
  - Compare results against Figure 1f-h patterns from methodology paper
  - Add statistical significance tests for accuracy vs chance using permutation tests
  - Validate DV magnitude ranges against expected values (0.5-2.0 log-odds units)
  - Check temporal evolution patterns match paper expectations (monotonic increase)
  - Verify peak accuracy timing occurs after stimulus onset (>0.5s)
  - Test regional performance differences against paper benchmarks

### 5.2 Overfitting Detection & Analysis  
- **Objective**: Ensure model generalization and validate regularization approach
- **Tasks**:
  - Implement cross-validation performance assessment with held-out test sets
  - Add permutation tests to verify prediction accuracy significance
  - Analyze temporal consistency of model weights across time points
  - Validate geometric mean regularization approach stability
  - Check for suspicious patterns (>95% accuracy, sudden jumps)
  - Assess regularization parameter stability (CV < 0.5)

### 5.3 Regional Analysis Preparation
- **Objective**: Prepare region-specific analysis for GLM modeling
- **Tasks**:
  - Group neurons by brain region (ADS, M1, M2, NAc, etc.)
  - Calculate region-specific decision variables and prediction accuracies
  - Export region-wise performance summaries for GLM analysis
  - Document regional performance differences and peak timing variations
  - Prepare region-based DV time series for click-DV correlation analysis

### 5.4 Results Export & Documentation
- **Objective**: Export processed data and create comprehensive documentation
- **Tasks**:
  - Export decision variables in formats suitable for GLM analysis (.npz, .csv)
  - Create comprehensive performance summary with key statistics
  - Generate publication-quality figures following paper standards
  - Export timing alignment information for click data integration
  - Save model parameters and regularization results
  - Document essential findings and validation results

### 5.5 GLM Preparation & Session Summary
- **Objective**: Complete handoff to GLM analysis phase
- **Tasks**:
  - Format decision variables for click-DV GLM analysis
  - Align DV time series with behavioral event timing (cpoke_in reference)
  - Create final session validation report with pass/fail criteria
  - Prepare handoff documentation with key parameters and results
  - Export session metadata and quality control metrics
  - Generate summary plots for GLM analysis planning

## Implementation Guidelines

### Output Philosophy (from dv_plan.md)
- Minimize print statements to essential information only (data shapes, key statistics, validation results)
- Focus on code execution over extensive documentation
- Brief, technical markdown explanations for methodology
- Output essential performance metrics only

### Code Structure Requirements
- Direct data examination over generic patterns
- Simple, exploratory code appropriate for research notebook
- Complete functions only (no incomplete signatures)
- Professional formatting without decorative elements

### Validation Criteria
Based on methodology paper benchmarks:
- **Baseline accuracy**: >0.5 (above chance at trial start)
- **Peak accuracy**: >0.6 (reaching 60%+ as in paper Figure 1h)
- **Temporal evolution**: Monotonic increase with peak >0.5s after stimulus
- **DV magnitude**: Standard deviation >0.5 log-odds units
- **Choice consistency**: Correlation with choices >0.3
- **Regional differences**: Peak accuracy 0.55-0.8 depending on region

### Expected Regional Performance (from methodology)
- **Strongest regions**: M1, dmFC, ADS (peak accuracy ~0.75-0.8)
- **Moderate regions**: mPFC, S1 (peak accuracy ~0.65-0.7)  
- **Weaker regions**: HPC, BLA (peak accuracy ~0.55-0.65)

### File Export Specifications
- **Decision variables**: (n_timepoints, n_trials) arrays in .npz format
- **Metadata**: JSON format with session parameters and validation results
- **Figures**: PNG format, publication quality (300 DPI)
- **Time alignment**: Explicit time_bins arrays for GLM synchronization
- **Regional summaries**: CSV format for easy GLM import

### Session Quality Control
- Must pass all validation criteria from methodology paper
- Session must have ≥300 trials and ≤8% lapse rate
- Neural data must pass quality filters (no unknown regions, CC excluded)
- Model performance must exceed chance with statistical significance
- Regularization must be stable (λ coefficient of variation < 0.5)

## Key References

- **Primary methodology**: `decision_variables_methodology.md` - Complete implementation details
- **Original plan**: `dv_plan.md` - Section 5 specifications
- **Paper reference**: "Brain-wide coordination of decision formation and commitment" (Bondy et al., 2025 Draft)
- **Implementation**: `decision_variables_extraction.ipynb` - Sections 1-4 completed

## Success Criteria

Section 5 is complete when:
1. ✅ All benchmark validations pass with statistical significance
2. ✅ No overfitting indicators detected
3. ✅ Regional analysis prepared for GLM phase
4. ✅ All results exported in specified formats
5. ✅ Comprehensive documentation generated
6. ✅ GLM preparation materials ready

## Next Steps After Section 5

Upon completion, the project will be ready for:
- GLM implementation linking click inputs to decision variables
- Cross-session comparison of DV extraction results
- Publication of decision variables extraction methodology validation
- Integration with broader ClickDV project analysis pipeline