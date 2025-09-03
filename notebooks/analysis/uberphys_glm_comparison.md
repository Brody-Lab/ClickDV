# Comparison: UberPhys GLM Implementation vs ClickDV GLM Plan

## UberPhys Project Overview
UberPhys is a large-scale, multi-probe Neuropixels recording project during the Poisson Clicks task. It focuses on neural spike data analysis across multiple brain regions in rats performing decision-making tasks.

## Key Similarities

### 1. **Same Experimental Paradigm**
- Both projects analyze data from the **Poisson Clicks task**
- Both work with rat behavioral data (A324, A327, C211)
- Both connect click inputs to neural/decision variables

### 2. **GLM Framework**
- Both use GLMs to link stimulus (clicks) to neural responses
- Both projects originated from the Brody-Daw lab

## Key Differences

### 1. **Target Variable**
| Aspect | UberPhys | ClickDV |
|--------|----------|---------|
| **Predicts** | Individual neuron spike trains | Population-level decision variables (DVs) |
| **Scale** | Single-cell GLMs | Population decoding GLMs |
| **Output** | Spike rates/counts | Continuous decision variable |

### 2. **Implementation Language**
| Aspect | UberPhys | ClickDV |
|--------|----------|---------|
| **Language** | MATLAB | Python |
| **GLM Package** | neuroGLM/MATLAB glmnet | Python glmnet-py |
| **Data Format** | .mat Cells files | HDF5 (.h5) files |

### 3. **GLM Technical Details**

#### UberPhys Approach:
```matlab
% From generate_kernels.m
[stats,p,spikes] = fit_glm_to_Cells(joined_Cells,
    'fit_method','neuroglmfit',
    'bin_size_s',0.01,
    'kfold',1,
    'lambda',30,  % Fixed lambda value
    'alpha',0,    % Ridge regression (L2)
    'separate_clicks_by_commitment',true);
```

**Key features:**
- Uses `fit_glm_to_Cells` function from labwide_PBups_analysis repo
- Requires neuroGLM and neuro_glmnet MATLAB packages
- Fixed lambda=30 (no cross-validation shown)
- **alpha=0** indicates Ridge regression (L2), not Lasso
- Separates clicks by commitment time (pre/post decision)
- 10ms bin size

#### ClickDV Plan:
```python
# From our plan
fit = cvglmnet(
    x=X_train, 
    y=y_train,
    family='gaussian',
    alpha=1.0,  # Pure Lasso (L1)
    nfolds=10,  # 10-fold cross-validation
    standardize=True,
    intr=True
)
```

**Key features:**
- Uses cvglmnet with built-in cross-validation
- **alpha=1.0** for pure Lasso (L1) regularization
- Automatic lambda selection via CV
- Gaussian basis functions for temporal representation
- Predicts continuous DV values

### 4. **Basis Functions**

| Aspect | UberPhys | ClickDV Plan |
|--------|----------|--------------|
| **Temporal Basis** | Not explicitly shown in code | 20 Gaussian basis functions |
| **Time Lags** | Implicit in neuroGLM | Log-spaced 10ms - 1s |
| **Click Representation** | Separate left/right kernels | Convolved with Gaussian basis |

### 5. **Data Processing Pipeline**

#### UberPhys:
1. Load Cells files with `uberphys.load_Cells_files()`
2. Join multiple probe data with `join_cells_files()`
3. Apply quality criteria with `uberphys.apply_inclusion_criteria()`
4. Import anatomical info with `uberphys.import_implant_table_to_Cells()`
5. Fit GLMs per neuron

#### ClickDV:
1. Load HDF5 file with `load_session_data()`
2. Extract DataFrames (trials, clicks, DVs)
3. Create Gaussian basis features
4. Build feature matrix X
5. Fit single GLM for DV prediction

## Important Insights for ClickDV Implementation

### 1. **Commitment Time Consideration**
UberPhys separates clicks into pre/post commitment periods. This could be valuable for ClickDV:
- Consider adding commitment time as a feature
- Potentially fit separate models for pre/post commitment

### 2. **Multi-Region Analysis**
UberPhys analyzes specific brain regions (dmFC, M1, ADS, mPFC, NAc). ClickDV could:
- Compare GLM weights across different recording sessions
- Investigate region-specific click integration

### 3. **Regularization Difference**
- UberPhys uses Ridge (L2) with fixed lambda
- ClickDV plans Lasso (L1) with CV-selected lambda
- **Recommendation**: Try both L1 and L2, compare sparsity vs smoothness

### 4. **Missing Basis Function Implementation**
The UberPhys code doesn't show explicit basis function implementation, suggesting it's handled within neuroGLM. For ClickDV:
- Our Gaussian basis approach is more explicit and interpretable
- Consider comparing with raised cosine basis (common in neuroGLM)

## Recommendations for ClickDV

1. **Start Simple**: Implement the planned Gaussian basis + Lasso approach first
2. **Add Complexity Gradually**:
   - Test Ridge (alpha=0) vs Lasso (alpha=1)
   - Try elastic net (0 < alpha < 1)
   - Experiment with different basis function designs
3. **Leverage UberPhys Insights**:
   - Consider commitment time splitting
   - Implement similar quality criteria for data filtering
4. **Cross-Validation**: Stick with the planned CV approach rather than fixed lambda
5. **Validation**: Compare results with sklearn LogisticRegression baseline from v003 notebook

## Code Availability
- UberPhys GLM: Requires MATLAB repos (neuroGLM, labwide_PBups_analysis, neuro_glmnet)
- ClickDV: Self-contained Python implementation with glmnet-py

This comparison shows that while both projects analyze similar data, ClickDV takes a population-level approach to understanding decision variables, while UberPhys focuses on single-neuron responses. The ClickDV plan's use of explicit Gaussian basis functions and Lasso regularization should provide interpretable temporal filters for understanding click integration dynamics.