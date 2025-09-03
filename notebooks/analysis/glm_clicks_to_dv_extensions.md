# GLM Analysis Extensions and Improvements

## Critical Improvements for GLM Implementation

### 1. Temporal Cross-Validation Strategy
**Issue**: Random train/test split breaks temporal structure and trial dependencies.

**Solution**:
```python
# Option A: Time-based split preserving trial structure
split_trial = 192  # 80% of 240 trials
train_mask = dv_df_240['trial_id'] < split_trial
test_mask = dv_df_240['trial_id'] >= split_trial

# Option B: Block cross-validation
def block_cv_split(trial_ids, n_blocks=5):
    """Create temporal blocks for cross-validation"""
    trials_per_block = len(trial_ids) // n_blocks
    for i in range(n_blocks):
        test_trials = trial_ids[i*trials_per_block:(i+1)*trials_per_block]
        train_trials = np.setdiff1d(trial_ids, test_trials)
        yield train_trials, test_trials
```

### 2. Vectorized Feature Creation
**Issue**: Nested loops for feature creation are computationally inefficient.

**Solution**:
```python
def create_features_vectorized(click_times_left, click_times_right, 
                               eval_times, centers, widths):
    """
    Vectorized feature creation using broadcasting
    Returns: (n_timepoints, n_features) matrix
    """
    # Create time lag matrices using broadcasting
    eval_times = eval_times[:, np.newaxis]
    click_left = click_times_left[np.newaxis, :]
    click_right = click_times_right[np.newaxis, :]
    
    # Compute all time lags at once
    lags_left = eval_times - click_left  # (n_eval, n_clicks_left)
    lags_right = eval_times - click_right  # (n_eval, n_clicks_right)
    
    # Apply Gaussian basis functions
    features = []
    for center, width in zip(centers, widths):
        # Sum over clicks for each eval time
        feat_left = np.sum(gaussian_basis(lags_left, center, width), axis=1)
        feat_right = np.sum(gaussian_basis(lags_right, center, width), axis=1)
        features.extend([feat_left, feat_right])
    
    return np.array(features).T
```

### 3. Enhanced Temporal Features
**Extension**: Add biologically-motivated features beyond Gaussian basis.

```python
def create_enhanced_features(click_df, eval_time, trial_id):
    """Generate additional temporal features"""
    features = {}
    
    # 1. Running evidence with exponential decay
    tau_evidence = 0.5  # decay time constant
    evidence = 0
    for _, click in click_df[click_df['trial_id'] == trial_id].iterrows():
        if click['click_time'] < eval_time:
            decay = np.exp(-(eval_time - click['click_time']) / tau_evidence)
            evidence += decay * (1 if click['side'] == 'right' else -1)
    features['running_evidence'] = evidence
    
    # 2. Click rate in sliding windows
    windows = [0.1, 0.25, 0.5, 1.0]  # seconds
    for window in windows:
        mask = (click_df['click_time'] > eval_time - window) & \
               (click_df['click_time'] < eval_time)
        features[f'click_rate_{window}s'] = len(click_df[mask]) / window
    
    # 3. Time since last click (each side)
    last_left = click_df[(click_df['side'] == 'left') & 
                         (click_df['click_time'] < eval_time)]['click_time'].max()
    last_right = click_df[(click_df['side'] == 'right') & 
                          (click_df['click_time'] < eval_time)]['click_time'].max()
    features['time_since_left'] = eval_time - last_left if not np.isnan(last_left) else np.inf
    features['time_since_right'] = eval_time - last_right if not np.isnan(last_right) else np.inf
    
    # 4. Asymmetric temporal kernels (alpha function)
    def alpha_kernel(t, tau_rise, tau_decay):
        return (np.exp(-t/tau_decay) - np.exp(-t/tau_rise)) * (t > 0)
    
    # Different rise/decay times for more biological realism
    features['alpha_fast'] = alpha_kernel(lags, 0.01, 0.05)
    features['alpha_slow'] = alpha_kernel(lags, 0.1, 0.5)
    
    return features
```

### 4. Model Comparison Framework
**Extension**: Systematic comparison of regularization strategies.

```python
def compare_regularization_methods(X_train, y_train, X_test, y_test):
    """Compare Ridge, Elastic Net, and Lasso"""
    results = {}
    
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]  # Ridge to Lasso spectrum
    
    for alpha in alphas:
        fit = cvglmnet(
            x=X_train, 
            y=y_train,
            family='gaussian',
            alpha=alpha,
            nfolds=10,
            standardize=True
        )
        
        y_pred = glmnetPredict(fit['glmnet_fit'], newx=X_test, 
                               s=fit['lambda_min']).flatten()
        
        results[alpha] = {
            'name': f"{'Lasso' if alpha==1 else 'Ridge' if alpha==0 else f'ElasticNet(α={alpha})'}",
            'r2': calculate_r2(y_test, y_pred),
            'rmse': np.sqrt(np.mean((y_test - y_pred)**2)),
            'n_nonzero': np.sum(fit['glmnet_fit']['beta'][:, -1] != 0),
            'lambda_min': fit['lambda_min'],
            'fit': fit
        }
    
    return results
```

### 5. Comprehensive Residual Diagnostics
**Extension**: Identify systematic model failures.

```python
def residual_diagnostics(y_true, y_pred, trial_ids, time_bins):
    """Comprehensive residual analysis"""
    residuals = y_true - y_pred
    
    # 1. Autocorrelation of residuals
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=20, return_df=True)
    
    # 2. Per-trial R²
    trial_r2 = {}
    for trial in np.unique(trial_ids):
        mask = trial_ids == trial
        if mask.sum() > 10:  # Need enough points
            trial_r2[trial] = calculate_r2(y_true[mask], y_pred[mask])
    
    # 3. Heteroscedasticity test (Breusch-Pagan)
    from scipy import stats
    _, p_value = stats.normaltest(residuals)
    
    # 4. Residuals vs time within trial
    time_residuals = pd.DataFrame({
        'time': time_bins,
        'residual': residuals,
        'trial': trial_ids
    })
    
    # 5. Check for systematic patterns
    patterns = {
        'autocorrelation': lb_test['lb_pvalue'].min() < 0.05,
        'heteroscedastic': p_value < 0.05,
        'trial_variation': np.std(list(trial_r2.values())) > 0.2,
        'temporal_trend': np.abs(np.corrcoef(time_bins, residuals)[0,1]) > 0.1
    }
    
    return {
        'ljung_box': lb_test,
        'trial_r2': trial_r2,
        'patterns': patterns,
        'recommendations': generate_recommendations(patterns)
    }
```

### 6. Adaptive Basis Function Design
**Extension**: Data-driven basis function placement and width.

```python
def adaptive_basis_design(click_data, n_basis=20):
    """Design basis functions based on click statistics"""
    
    # Compute inter-click intervals
    all_clicks = np.sort(click_data['click_time'].values)
    intervals = np.diff(all_clicks)
    
    # Place centers at quantiles of interval distribution
    quantiles = np.linspace(0, 100, n_basis)
    centers = np.percentile(intervals, quantiles)
    
    # Adaptive widths based on local density
    widths = []
    for i, center in enumerate(centers):
        # Width proportional to distance to neighbors
        if i == 0:
            width = (centers[1] - centers[0]) / 2
        elif i == len(centers) - 1:
            width = (centers[-1] - centers[-2]) / 2
        else:
            width = (centers[i+1] - centers[i-1]) / 4
        widths.append(width)
    
    return centers, np.array(widths)
```

### 7. Sparse Matrix Optimization
**Extension**: Memory and computation efficiency for large datasets.

```python
from scipy.sparse import csr_matrix, hstack

def build_sparse_feature_matrix(click_df, dv_df, centers, widths):
    """Build sparse feature matrix for efficiency"""
    
    feature_blocks = []
    
    for basis_idx, (center, width) in enumerate(zip(centers, widths)):
        # Create sparse matrix for this basis function
        rows, cols, data = [], [], []
        
        for i, (_, dv_row) in enumerate(dv_df.iterrows()):
            # Only store non-zero entries
            features = compute_basis_features(...)
            for j, val in enumerate(features):
                if abs(val) > 1e-10:  # Threshold for sparsity
                    rows.append(i)
                    cols.append(j)
                    data.append(val)
        
        sparse_block = csr_matrix((data, (rows, cols)), 
                                  shape=(len(dv_df), 2))  # 2 for left/right
        feature_blocks.append(sparse_block)
    
    # Horizontally stack sparse matrices
    X_sparse = hstack(feature_blocks)
    return X_sparse
```

### 8. Bootstrap Confidence Intervals
**Extension**: Assess coefficient stability and significance.

```python
def bootstrap_coefficient_cis(X, y, n_bootstrap=100, confidence=0.95):
    """Compute bootstrap confidence intervals for GLM coefficients"""
    
    n_samples = len(X)
    n_features = X.shape[1]
    coef_samples = np.zeros((n_bootstrap, n_features))
    
    for b in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        
        # Fit model
        fit_boot = cvglmnet(x=X_boot, y=y_boot, family='gaussian', 
                            alpha=1.0, nfolds=5)
        
        # Extract coefficients at optimal lambda
        best_idx = np.argmin(np.abs(fit_boot['lambdau'] - fit_boot['lambda_min']))
        coef_samples[b] = fit_boot['glmnet_fit']['beta'][:, best_idx]
    
    # Compute confidence intervals
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(coef_samples, alpha * 100, axis=0)
    ci_upper = np.percentile(coef_samples, (1 - alpha) * 100, axis=0)
    ci_mean = np.mean(coef_samples, axis=0)
    ci_std = np.std(coef_samples, axis=0)
    
    # Identify significant coefficients (CI doesn't include 0)
    significant = (ci_lower > 0) | (ci_upper < 0)
    
    return {
        'mean': ci_mean,
        'std': ci_std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': significant,
        'samples': coef_samples
    }
```

### 9. Advanced Feature Importance Analysis
**Extension**: Beyond raw coefficients to understand feature contributions.

```python
def feature_importance_analysis(model, X, y, feature_names):
    """Comprehensive feature importance metrics"""
    
    # 1. Standardized coefficients
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    std_importance = model.coef_ * X_std.std(axis=0)
    
    # 2. Permutation importance
    baseline_score = model.score(X, y)
    perm_importance = []
    for i in range(X.shape[1]):
        X_perm = X.copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        perm_score = model.score(X_perm, y)
        perm_importance.append(baseline_score - perm_score)
    
    # 3. Partial dependence
    def partial_dependence(feature_idx, n_points=50):
        X_pd = X.copy()
        feature_range = np.linspace(X[:, feature_idx].min(), 
                                   X[:, feature_idx].max(), n_points)
        pd_values = []
        for val in feature_range:
            X_pd[:, feature_idx] = val
            pd_values.append(model.predict(X_pd).mean())
        return feature_range, pd_values
    
    # 4. SHAP-like local explanations
    def local_explanation(x_instance):
        """Decompose prediction into feature contributions"""
        baseline_pred = model.predict(X.mean(axis=0).reshape(1, -1))[0]
        instance_pred = model.predict(x_instance.reshape(1, -1))[0]
        
        contributions = {}
        for i, name in enumerate(feature_names):
            x_modified = X.mean(axis=0).copy()
            x_modified[i] = x_instance[i]
            contrib = model.predict(x_modified.reshape(1, -1))[0] - baseline_pred
            contributions[name] = contrib
        
        return contributions, instance_pred - baseline_pred
    
    return {
        'standardized': std_importance,
        'permutation': perm_importance,
        'partial_dependence': partial_dependence,
        'local_explanation': local_explanation
    }
```

### 10. Temporal Predictability Analysis
**Extension**: Understand when the model becomes predictive within trials.

```python
def temporal_predictability_analysis(dv_df, y_true, y_pred):
    """Analyze R² as a function of time within trial"""
    
    # Group by time bins
    time_bins = np.unique(dv_df['time_bin'])
    cumulative_r2 = []
    instantaneous_r2 = []
    
    for t in time_bins:
        # Cumulative R² up to time t
        mask_cumul = dv_df['time_bin'] <= t
        if mask_cumul.sum() > 10:
            r2_cumul = calculate_r2(y_true[mask_cumul], y_pred[mask_cumul])
            cumulative_r2.append(r2_cumul)
        else:
            cumulative_r2.append(np.nan)
        
        # Instantaneous R² at time t (within small window)
        window = 0.1  # 100ms window
        mask_inst = np.abs(dv_df['time_bin'] - t) < window/2
        if mask_inst.sum() > 10:
            r2_inst = calculate_r2(y_true[mask_inst], y_pred[mask_inst])
            instantaneous_r2.append(r2_inst)
        else:
            instantaneous_r2.append(np.nan)
    
    # Find critical time point where model becomes predictive
    threshold_r2 = 0.1  # Minimum R² to consider predictive
    predictive_times = time_bins[np.array(cumulative_r2) > threshold_r2]
    critical_time = predictive_times[0] if len(predictive_times) > 0 else np.nan
    
    return {
        'time_bins': time_bins,
        'cumulative_r2': cumulative_r2,
        'instantaneous_r2': instantaneous_r2,
        'critical_time': critical_time,
        'max_r2': np.nanmax(cumulative_r2),
        'final_r2': cumulative_r2[-1]
    }
```

## Implementation Priority

### High Priority (Core Improvements)
1. Temporal cross-validation (#1)
2. Vectorized feature creation (#2)
3. Residual diagnostics (#5)
4. Bootstrap confidence intervals (#8)

### Medium Priority (Enhanced Analysis)
5. Enhanced temporal features (#3)
6. Model comparison framework (#4)
7. Temporal predictability analysis (#10)

### Low Priority (Optimization & Advanced)
8. Sparse matrix optimization (#7)
9. Adaptive basis design (#6)
10. Advanced feature importance (#9)

## Next Steps

1. Implement high-priority improvements first
2. Test on first 240 trials as planned
3. Validate improvements with synthetic data where ground truth is known
4. Scale to full dataset once core methodology is solid
5. Document computational requirements and runtime for each enhancement