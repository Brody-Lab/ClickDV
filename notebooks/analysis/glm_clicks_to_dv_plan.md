# GLM Plan: Connecting Clicks to Decision Variables using glmnet with Lasso

## Overview
Build a GLM to predict decision variables DV(t) from click input data for the first 240 trials of session A324_2023-07-27, using Gaussian basis functions to represent click history and the python glmnet package with Lasso (L1) regularization.

## Key Concept: glmnet's Built-in Cross-Validation
**glmnet doesn't need manual train/test splitting!** The `cvglmnet` function:
- Automatically performs k-fold cross-validation internally
- Selects optimal regularization (lambda) via CV
- Returns a model trained on ALL provided data
- Provides CV performance metrics for evaluation

## Step-by-Step Plan

### 1. ✅ Create New Analysis Notebook [COMPLETED]
- Notebook already created at `notebooks/analysis/glm_clicks_to_dv_glmnet.ipynb`

### 2. ✅ Load and Prepare Data [COMPLETED]
- Data loading function implemented using pd.HDFStore
- Filter to first 240 trials implemented

### 3. Design Gaussian Basis Functions
Create a set of Gaussian kernels to capture click history at different timescales:
```python
# Define basis function parameters
n_basis = 20  # Number of Gaussian basis functions
tau_min = 0.01  # Minimum time lag (10ms)
tau_max = 1.0   # Maximum time lag (1000ms)

# Log-spaced centers for basis functions
centers = np.logspace(np.log10(tau_min), np.log10(tau_max), n_basis)

# Width of each Gaussian (can be proportional to spacing)
widths = 0.4 * np.diff(np.concatenate([[0], centers]))

def gaussian_basis(t, center, width):
    """Gaussian basis function"""
    return np.exp(-(t - center)**2 / (2 * width**2))
```

### 4. Convolve Clicks with Basis Functions
For each trial and time point, create feature vectors:
```python
def create_click_features(left_clicks, right_clicks, eval_time, centers, widths):
    """
    Create feature vector by convolving click history with Gaussian basis
    
    Returns:
    - features: (2*n_basis,) array - first n_basis for left, next n_basis for right
    """
    n_basis = len(centers)
    features = np.zeros(2 * n_basis)
    
    # Process left and right clicks
    for side_idx, clicks in enumerate([left_clicks, right_clicks]):
        # Get clicks before eval_time
        past_clicks = clicks[clicks < eval_time]
        
        # For each basis function
        for i, (center, width) in enumerate(zip(centers, widths)):
            # Time lags from eval_time
            lags = eval_time - past_clicks
            # Sum of Gaussian responses
            features[side_idx * n_basis + i] = np.sum(gaussian_basis(lags, center, width))
    
    return features
```

### 5. Build Feature Matrix X
For each DV time point, construct features:
- **2n_basis features**: n_basis for left clicks, n_basis for right clicks
- **Optional additional features**:
  - Difference features (right - left) for each basis
  - Cumulative click counts
  - Time since trial start
  
```python
# Build feature matrix
X = []
y = []
valid_mask = []

for trial_id in range(240):
    # Get clicks for this trial
    trial_clicks = click_df_240[click_df_240['trial_id'] == trial_id]
    left_clicks = trial_clicks[trial_clicks['click_side'] == 'left']['click_time']
    right_clicks = trial_clicks[trial_clicks['click_side'] == 'right']['click_time']
    
    # Get DVs for this trial
    trial_dvs = dv_df_240[dv_df_240['trial_id'] == trial_id]
    
    for _, dv_row in trial_dvs.iterrows():
        eval_time = dv_row['time_bin']
        
        # Create Gaussian basis features
        features = create_click_features(
            left_clicks, right_clicks, eval_time, centers, widths
        )
        X.append(features)
        y.append(dv_row['decision_variable'])
        valid_mask.append(dv_row['is_valid'])

X = np.array(X)
y = np.array(y)
valid_mask = np.array(valid_mask)

# Keep only valid DV points
X_valid = X[valid_mask]
y_valid = y[valid_mask]
```

### 6. Implement GLM with glmnet's Built-in Cross-Validation
**NO MANUAL TRAIN/TEST SPLIT NEEDED!** glmnet handles this internally:
```python
from glmnet_python import cvglmnet, glmnetPredict

# cvglmnet performs k-fold CV automatically on ALL your data
# No need to split beforehand - it does this internally!
fit = cvglmnet(
    x=X_valid,  # ALL your valid data - cvglmnet splits it internally
    y=y_valid,  # ALL your valid labels
    family='gaussian',  # For continuous DV prediction
    alpha=1.0,  # Pure Lasso (L1 penalty)
    nfolds=10,  # Automatically creates 10 folds internally
    standardize=True,  # glmnet standardizes features automatically
    intr=True  # Include intercept
)

# What cvglmnet did internally:
# 1. Split X_valid into 10 folds
# 2. For each lambda value:
#    - Train on 9 folds, test on 1 fold
#    - Rotate through all 10 folds
#    - Average the performance
# 3. Select best lambda based on CV performance
# 4. Retrain final model on ALL data with best lambda

# The fit object contains:
# - fit['lambdau']: Full lambda sequence tested
# - fit['cvm']: Cross-validated MSE for each lambda
# - fit['cvsd']: Standard error of CV MSE
# - fit['lambda_min']: Lambda with minimum CV error
# - fit['lambda_1se']: Conservative lambda (1 SE rule)
# - fit['glmnet_fit']: Final model trained on ALL data

print(f"Optimal lambda (min CV error): {fit['lambda_min']:.6f}")
print(f"Conservative lambda (1SE rule): {fit['lambda_1se']:.6f}")
print(f"CV MSE at optimal lambda: {fit['cvm'][fit['lambda_min_idx']]:.4f}")

# Extract coefficients at optimal lambda
best_lambda_idx = np.where(fit['lambdau'] == fit['lambda_min'])[0][0]
best_coefs = fit['glmnet_fit']['beta'][:, best_lambda_idx]

# Count non-zero coefficients (automatic feature selection by Lasso)
n_selected = np.sum(best_coefs != 0)
print(f"Selected {n_selected}/{len(best_coefs)} features")
```

### 7. Model Evaluation Using CV Results
```python
# The CV performance is already computed by glmnet!
best_idx = np.where(fit['lambdau'] == fit['lambda_min'])[0][0]
cv_mse = fit['cvm'][best_idx]
cv_std = fit['cvsd'][best_idx]
print(f"Cross-validated MSE: {cv_mse:.4f} ± {cv_std:.4f}")
print(f"Cross-validated RMSE: {np.sqrt(cv_mse):.4f}")

# Optional: For additional validation, create a holdout set
# This is ONLY if you want extra confidence beyond CV results
if False:  # Set to True if you want holdout validation
    # Create holdout split BEFORE running cvglmnet
    np.random.seed(42)
    n_total = len(X_valid)
    n_model = int(0.8 * n_total)
    indices = np.random.permutation(n_total)
    
    X_model = X_valid[indices[:n_model]]
    y_model = y_valid[indices[:n_model]]
    X_holdout = X_valid[indices[n_model:]]
    y_holdout = y_valid[indices[n_model:]]
    
    # Run cvglmnet on model set only
    fit_holdout = cvglmnet(x=X_model, y=y_model, family='gaussian', 
                           alpha=1.0, nfolds=10, standardize=True, intr=True)
    
    # Evaluate on holdout
    y_pred_holdout = glmnetPredict(
        fit_holdout['glmnet_fit'], 
        newx=X_holdout, 
        s=fit_holdout['lambda_min']
    ).flatten()
    
    holdout_mse = np.mean((y_holdout - y_pred_holdout)**2)
    print(f"Holdout MSE: {holdout_mse:.4f}")

# Make predictions on all data to visualize fit quality
y_pred_all = glmnetPredict(
    fit['glmnet_fit'], 
    newx=X_valid, 
    s=fit['lambda_min']
).flatten()

# Calculate R² using glmnet's CV approach
# Note: glmnet uses deviance ratio which is similar to R²
# For Gaussian family, deviance ratio ≈ R²
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

r2_apparent = calculate_r2(y_valid, y_pred_all)
print(f"Apparent R² (on all data): {r2_apparent:.3f}")
print(f"Note: This is optimistic. Use CV MSE for unbiased estimate.")
```

### 8. Visualizations
```python
# 1. Cross-validation curve (glmnet already computed this!)
plt.figure(figsize=(8, 5))
plt.errorbar(np.log(fit['lambdau']), fit['cvm'], yerr=fit['cvsd'], 
             alpha=0.7, label='CV MSE ± 1 std')
plt.axvline(np.log(fit['lambda_min']), color='r', linestyle='--', label='λ_min')
plt.axvline(np.log(fit['lambda_1se']), color='g', linestyle='--', label='λ_1se')
plt.xlabel('log(Lambda)')
plt.ylabel('Mean-Squared Error')
plt.title('Cross-Validation Curve (Computed by glmnet)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Regularization path (glmnet computed full path automatically)
plt.figure(figsize=(10, 6))
for i in range(fit['glmnet_fit']['beta'].shape[0]):
    plt.plot(np.log(fit['lambdau']), fit['glmnet_fit']['beta'][i, :], alpha=0.5)
plt.axvline(np.log(fit['lambda_min']), color='r', linestyle='--', label='λ_min')
plt.xlabel('log(Lambda)')
plt.ylabel('Coefficient Value')
plt.title('Lasso Path - Feature Selection by glmnet')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Learned temporal filters
left_coefs = best_coefs[:n_basis]
right_coefs = best_coefs[n_basis:2*n_basis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.stem(centers, left_coefs)
ax1.set_xlabel('Time lag (s)')
ax1.set_ylabel('Weight')
ax1.set_title('Left Click Filter (Selected by Lasso)')
ax1.grid(True, alpha=0.3)

ax2.stem(centers, right_coefs)
ax2.set_xlabel('Time lag (s)')
ax2.set_ylabel('Weight')
ax2.set_title('Right Click Filter (Selected by Lasso)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()

# 4. Model fit quality
plt.figure(figsize=(8, 8))
plt.scatter(y_valid, y_pred_all, alpha=0.3, s=1)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 
         'r--', label='Perfect prediction')
plt.xlabel('Actual DV')
plt.ylabel('Predicted DV')
plt.title(f'Model Fit (Apparent R² = {r2_apparent:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
```

### 9. Optional: Alternative Regularization with glmnet
```python
# glmnet makes it easy to compare different regularization approaches!

# 1. Compare Lasso vs Ridge vs Elastic Net
alphas = [1.0, 0.5, 0.0]  # Lasso, Elastic Net, Ridge
alpha_names = ['Lasso (α=1)', 'Elastic Net (α=0.5)', 'Ridge (α=0)']

for alpha, name in zip(alphas, alpha_names):
    fit_alpha = cvglmnet(x=X_valid, y=y_valid, family='gaussian', 
                         alpha=alpha, nfolds=10, standardize=True, intr=True)
    print(f"{name}: CV MSE = {fit_alpha['cvm'][np.where(fit_alpha['lambdau'] == fit_alpha['lambda_min'])[0][0]]:.4f}")

# 2. Use lambda_1se for more regularization (built into glmnet!)
conservative_coefs = fit['glmnet_fit']['beta'][:, np.where(fit['lambdau'] == fit['lambda_1se'])[0][0]]
n_selected_conservative = np.sum(conservative_coefs != 0)
print(f"Conservative model: {n_selected_conservative} features (vs {n_selected} with lambda_min)")
```

## Key Implementation Details

### Understanding glmnet's Built-in Features:

#### Automatic Cross-Validation (`cvglmnet`):
- **No manual train/test split needed** - cvglmnet handles k-fold CV internally
- Automatically generates lambda sequence
- Provides CV performance metrics (cvm, cvsd)
- Returns both lambda_min and lambda_1se
- Final model is trained on ALL data with optimal lambda

#### Feature Standardization:
- glmnet automatically standardizes when `standardize=True`
- Coefficients are returned on original scale
- No need to manually standardize features

#### Regularization Path:
- glmnet computes entire regularization path efficiently
- Stores all coefficients for all lambda values
- Enables easy comparison of different regularization levels

#### Feature Selection:
- Lasso (alpha=1.0) automatically selects features by setting coefficients to exactly zero
- Number of selected features depends on lambda value
- lambda_1se provides more aggressive feature selection than lambda_min

### Why Use glmnet's Built-in CV Instead of Manual Splitting?
1. **Efficiency**: Uses warm starts across lambda values
2. **Consistency**: Ensures same folds used for all lambdas
3. **Automatic**: Handles fold creation, rotation, and averaging
4. **Complete**: Provides standard errors for model selection
5. **Best Practice**: Final model uses all available data

### Expected Outputs
1. Sparse GLM model with automatically selected temporal features
2. Cross-validation curve from glmnet's internal CV
3. Identification of critical time lags via Lasso feature selection
4. CV-based performance metrics (unbiased estimates)
5. Clear visualization of learned temporal filters

This approach leverages glmnet's sophisticated built-in functionality rather than reimplementing standard ML practices.