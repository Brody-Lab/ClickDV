# GLM Plan: Connecting Clicks to Decision Variables using glmnet with Lasso

## Overview
Build a GLM to predict decision variables DV(t) from click input data for the first 240 trials of session A324_2023-07-27, using Gaussian basis functions to represent click history and the python glmnet package with Lasso (L1) regularization.

## Step-by-Step Plan

### 1. Create New Analysis Notebook
- Create `notebooks/analysis/glm_clicks_to_dv_glmnet.ipynb`
- Import required libraries including glmnet

### 2. Load and Prepare Data (First 240 Trials)
- Load the H5 file using the `load_session_data()` function from v003 notebook
- Extract trial_df, click_df, and dv_df DataFrames
- Filter to first 240 trials:
  - `trial_df_240 = trial_df[trial_df['trial_id'] < 240]`
  - `click_df_240 = click_df[click_df['trial_id'] < 240]`
  - `dv_df_240 = dv_df[dv_df['trial_id'] < 240]`

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
def create_click_features(click_times, eval_time, centers, widths):
    """
    Create feature vector by convolving click history with Gaussian basis
    
    Returns:
    - features_left: (n_basis,) array of left click convolutions
    - features_right: (n_basis,) array of right click convolutions
    """
    features_left = np.zeros(n_basis)
    features_right = np.zeros(n_basis)
    
    # Get all clicks before eval_time
    past_clicks_left = click_times_left[click_times_left < eval_time]
    past_clicks_right = click_times_right[click_times_right < eval_time]
    
    # For each basis function
    for i, (center, width) in enumerate(zip(centers, widths)):
        # Time lags from eval_time
        lags_left = eval_time - past_clicks_left
        lags_right = eval_time - past_clicks_right
        
        # Sum of Gaussian responses
        features_left[i] = np.sum(gaussian_basis(lags_left, center, width))
        features_right[i] = np.sum(gaussian_basis(lags_right, center, width))
    
    return features_left, features_right
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
        feat_left, feat_right = create_click_features(
            left_clicks, right_clicks, eval_time, centers, widths
        )
        
        # Combine features
        features = np.concatenate([feat_left, feat_right])
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

### 6. Manual Train/Test Split
```python
# Simple train/test split without sklearn
np.random.seed(42)
n_samples = len(X_valid)
n_train = int(0.8 * n_samples)

# Random shuffle indices
indices = np.random.permutation(n_samples)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

# Split data
X_train = X_valid[train_idx]
y_train = y_valid[train_idx]
X_test = X_valid[test_idx]
y_test = y_valid[test_idx]

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

### 7. Implement GLM with glmnet (Lasso)
```python
from glmnet_python import cvglmnet, glmnetPredict

# Use cvglmnet for built-in cross-validation on training data
fit = cvglmnet(
    x=X_train, 
    y=y_train,
    family='gaussian',  # For continuous DV prediction
    alpha=1.0,  # Pure Lasso (L1 penalty)
    nfolds=10,  # 10-fold cross-validation
    standardize=True,  # Standardize features
    intr=True  # Include intercept
)

# The fit object contains:
# - lambda sequence (fit['lambdau'])
# - CV mean squared errors (fit['cvm'])
# - Optimal lambda values:
#   - fit['lambda_min']: lambda with minimum CV error
#   - fit['lambda_1se']: largest lambda within 1 SE of minimum

print(f"Optimal lambda (min CV error): {fit['lambda_min']:.6f}")
print(f"Conservative lambda (1SE rule): {fit['lambda_1se']:.6f}")

# Extract coefficients at optimal lambda
best_lambda = fit['lambda_min']
coef_indices = np.where(fit['lambdau'] == best_lambda)[0][0]
best_coefs = fit['glmnet_fit']['beta'][:, coef_indices]

# Count non-zero coefficients (selected features)
n_selected = np.sum(best_coefs != 0)
print(f"Selected {n_selected}/{len(best_coefs)} features")
```

### 8. Model Evaluation
```python
# Predict on test set
y_pred_test = glmnetPredict(
    fit['glmnet_fit'], 
    newx=X_test, 
    s=fit['lambda_min']
).flatten()

# Calculate metrics without sklearn
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Performance metrics
r2 = calculate_r2(y_test, y_pred_test)
mse = np.mean((y_test - y_pred_test)**2)
rmse = np.sqrt(mse)
corr = np.corrcoef(y_test, y_pred_test)[0, 1]

print(f"Test R²: {r2:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test Correlation: {corr:.3f}")

# Also evaluate on training set to check for overfitting
y_pred_train = glmnetPredict(
    fit['glmnet_fit'], 
    newx=X_train, 
    s=fit['lambda_min']
).flatten()

train_r2 = calculate_r2(y_train, y_pred_train)
print(f"Train R²: {train_r2:.3f}")
```

### 9. Visualizations
- **Basis functions**: Plot the Gaussian kernels
- **Cross-validation curve**: 
  ```python
  plt.errorbar(np.log(fit['lambdau']), fit['cvm'], yerr=fit['cvsd'])
  plt.axvline(np.log(fit['lambda_min']), color='r', linestyle='--', label='λ_min')
  plt.axvline(np.log(fit['lambda_1se']), color='g', linestyle='--', label='λ_1se')
  plt.xlabel('log(Lambda)')
  plt.ylabel('Mean-Squared Error')
  plt.legend()
  ```
- **Lasso path**: Coefficient trajectories from `fit['glmnet_fit']['beta']`
- **Learned temporal filters**: 
  ```python
  # Reshape coefficients to visualize temporal structure
  left_coefs = best_coefs[:n_basis]
  right_coefs = best_coefs[n_basis:2*n_basis]
  
  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
  plt.stem(centers, left_coefs)
  plt.xlabel('Time lag (s)')
  plt.ylabel('Weight')
  plt.title('Left click filter')
  
  plt.subplot(1, 2, 2)  
  plt.stem(centers, right_coefs)
  plt.xlabel('Time lag (s)')
  plt.ylabel('Weight')
  plt.title('Right click filter')
  ```
- **Prediction quality**: Actual vs predicted DVs scatter plot

### 10. Sensitivity Analysis
- Compare lambda_min vs lambda_1se (more regularized)
- Vary number of basis functions (10, 20, 30)
- Try different basis function widths
- Compare linear vs log-spaced centers

## Key Implementation Details

### No sklearn dependency:
- Manual train/test split with numpy
- Custom R² calculation
- All metrics computed with numpy only

### glmnet handles:
- Cross-validation internally
- Lambda sequence generation
- Standardization
- Regularization path computation

### Why Pure Lasso (alpha=1.0)?
- Produces sparse solutions (many exact zeros)
- Automatic feature selection
- More interpretable than ridge or elastic net
- Identifies most relevant time lags for click integration

### Advantages of Built-in CV:
- Automatic lambda sequence generation
- Proper CV fold management
- Returns both lambda_min and lambda_1se
- Includes standard errors for model selection

## Expected Outputs
1. Sparse GLM model with selected temporal features
2. Cross-validation curve showing optimal regularization
3. Identification of critical time lags for click integration
4. Quantitative performance metrics (R², MSE, correlation)
5. Clear visualization of temporal filters for left/right clicks

This approach minimizes external dependencies, using mainly numpy and glmnet's built-in functionality.