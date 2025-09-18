# sklearn vs python-glmnet Comparison and Fix Plan

## Problem Summary

We discovered that python-glmnet's cross-validation is selecting 0 features (maximum regularization) while sklearn's LassoCV properly selects ~8-10 features with better predictive performance. Investigation revealed:

1. **CV metric issue**: python-glmnet returns negative CV scores (likely -R² or negative log-likelihood), but the code uses `argmin()` which incorrectly selects the worst model
2. **Standardization difference**: sklearn uses standardized features while our glmnet uses `standardize=False`
3. **Performance gap**: sklearn achieves correlation ~0.5-0.6 while glmnet with min lambda gets ~0.42

## Root Cause

The main issue is that python-glmnet should handle regularization selection internally, but either:
- The package has a bug in CV interpretation
- We're not using it correctly (wrong parameters or attributes)

## Solution Plan

### 1. Fix Model Initialization
```python
model = ElasticNet(
    alpha=0.95,  # Lasso penalty
    n_lambda=100,
    min_lambda_ratio=1e-3,
    standardize=True,  # <- CHANGE: Match sklearn's standardization
    fit_intercept=True,
    n_splits=10,
    random_state=42
)
```

### 2. Let glmnet Handle CV Internally
- Remove manual `argmin(cv_mean)` lambda selection
- Use glmnet's built-in optimal lambda selection
- Check for attributes like `model.lambda_best_` or `model.lambda_opt_`

### 3. Interpret Model Results
After fitting, examine:
- `model.lambda_path_`: Understand regularization path
- `model.cv_mean_score_`: Check what metric is optimized (MSE vs negative LL)
- `model.coef_path_`: See how features are selected across lambdas
- Compare best CV, 1SE, and minimum lambda results

### 4. Fair Method Comparison
With both methods using standardization:
- Compare CV-selected regularization levels
- Check if prediction correlations with DV are now similar
- Understand remaining performance differences

### 5. Scientific Interpretation
- Which temporal scales (10ms to 500ms) are most important?
- How does regularization affect temporal filter shapes?
- Do left and right clicks show different temporal dynamics?

## Key Questions to Answer

1. **Why are CV scores negative?** Is glmnet optimizing -R² or negative log-likelihood instead of MSE?
2. **Does standardization matter scientifically?** Should we preserve natural feature scales (click counts) or normalize?
3. **Why the correlation discrepancy?** With similar predictions between methods, why different target correlations?

## Expected Outcomes

After implementing this plan:
- python-glmnet should properly select ~10-15 features via CV
- Performance should be comparable to sklearn (~0.5-0.6 correlation)
- We'll understand what metric glmnet actually optimizes
- Clear interpretation of which temporal features matter most

## Implementation Notes

- Keep all diagnostic/interpretation code to understand the model
- Let glmnet do computations automatically (no manual CV fixes)
- Focus on understanding and visualizing what the model learned
- Compare standardized vs unstandardized to understand scientific implications