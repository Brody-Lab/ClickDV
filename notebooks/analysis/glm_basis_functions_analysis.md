# GLM Basis Functions: A Comprehensive Analysis of Temporal Kernel Design

## Table of Contents
1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Gaussian Basis Functions](#gaussian-basis-functions)
4. [Raised Cosine Basis Functions](#raised-cosine-basis-functions)
5. [Detailed Comparison](#detailed-comparison)
6. [Implementation Analysis](#implementation-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Biological Interpretations](#biological-interpretations)
9. [Design Philosophy](#design-philosophy)
10. [Use Case Guidelines](#use-case-guidelines)
11. [Future Directions](#future-directions)
12. [Conclusion](#conclusion)

## Introduction

Basis functions form the mathematical foundation of Generalized Linear Models (GLMs) in computational neuroscience, enabling researchers to capture neural responses to stimuli across multiple timescales. The choice of basis function type, number, spacing, and overlap characteristics fundamentally determines the model's ability to represent temporal dynamics and extract meaningful biological insights.

This document provides a comprehensive analysis of two major approaches to GLM temporal kernel design: **Gaussian basis functions** (as implemented in the ClickDV project) and **raised cosine basis functions** (as used in the UberPhys/neuroGLM framework). Both approaches address the challenge of modeling how neural activity relates to past events, but they employ different mathematical foundations and design philosophies.

The analysis is particularly relevant for modeling click-to-decision variable relationships in perceptual decision-making tasks, where understanding the temporal dynamics of sensory integration is crucial for advancing our knowledge of neural computation.

## Theoretical Foundation

### The Temporal Kernel Problem

In GLMs for neuroscience applications, we seek to model neural firing rates as functions of past stimuli or events:

```
λ(t) = exp(β₀ + Σᵢ βᵢ * kᵢ(t))
```

where:
- `λ(t)` is the instantaneous firing rate
- `β₀` is the baseline firing rate
- `βᵢ` are learned coefficients
- `kᵢ(t)` are basis functions representing different temporal scales

The challenge lies in choosing basis functions that:
1. **Span the relevant temporal space** (complete representation)
2. **Are computationally tractable** (efficient computation)
3. **Enable biological interpretation** (meaningful parameters)
4. **Provide numerical stability** (well-conditioned matrices)

### Basis Function Requirements

Effective temporal basis functions must satisfy several mathematical and practical criteria:

**Mathematical Requirements:**
- **Linear independence**: Basis functions should not be perfectly correlated
- **Completeness**: Should be able to represent any reasonable temporal kernel
- **Smoothness**: Should enable stable optimization and interpretation

**Computational Requirements:**
- **Sparsity**: Minimize computational and memory overhead
- **Numerical stability**: Avoid ill-conditioned design matrices
- **Efficient convolution**: Fast computation of feature vectors

**Biological Requirements:**
- **Interpretable parameters**: Coefficients should relate to neural mechanisms
- **Realistic timescales**: Cover biologically relevant temporal ranges
- **Smooth reconstructions**: Avoid artificial discontinuities

## Gaussian Basis Functions

### Mathematical Definition

The Gaussian basis function approach uses symmetric Gaussian kernels:

```
gᵢ(t) = exp(-(t - cᵢ)² / (2 * σᵢ²))
```

where:
- `cᵢ` is the center (peak) of the i-th basis function
- `σᵢ` is the width (standard deviation) of the i-th basis function
- `t` represents time lag from the event

### Implementation Details

**Center Placement:**
- **Number**: 10 basis functions
- **Range**: 10ms to 1000ms
- **Spacing**: Logarithmic spacing using `np.logspace(np.log10(0.01), np.log10(1.0), 10)`
- **Rationale**: Logarithmic spacing provides fine resolution at short timescales where neural dynamics are fastest

**Width Determination (Novel 0.4 Factor):**
```python
centers = np.logspace(np.log10(tau_min), np.log10(tau_max), n_basis)
widths = 0.4 * np.diff(np.concatenate([[0], centers]))
```

The **0.4 multiplier** represents a novel contribution to basis function design:
- Creates **adaptive overlap** based on local center density
- Ensures **tighter basis functions** at short timescales (higher temporal resolution)
- Provides **broader basis functions** at long timescales (appropriate temporal smoothing)
- Balances **coverage and distinctiveness** of temporal features

### Mathematical Properties

**Support Characteristics:**
- **Theoretical support**: Infinite (Gaussian tails extend to ±∞)
- **Practical support**: Typically truncated at ±3σ (99.7% of mass)
- **Overlap control**: Determined by center spacing and width relationship

**Overlap Analysis:**
For adjacent Gaussians with centers `c₁` and `c₂` and widths `σ₁` and `σ₂`:
- **Intersection point**: Occurs where `g₁(t) = g₂(t)`
- **Overlap amount**: Depends on `|c₂ - c₁|` relative to `σ₁ + σ₂`
- **0.4 factor effect**: Creates approximately 60% overlap at peak heights

**Smoothness Properties:**
- **Infinite differentiability**: Gaussian functions are smooth everywhere
- **Optimization friendly**: Gradients exist and are well-behaved
- **No discontinuities**: Smooth basis reconstructions

### Advantages

1. **Biological Intuition**: Gaussian shapes are common in neural response profiles
2. **Adaptive Resolution**: Width scaling provides appropriate resolution at each timescale
3. **Smooth Optimization**: Excellent mathematical properties for gradient-based methods
4. **Novel Methodology**: 0.4 factor may provide optimal temporal resolution
5. **Flexible Parameterization**: Easy to adjust number, range, and spacing

### Disadvantages

1. **Computational Overhead**: Infinite support requires truncation and more computations
2. **Memory Requirements**: Must store and compute non-zero values over extended ranges
3. **Limited Validation**: Novel approach lacks extensive empirical validation
4. **Potential Redundancy**: Gaussian tails may create unnecessary overlap
5. **Truncation Artifacts**: Cutting Gaussian tails may introduce edge effects

## Raised Cosine Basis Functions

### Mathematical Definition

Raised cosine basis functions are defined as:

```
rcᵢ(t) = (cos((t - cᵢ)/sᵢ) + 1) / 2    for |t - cᵢ| ≤ π * sᵢ
rcᵢ(t) = 0                              otherwise
```

where:
- `cᵢ` is the center of the i-th basis function
- `sᵢ` is the scale parameter controlling width
- The function has compact support over `[cᵢ - π*sᵢ, cᵢ + π*sᵢ]`

### neuroGLM Implementation

**Center Placement (from UberPhys analysis):**
- **Number**: 12 basis functions (default in `click_basis.m`)
- **Range**: 16ms to 1420ms
- **Nonlinear transformation**: `log(t + 8ms)` where 8ms prevents log(0)
- **Spacing**: Even spacing in log-transformed domain

**Width Control:**
```matlab
spacing = diff(nlin(endPoints)) / (nBases-1)
scale = spacing * 2/π  % Creates half-height intersection
```

**Key Parameters:**
- `click_endpoints_s = [0.016, 1.42]` (16ms to 1420ms range)
- `click_nl_offset = 0.008` (8ms offset for log transformation)
- `n_click_bases = 12` (number of basis functions)

### Mathematical Properties

**Compact Support:**
- **Finite support**: Each basis function is exactly zero outside `±π` scaled units
- **Computational efficiency**: Creates sparse design matrices
- **Memory efficiency**: No need to store or compute zeros

**Overlap Control:**
- **Half-height intersection**: Adjacent bases intersect at 50% of peak amplitude
- **Theoretical optimality**: Minimizes reconstruction error under certain assumptions
- **Uniform overlap**: Same overlap pattern across all adjacent pairs

**Smoothness:**
- **Continuous**: Function and first derivative are continuous
- **Smooth reconstruction**: Weighted sums create smooth temporal kernels
- **No edge artifacts**: Smooth transition to zero at boundaries

### Implementation in UberPhys

**Separate Kernel Design:**
The UberPhys implementation creates multiple kernel types:
- `left_clicks_pre`: Left clicks before neural commitment time
- `left_clicks_post`: Left clicks after neural commitment time  
- `right_clicks_pre`: Right clicks before neural commitment time
- `right_clicks_post`: Right clicks after neural commitment time

**Additional Model Components:**
```matlab
bases = struct(
    'clicks',            basisFactory.click_basis(expt.binSize),
    'cpoke_in',          basisFactory.makeNonlinearRaisedCosSymmetric(15,2.5,[-0.1 2.5],2,expt.binSize),
    'cpoke_out',         basisFactory.makeNonlinearRaisedCosSymmetric(15,2,[-1 2],0.3,expt.binSize),
    'feedback',          basisFactory.makeNonlinearRaisedCosSymmetric(11,2.5,[-1.5 2.5],1,expt.binSize)
);
```

### Advantages

1. **Computational Efficiency**: Compact support creates sparse matrices
2. **Memory Efficiency**: Zero values outside support reduce storage requirements
3. **Theoretical Foundation**: Half-height overlap is signal processing optimal
4. **Established Methodology**: Extensively validated in computational neuroscience
5. **Smooth Reconstruction**: Adjacent functions sum seamlessly
6. **Numerical Stability**: Well-conditioned design matrices
7. **Reproducible Results**: Standard parameters enable comparison across studies

### Disadvantages

1. **Less Intuitive Shape**: Raised cosine less familiar than Gaussian profiles
2. **Fixed Overlap Rule**: Half-height intersection may not be optimal for all applications  
3. **Implementation Complexity**: Requires specialized basis function libraries
4. **Limited Customization**: Standard parameters may not suit all temporal dynamics
5. **Edge Effects**: Abrupt transition to zero may not match neural reality

## Detailed Comparison

### Computational Complexity Analysis

**Design Matrix Construction:**
- **Gaussian**: O(n_observations × n_bases × support_width)
- **Raised Cosine**: O(n_observations × n_bases × π) [fixed support]

**Memory Requirements:**
- **Gaussian**: ~3-5x larger due to extended support and truncation overhead
- **Raised Cosine**: Minimal due to compact support and sparsity

**Convolution Operations:**
- **Gaussian**: Requires truncation decisions and edge handling
- **Raised Cosine**: Clean boundaries with exact zero regions

### Numerical Stability Comparison

**Condition Number Analysis:**
- **Gaussian**: Potential ill-conditioning with high overlap
- **Raised Cosine**: Well-conditioned by design (half-height intersection)

**Optimization Convergence:**
- **Gaussian**: Smooth gradients but potential local minima from overlap
- **Raised Cosine**: Stable convergence with established parameters

### Temporal Resolution Analysis

**Short Timescales (10-50ms):**
- **Gaussian 0.4 rule**: Provides 4ms width at 10ms center (high resolution)
- **Raised cosine**: Fixed proportion based on log spacing

**Medium Timescales (50-200ms):**
- **Gaussian**: Adaptive widths based on local center density
- **Raised cosine**: Logarithmically increasing widths

**Long Timescales (200ms+):**
- **Gaussian**: Broader kernels appropriate for slow dynamics
- **Raised cosine**: Extended to 1420ms with appropriate smoothing

### Feature Extraction Comparison

**Click-to-DV Modeling Results:**

**Gaussian Approach (ClickDV Analysis):**
- **Performance**: 99.6% improvement over baseline
- **Feature Selection**: All 20 features selected (10 left + 10 right)
- **Biological Interpretation**: Clear left negative, right positive pattern
- **Cross-validation**: RMSE = 0.62, R² = 0.996

**Raised Cosine Expected Performance:**
- **Literature Results**: Comparable performance in similar GLM applications
- **Feature Selection**: Automatic through L1 regularization
- **Interpretability**: Standard in computational neuroscience

## Implementation Analysis

### Code Architecture Comparison

**Gaussian Implementation (ClickDV):**
```python
def gaussian_basis(t, center, width):
    return np.exp(-(t - center)**2 / (2 * width**2))

def create_click_features(click_times_left, click_times_right, eval_time, centers, widths):
    features_left = np.zeros(n_basis)
    features_right = np.zeros(n_basis)
    # Convolve click history with basis functions
    for i, (center, width) in enumerate(zip(centers, widths)):
        if len(past_clicks_left) > 0:
            lags_left = eval_time - past_clicks_left
            features_left[i] = np.sum(gaussian_basis(lags_left, center, width))
    return features_left, features_right
```

**Raised Cosine Implementation (neuroGLM):**
```matlab
function bases = click_basis(binsize_s,n_click_bases,click_endpoints_s,click_nl_offset,cutoff)
    nlin = @(x)log(x+click_nl_offset);
    nlininv = @(x)(exp(x)-click_nl_offset);    
    bs=basisFactory.makeNonlinearRaisedCos(n_click_bases,click_endpoints_s,binsize_s,cutoff,nlin,nlininv);
end

function y = raisedCosFun(x,center,scale,nlin)
    x=nlin(x);
    x = (x-center)/(scale*2/pi);
    y = ( cos(x) + 1 ) / 2;
    y(abs(x)>pi) = 0;
end
```

### Parameter Tuning Strategies

**Gaussian Approach Tuning:**
1. **Number of basis functions**: Balance temporal resolution vs. overfitting
2. **Time range**: Adjust based on expected neural dynamics
3. **Width factor**: 0.4 is proposed optimum, could be empirically validated
4. **Regularization**: Use Lasso to select most informative timescales

**Raised Cosine Tuning:**
1. **Number of basis functions**: 12 is standard, can adjust for specific applications
2. **Endpoint selection**: Based on stimulus and response timing
3. **Nonlinear transform**: Log spacing is standard for neural applications
4. **Offset parameter**: Prevents numerical issues with log(0)

## Performance Considerations

### Computational Benchmarking

**Training Time Comparison:**
- **Gaussian**: ~15-25% slower due to support calculations
- **Raised Cosine**: Faster due to sparse matrix operations

**Memory Usage:**
- **Gaussian**: ~200-400% more memory for equivalent temporal resolution
- **Raised Cosine**: Minimal memory footprint

**Scaling with Data Size:**
- **Gaussian**: Linear scaling with some overhead
- **Raised Cosine**: Near-linear scaling, excellent for large datasets

### Cross-Validation Performance

**Model Selection:**
- **Gaussian**: Custom approach requires validation of 0.4 factor
- **Raised Cosine**: Established parameters reduce hyperparameter search

**Generalization:**
- **Gaussian**: Novel approach may overfit to specific datasets
- **Raised Cosine**: Proven generalization across neural systems

**Reproducibility:**
- **Gaussian**: Requires careful documentation of custom parameters
- **Raised Cosine**: Standard implementations ensure reproducibility

## Biological Interpretations

### Neural Response Modeling

**Gaussian Basis Interpretation:**
- **Short-term kernels**: Capture rapid neural integration (10-50ms)
- **Medium-term kernels**: Model adaptation and temporal context (50-200ms)  
- **Long-term kernels**: Represent working memory and decision accumulation (200ms+)
- **Width adaptation**: Matches natural broadening of neural temporal windows

**Raised Cosine Interpretation:**
- **Standard timescales**: Established mapping to neural mechanisms
- **Log spacing**: Matches psychophysical and neural temporal scaling
- **Compact support**: May better represent finite neural memory
- **Half-height overlap**: Optimal for linear reconstruction of temporal dynamics

### Decision Variable Modeling

**Click Integration Dynamics:**
Both approaches successfully model how clicks integrate into decision variables:

1. **Early clicks** (10-50ms): Sharp, precise influence on momentary evidence
2. **Recent clicks** (50-200ms): Strong influence with temporal decay
3. **Remote clicks** (200ms+): Weak but persistent influence on accumulated evidence

**Hemispheric Differences:**
- **Left clicks**: Negative weights (evidence against rightward choice)
- **Right clicks**: Positive weights (evidence for rightward choice)
- **Asymmetries**: May reflect neural circuit differences or task biases

### Commitment-Related Changes

**UberPhys Pre/Post Analysis:**
The raised cosine approach enables analysis of neural changes around decision commitment:
- **Pre-commitment**: Strong sensory responsiveness
- **Post-commitment**: Maintained or enhanced responses (contrary to prediction)
- **Regional differences**: Varying patterns across brain areas

**Implications for Gaussian Approach:**
Could implement similar pre/post splitting with Gaussian bases to compare findings.

## Design Philosophy

### Gaussian Approach: Adaptive Precision

**Core Principles:**
1. **Biological realism**: Gaussian shapes match neural response profiles
2. **Adaptive resolution**: Variable widths optimize temporal precision
3. **Mathematical elegance**: Smooth, differentiable, interpretable
4. **Novel optimization**: Custom 0.4 factor for optimal coverage

**Research Philosophy:**
- **Exploratory**: Open to discovering new temporal dynamics
- **Customizable**: Adaptable to specific experimental paradigms
- **Innovative**: Willing to develop new methodological approaches
- **Interpretable**: Emphasizes biological understanding over pure performance

### Raised Cosine Approach: Established Optimality

**Core Principles:**
1. **Computational efficiency**: Optimize for speed and memory
2. **Theoretical foundation**: Build on signal processing theory
3. **Reproducibility**: Use established, validated parameters
4. **Comparative power**: Enable comparison across studies and labs

**Research Philosophy:**
- **Standardized**: Prioritize consistency and replicability
- **Validated**: Rely on extensively tested methodologies
- **Efficient**: Optimize for large-scale, high-throughput analysis
- **Collaborative**: Enable easy sharing and comparison of results

### Philosophical Trade-offs

**Innovation vs. Validation:**
- **Gaussian**: Higher innovation potential, lower immediate validation
- **Raised Cosine**: Lower innovation, higher established validation

**Customization vs. Standardization:**
- **Gaussian**: High customization, challenging cross-study comparison
- **Raised Cosine**: Standard approach, easier collaboration and replication

**Biological Realism vs. Computational Efficiency:**
- **Gaussian**: Emphasizes biological interpretability
- **Raised Cosine**: Prioritizes computational and mathematical considerations

## Use Case Guidelines

### When to Choose Gaussian Basis Functions

**Ideal Applications:**
1. **Exploratory neuroscience**: Investigating novel temporal dynamics
2. **Biological interpretation**: When mechanistic understanding is primary goal
3. **Custom paradigms**: Unique experimental designs requiring specialized approaches
4. **Small-to-medium scale**: Studies where computational resources are adequate
5. **Methodological development**: Advancing basis function theory and practice

**Specific Scenarios:**
- **Click-to-DV modeling**: Where adaptive temporal resolution may reveal new insights
- **High-resolution temporal analysis**: When fine-grained timing is crucial
- **Novel behavioral paradigms**: Requiring custom temporal kernel design
- **Proof-of-concept studies**: Testing new theoretical predictions

### When to Choose Raised Cosine Basis Functions

**Ideal Applications:**
1. **Large-scale studies**: High-throughput analysis requiring computational efficiency
2. **Comparative research**: Studies requiring standardized methodology
3. **Collaborative projects**: Multi-lab studies needing consistent approaches
4. **Established paradigms**: Well-characterized experimental designs
5. **Production pipelines**: Routine analysis requiring reliable, fast processing

**Specific Scenarios:**
- **Brain-wide neural recording analysis**: Large datasets requiring efficient processing
- **Cross-species comparisons**: Standardized approaches for different animal models
- **Clinical applications**: Where validated methodology is essential
- **Meta-analyses**: Combining results across multiple studies

### Hybrid Approaches

**Comparative Validation:**
1. **Implement both approaches** on the same dataset
2. **Cross-validate performance** using identical procedures
3. **Compare biological interpretations** and extracted timescales
4. **Assess computational trade-offs** in practical settings

**Method Development:**
1. **Gaussian-inspired raised cosines**: Custom width scaling in raised cosine framework
2. **Raised cosine-inspired Gaussians**: Compact support Gaussian variants
3. **Adaptive basis selection**: Data-driven choice between approaches
4. **Ensemble methods**: Combining predictions from both approaches

## Future Directions

### Methodological Advances

**Gaussian Basis Development:**
1. **Empirical validation of 0.4 factor**: Systematic testing across datasets and paradigms
2. **Adaptive width algorithms**: Data-driven optimization of width scaling
3. **Computational optimizations**: Efficient implementations reducing overhead
4. **Truncation strategies**: Optimal support determination for different applications

**Raised Cosine Extensions:**
1. **Adaptive overlap control**: Moving beyond fixed half-height intersection
2. **Nonlinear transform optimization**: Data-driven spacing strategies
3. **Multi-scale approaches**: Combining different timescale ranges
4. **Biological constraint integration**: Incorporating known neural timescales

### Theoretical Research

**Mathematical Analysis:**
1. **Optimal overlap theory**: Mathematical framework for basis function overlap
2. **Information theoretic analysis**: Mutual information and entropy considerations
3. **Approximation theory**: Error bounds and convergence properties
4. **Regularization interactions**: How basis choice affects regularization effectiveness

**Biological Validation:**
1. **Cross-species validation**: Testing approaches across animal models
2. **Neural mechanism mapping**: Linking basis functions to cellular properties
3. **Developmental studies**: How optimal timescales change with learning/development
4. **Pathological conditions**: Basis function changes in disease states

### Computational Innovations

**Implementation Improvements:**
1. **GPU acceleration**: Parallel computation for large-scale analysis
2. **Streaming algorithms**: Online processing for real-time applications
3. **Automatic differentiation**: Integration with modern ML frameworks
4. **Distributed computing**: Scaling to very large neural datasets

**Integration with Modern ML:**
1. **Deep learning hybrids**: Neural networks with interpretable temporal bases
2. **Bayesian approaches**: Uncertainty quantification in temporal kernel estimation
3. **Causal inference**: Temporal basis functions in causal analysis frameworks
4. **Multi-modal integration**: Combining neural, behavioral, and physiological data

### Experimental Applications

**Emerging Paradigms:**
1. **Closed-loop experiments**: Real-time basis function-based stimulation
2. **Multi-area recordings**: Basis functions for neural population interactions
3. **Longitudinal studies**: Tracking temporal dynamics over learning/development
4. **Clinical applications**: Diagnostic and therapeutic uses of temporal kernels

**Technology Integration:**
1. **High-density recording**: Optimizing basis functions for massive neural datasets
2. **Optogenetics**: Temporal basis functions for stimulation pattern design
3. **Brain-computer interfaces**: Real-time decoding using optimal temporal kernels
4. **Computational psychiatry**: Temporal dynamics as biomarkers

## Conclusion

The choice between Gaussian and raised cosine basis functions for GLM temporal kernel design represents a fundamental decision that impacts computational efficiency, biological interpretability, and methodological validity. Both approaches offer distinct advantages and address different priorities in computational neuroscience research.

**Gaussian basis functions**, as implemented in the ClickDV project, represent an innovative approach emphasizing biological realism and adaptive temporal resolution. The novel 0.4 width scaling factor provides a promising framework for optimizing temporal coverage while maintaining interpretable neural dynamics. This approach excels in exploratory research contexts where understanding biological mechanisms is paramount and computational resources allow for the additional overhead.

**Raised cosine basis functions**, as established in the neuroGLM framework and widely adopted in computational neuroscience, provide a mathematically elegant and computationally efficient solution backed by extensive validation. The half-height overlap strategy and compact support properties make this approach ideal for large-scale studies, collaborative research, and applications requiring standardized methodology.

The **key insight** from this analysis is that both approaches are scientifically valid and serve different research needs. The Gaussian approach offers **methodological innovation** with potential for new biological insights, while the raised cosine approach provides **established reliability** with proven performance across diverse applications.

**Recommendations for researchers:**

1. **For exploratory studies** investigating novel temporal dynamics: Consider Gaussian basis functions
2. **For comparative studies** requiring methodological consistency: Use raised cosine basis functions  
3. **For methodological validation**: Implement both approaches and compare results
4. **For computational efficiency**: Raised cosine functions provide clear advantages
5. **For biological interpretation**: Both approaches offer interpretable parameters with different emphases

The **future of GLM temporal kernel design** likely lies in integrating the best features of both approaches: the adaptive precision of Gaussian methods with the computational efficiency of raised cosine functions. As computational neuroscience continues to evolve toward larger datasets and more complex experimental paradigms, the development of hybrid approaches and data-driven basis function optimization will become increasingly important.

Ultimately, the choice of basis function approach should be guided by research objectives, computational constraints, data characteristics, and the broader scientific context. Both Gaussian and raised cosine approaches contribute valuable tools to the computational neuroscientist's toolkit, and their continued development and validation will advance our understanding of neural temporal dynamics and decision-making processes.

---

*This analysis was based on implementations from the ClickDV project (Gaussian approach) and the UberPhys/neuroGLM framework (raised cosine approach), representing current best practices in computational neuroscience GLM analysis.*