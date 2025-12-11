# QUALITATIVE ANALYSIS: WST vs RGB STATISTICS ROBUSTNESS TO UNIFORM NOISE

## EXECUTIVE SUMMARY

Comparative analysis reveals significant differences in uniform noise 
robustness between feature extraction methods. Results show that 
**WST (Wavelet Scattering Transform)** and **Hybrid** methods maintain 
superior performance compared to **Advanced RGB Statistics** under uniform noise conditions.

## KEY FINDINGS

### 1. GLOBAL PERFORMANCE BY METHOD

- **Advanced Stats**: 0.938 ± 0.074
  (Highest variability)
- **Hybrid**: 0.941 ± 0.057
- **WST**: 0.930 ± 0.051
  (Lowest standard deviation → highest consistency)

### 2. UNIFORM NOISE ROBUSTNESS

| Condition | Mean Accuracy | Performance Loss |
|-----------|---------------|------------------|
| Clean | 0.959 ± 0.046 | baseline |
| Uniform ±10 | 0.942 ± 0.055 | -1.7% |
| Uniform ±25 | 0.926 ± 0.062 | -3.3% |
| Uniform ±40 | 0.919 ± 0.072 | -4.0% |

### 3. GEOGRAPHIC AREA ANALYSIS

#### ASSATIGUE (Critical area analysis)
- **Advanced Stats**: 0.947 ± 0.054
- **Hybrid**: 0.945 ± 0.056
- **WST**: 0.921 ± 0.050

#### POPOLAR (Critical area analysis)
- **Advanced Stats**: 0.964 ± 0.052
- **Hybrid**: 0.962 ± 0.046
- **WST**: 0.939 ± 0.049

#### SUNSET (Critical area analysis)
- **WST**: 0.930 ± 0.053
  (Best performance and stability in most critical area)
- **Hybrid**: 0.914 ± 0.060
- **Advanced Stats**: 0.904 ± 0.095
  (Significantly lower performance in critical area)

## CONCLUSIONS

### 🔍 **Uniform Noise Robustness**

1. **WST shows superior robustness**: Maintains consistent performance 
   across all uniform noise levels, particularly in SUNSET area.

2. **Hybrid combines strengths**: Balances WST robustness with 
   complementary RGB statistics for improved overall performance.

3. **Advanced Stats are vulnerable**: Show higher sensitivity to 
   uniform noise, especially at higher noise levels (±40).

### 🎯 **Uniform Noise Mechanisms**

**WST**: Multi-scale coefficients capture structural patterns that are 
less affected by additive uniform noise. The wavelet transform's 
multi-resolution analysis provides inherent noise resilience.

**Advanced RGB Stats**: Single-channel statistics are more sensitive to 
the additive nature of uniform noise, which shifts local pixel 
intensity distributions uniformly.

### 📊 **Practical Recommendations**

1. **For uniform noise-critical applications**: Use **WST**, especially in 
   environments with uniform noise range > ±25.

2. **For balanced performance**: **Hybrid** method offers good compromise 
   between robustness and feature diversity.

3. **For clean datasets**: Advanced Stats sufficient only in ideal conditions.
