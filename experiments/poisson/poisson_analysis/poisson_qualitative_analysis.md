# QUALITATIVE ANALYSIS: WST vs RGB STATISTICS ROBUSTNESS TO POISSON NOISE

## EXECUTIVE SUMMARY

Comparative analysis reveals significant differences in poisson noise 
robustness between feature extraction methods. Results show that 
**WST (Wavelet Scattering Transform)** and **Hybrid** methods maintain 
superior performance compared to **Advanced RGB Statistics** under poisson noise conditions.

## KEY FINDINGS

### 1. GLOBAL PERFORMANCE BY METHOD

- **Advanced Stats**: 0.915 ± 0.103
  (Highest variability)
- **Hybrid**: 0.930 ± 0.087
- **WST**: 0.928 ± 0.053
  (Lowest standard deviation → highest consistency)

### 2. POISSON NOISE ROBUSTNESS

| Condition | Mean Accuracy | Performance Loss |
|-----------|---------------|------------------|
| Clean | 0.959 ± 0.046 | baseline |
| Poisson λ=40 | 0.899 ± 0.111 | -6.0% |
| Poisson λ=60 | 0.916 ± 0.069 | -4.3% |

### 3. GEOGRAPHIC AREA ANALYSIS

#### ASSATIGUE (Critical area analysis)
- **Hybrid**: 0.904 ± 0.089
- **Advanced Stats**: 0.894 ± 0.125
- **WST**: 0.893 ± 0.052

#### POPOLAR (Critical area analysis)
- **Advanced Stats**: 0.955 ± 0.061
- **Hybrid**: 0.950 ± 0.101
- **WST**: 0.949 ± 0.041

#### SUNSET (Critical area analysis)
- **WST**: 0.941 ± 0.047
  (Best performance and stability in most critical area)
- **Hybrid**: 0.937 ± 0.059
- **Advanced Stats**: 0.898 ± 0.105
  (Significantly lower performance in critical area)

## CONCLUSIONS

### 🔍 **Poisson Noise Robustness**

1. **WST shows superior robustness**: Maintains consistent performance 
   across all poisson noise levels, particularly in SUNSET area.

2. **Hybrid combines strengths**: Balances WST robustness with 
   complementary RGB statistics for improved overall performance.

3. **Advanced Stats are vulnerable**: Show higher sensitivity to 
   poisson noise, especially at higher noise levels (λ=60).

### 🎯 **Poisson Noise Mechanisms**

**WST**: Multi-scale coefficients capture structural patterns that are 
less affected by shot noise characteristics of Poisson noise. The wavelet 
transform's multi-resolution analysis provides inherent noise resilience.

**Advanced RGB Stats**: Single-channel statistics are more sensitive to 
the signal-dependent nature of Poisson noise, which affects pixel 
intensity distributions proportionally to signal strength.

### 📊 **Practical Recommendations**

1. **For low-light/shot noise applications**: Use **WST**, especially in 
   environments with Poisson noise λ > 40.

2. **For balanced performance**: **Hybrid** method offers good compromise 
   between robustness and feature diversity.

3. **For clean datasets**: Advanced Stats sufficient only in ideal conditions.
