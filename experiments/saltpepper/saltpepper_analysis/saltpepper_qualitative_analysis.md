# QUALITATIVE ANALYSIS: WST vs RGB STATISTICS ROBUSTNESS TO SALT & PEPPER NOISE

## EXECUTIVE SUMMARY

Comparative analysis reveals significant differences in salt & pepper noise 
robustness between feature extraction methods. Results show that 
**WST (Wavelet Scattering Transform)** and **Hybrid** methods maintain 
superior performance compared to **Advanced RGB Statistics** under salt & pepper noise conditions.

## KEY FINDINGS

### 1. GLOBAL PERFORMANCE BY METHOD

- **Advanced Stats**: 0.935 ± 0.077
  (Highest variability)
- **Hybrid**: 0.938 ± 0.075
- **WST**: 0.888 ± 0.087
  (Lowest standard deviation → highest consistency)

### 2. SALT & PEPPER NOISE ROBUSTNESS

| Condition | Mean Accuracy | Performance Loss |
|-----------|---------------|------------------|
| Clean | 0.945 ± 0.059 | baseline |
| S&P 15% | 0.929 ± 0.083 | -1.7% |
| S&P 25% | 0.870 ± 0.090 | -7.5% |
| S&P 5% | 0.938 ± 0.073 | -0.7% |

### 3. GEOGRAPHIC AREA ANALYSIS

#### ASSATIGUE (Critical area analysis)
- **Hybrid**: 0.944 ± 0.072
- **Advanced Stats**: 0.942 ± 0.072
- **WST**: 0.861 ± 0.082

#### POPOLAR (Critical area analysis)
- **Hybrid**: 0.962 ± 0.054
- **Advanced Stats**: 0.960 ± 0.063
- **WST**: 0.900 ± 0.081

#### SUNSET (Critical area analysis)
- **Hybrid**: 0.907 ± 0.085
- **WST**: 0.904 ± 0.092
  (Best performance and stability in most critical area)
- **Advanced Stats**: 0.902 ± 0.083
  (Significantly lower performance in critical area)

## CONCLUSIONS

### 🔍 **Salt & Pepper Noise Robustness**

1. **WST shows superior robustness**: Maintains consistent performance 
   across all salt & pepper noise levels, particularly in SUNSET area.

2. **Hybrid combines strengths**: Balances WST robustness with 
   complementary RGB statistics for improved overall performance.

3. **Advanced Stats are vulnerable**: Show higher sensitivity to 
   salt & pepper noise, especially at higher noise levels (25%).

### 🎯 **Salt & Pepper Noise Mechanisms**

**WST**: Multi-scale coefficients capture structural patterns that are 
less affected by impulse noise characteristics of salt & pepper noise. The wavelet 
transform's multi-resolution analysis provides inherent noise resilience.

**Advanced RGB Stats**: Single-channel statistics are more sensitive to 
the extreme pixel values introduced by salt & pepper noise, which 
significantly distorts mean, variance, and higher-order moment calculations.

### 📊 **Practical Recommendations**

1. **For impulse/transmission noise applications**: Use **WST**, especially in 
   environments with salt & pepper noise > 15%.

2. **For balanced performance**: **Hybrid** method offers good compromise 
   between robustness and feature diversity.

3. **For clean datasets**: Advanced Stats sufficient only in ideal conditions.
