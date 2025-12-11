# QUALITATIVE ANALYSIS: WST vs RGB STATISTICS ROBUSTNESS

## EXECUTIVE SUMMARY

Comparative analysis of 324 experiments reveals significant differences in 
noise robustness between feature extraction methods. Results show that 
**WST (Wavelet Scattering Transform)** and **Hybrid** methods maintain 
superior performance compared to **Advanced RGB Statistics** under noisy conditions.

## KEY FINDINGS

### 1. GLOBAL PERFORMANCE BY METHOD

- **Advanced Stats**: 0.870 ± 0.121
  (Highest variability)
- **Hybrid**: 0.913 ± 0.097
- **WST**: 0.913 ± 0.067
  (Lowest standard deviation → highest consistency)

### 2. NOISE ROBUSTNESS

| Condition | Mean Accuracy | Performance Loss |
|-----------|---------------|------------------|
| Clean | 0.959 ± 0.046 | baseline |
| Gaussian σ=30 | 0.893 ± 0.074 | -6.6% |
| Gaussian σ=50 | 0.845 ± 0.125 | -11.4% |

### 3. GEOGRAPHIC AREA ANALYSIS

#### ASSATIGUE (Critical area analysis)
- **Hybrid**: 0.889 ± 0.091
- **WST**: 0.871 ± 0.068
- **Advanced Stats**: 0.863 ± 0.091

#### POPOLAR (Critical area analysis)
- **WST**: 0.926 ± 0.061
- **Hybrid**: 0.920 ± 0.099
- **Advanced Stats**: 0.912 ± 0.085

#### SUNSET (Critical area analysis)
- **WST**: 0.943 ± 0.049
  (Best performance and stability in most critical area)
- **Hybrid**: 0.930 ± 0.099
- **Advanced Stats**: 0.835 ± 0.161
  (Significantly lower performance in critical area)

## FEATURE SELECTION ANALYSIS

### Most frequently selected features:

#### Advanced Stats
1. **B_cv**: 57 times
   (Coefficient of variation - sensitive to noise)
1. **B_p25**: 51 times
1. **G_iqr**: 45 times
   (Interquartile range - moderately robust statistic)
1. **B_mean**: 42 times
1. **R_iqr**: 41 times
   (Interquartile range - moderately robust statistic)

#### Hybrid
1. **R_WST0_std**: 68 times
   (Standard deviation of first WST coefficient - robust multi-scale feature)
1. **G_WST0_std**: 53 times
   (Standard deviation of first WST coefficient - robust multi-scale feature)
1. **B_p25**: 38 times
1. **B_cv**: 37 times
   (Coefficient of variation - sensitive to noise)
1. **B_p10**: 35 times

#### WST
1. **R_WST0_std**: 87 times
   (Standard deviation of first WST coefficient - robust multi-scale feature)
1. **G_WST0_std**: 71 times
   (Standard deviation of first WST coefficient - robust multi-scale feature)
1. **B_mean**: 55 times
1. **B_WST0_mean**: 51 times
1. **G_std**: 37 times

## CONCLUSIONS

### 🔍 **Noise Robustness**

1. **WST is most robust**: Shows lowest standard deviation (0.067) and 
   excellent performance in SUNSET area (most critical for noise).

2. **Hybrid combines best aspects**: Maintains WST robustness while 
   integrating selective RGB statistics.

3. **Advanced Stats are vulnerable**: Show highest variability (±0.121) 
   and significantly lower performance in SUNSET area.

### 🎯 **Robustness Mechanisms**

**WST**: Multi-scale coefficients capture structural information invariant 
to local noise perturbations. Features like `R_WST0_std` and `G_WST0_std` 
represent robust structural variations.

**Advanced RGB Stats**: Single-channel statistics (percentiles, IQR, CV) 
are more sensitive to local fluctuations introduced by Gaussian noise.

### 📊 **Practical Recommendations**

1. **For noise-critical applications**: Use **WST**, especially in 
   environments with Gaussian noise σ > 30.

2. **For balanced performance and interpretability**: **Hybrid** method 
   offers good compromise.

3. **For clean datasets with computational constraints**: Advanced Stats 
   sufficient only in ideal conditions.

### 🔬 **Experimental Evidence**

- **Spatial consistency**: WST maintains uniform performance across geographic areas
- **Temporal stability**: Lower standard deviation indicates less sensitivity to dataset variations
- **Feature selection efficacy**: WST coefficients dominate automatic feature selection
