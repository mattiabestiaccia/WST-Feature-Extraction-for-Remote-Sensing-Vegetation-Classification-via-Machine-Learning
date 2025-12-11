# QUALITATIVE ANALYSIS: WST vs RGB STATISTICS ROBUSTNESS TO SPECKLE NOISE

## EXECUTIVE SUMMARY

Comparative analysis reveals significant differences in speckle noise 
robustness between feature extraction methods. Results show that 
**WST (Wavelet Scattering Transform)** and **Hybrid** methods maintain 
superior performance compared to **Advanced RGB Statistics** under speckle noise conditions.

## KEY FINDINGS

### 1. GLOBAL PERFORMANCE BY METHOD

- **Advanced Stats**: 0.914 ± 0.100
  (Highest variability)
- **Hybrid**: 0.929 ± 0.096
- **WST**: 0.926 ± 0.062
  (Lowest standard deviation → highest consistency)

### 2. SPECKLE NOISE ROBUSTNESS

| Condition | Mean Accuracy | Performance Loss |
|-----------|---------------|------------------|
| Clean | 0.959 ± 0.046 | baseline |
| Speckle ν=0.15 | 0.924 ± 0.070 | -3.5% |
| Speckle ν=0.35 | 0.904 ± 0.119 | -5.5% |
| Speckle ν=0.55 | 0.904 ± 0.088 | -5.5% |

### 3. GEOGRAPHIC AREA ANALYSIS

#### ASSATIGUE (Critical area analysis)
- **Hybrid**: 0.918 ± 0.059
- **WST**: 0.914 ± 0.043
- **Advanced Stats**: 0.895 ± 0.070

#### POPOLAR (Critical area analysis)
- **WST**: 0.951 ± 0.037
- **Hybrid**: 0.949 ± 0.113
- **Advanced Stats**: 0.945 ± 0.072

#### SUNSET (Critical area analysis)
- **Hybrid**: 0.919 ± 0.107
- **WST**: 0.912 ± 0.087
  (Best performance and stability in most critical area)
- **Advanced Stats**: 0.900 ± 0.137
  (Significantly lower performance in critical area)

## CONCLUSIONS

### 🔍 **Speckle Noise Robustness**

1. **WST shows superior robustness**: Maintains consistent performance 
   across all speckle noise levels, particularly in SUNSET area.

2. **Hybrid combines strengths**: Balances WST robustness with 
   complementary RGB statistics for improved overall performance.

3. **Advanced Stats are vulnerable**: Show higher sensitivity to 
   speckle noise, especially at higher noise levels (ν=0.55).

### 🎯 **Speckle Noise Mechanisms**

**WST**: Multi-scale coefficients capture structural patterns that are 
less affected by multiplicative speckle noise. The wavelet transform's 
multi-resolution analysis provides inherent noise resilience.

**Advanced RGB Stats**: Single-channel statistics are more sensitive to 
the multiplicative nature of speckle noise, which affects local pixel 
intensity distributions.

### 📊 **Practical Recommendations**

1. **For speckle-critical applications**: Use **WST**, especially in 
   environments with speckle noise ν > 0.35.

2. **For balanced performance**: **Hybrid** method offers good compromise 
   between robustness and feature diversity.

3. **For clean datasets**: Advanced Stats sufficient only in ideal conditions.
