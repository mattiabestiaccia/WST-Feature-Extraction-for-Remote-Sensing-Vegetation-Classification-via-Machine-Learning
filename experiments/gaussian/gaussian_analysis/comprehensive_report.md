# COMPARATIVE REPORT: RANDOM FOREST EXPERIMENTS
============================================================

## EXECUTIVE SUMMARY

This report presents a comprehensive analysis of Random Forest classification
experiments comparing feature extraction methods (WST, Advanced Stats, Hybrid)
under different noise conditions and dataset sizes.

## EXPERIMENTAL SETUP
- Total experiments: 324
- Noise conditions: clean, gaussian30, gaussian50
- Geographic areas: assatigue, popolar, sunset
- Dataset types: mini, original, small
- Feature extraction methods: advanced_stats, hybrid, wst
- Feature selection (k values): 2, 5, 10, 20

## AVERAGE ACCURACY BY FEATURE EXTRACTION METHOD
- **Advanced Stats**: 0.870 ± 0.121 (108.0 experiments)
- **Hybrid**: 0.913 ± 0.097 (108.0 experiments)
- **WST**: 0.913 ± 0.067 (108.0 experiments)

## AVERAGE ACCURACY BY NOISE CONDITION
- **Clean**: 0.959 ± 0.046 (108.0 experiments)
- **Gaussian σ=30**: 0.893 ± 0.074 (108.0 experiments)
- **Gaussian σ=50**: 0.845 ± 0.125 (108.0 experiments)

## TOP 10 GLOBAL PERFORMANCES
- 1.000 | Advanced Stats | assatigue | Small | k=2 | Clean
- 1.000 | Hybrid | assatigue | Small | k=2 | Clean
- 1.000 | Hybrid | assatigue | Small | k=20 | Clean
- 1.000 | Advanced Stats | assatigue | Original | k=5 | Clean
- 1.000 | Advanced Stats | assatigue | Original | k=10 | Clean
- 1.000 | Hybrid | assatigue | Original | k=10 | Clean
- 1.000 | Advanced Stats | popolar | Mini | k=2 | Clean
- 1.000 | WST | popolar | Mini | k=2 | Clean
- 1.000 | Advanced Stats | popolar | Mini | k=5 | Clean
- 1.000 | Advanced Stats | popolar | Mini | k=10 | Clean

## PERFORMANCE BY GEOGRAPHIC AREA
### ASSATIGUE
  - Advanced Stats: 0.863 ± 0.091
  - Hybrid: 0.889 ± 0.091
  - WST: 0.871 ± 0.068

### POPOLAR
  - Advanced Stats: 0.912 ± 0.085
  - Hybrid: 0.920 ± 0.099
  - WST: 0.926 ± 0.061

### SUNSET
  - Advanced Stats: 0.835 ± 0.161
  - Hybrid: 0.930 ± 0.099
  - WST: 0.943 ± 0.049

## NOISE ROBUSTNESS ANALYSIS
### Average degradation per method (Clean → Gaussian σ=30)
- Advanced Stats: 0.100 ± 0.082
- Hybrid: 0.062 ± 0.069
- WST: 0.036 ± 0.052

### Average degradation per method (Clean → Gaussian σ=50)
- Advanced Stats: 0.199 ± 0.135
- Hybrid: 0.099 ± 0.130
- WST: 0.043 ± 0.068
