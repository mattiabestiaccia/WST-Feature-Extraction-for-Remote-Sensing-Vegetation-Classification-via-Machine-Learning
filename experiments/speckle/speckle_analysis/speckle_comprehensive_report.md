# COMPARATIVE REPORT: RANDOM FOREST EXPERIMENTS - SPECKLE NOISE
======================================================================

## EXECUTIVE SUMMARY

This report presents a comprehensive analysis of Random Forest classification
experiments comparing feature extraction methods (WST, Advanced Stats, Hybrid)
under different speckle noise conditions and dataset sizes.

## EXPERIMENTAL SETUP
- Total experiments: 432
- Noise conditions: clean, speckle15, speckle35, speckle55
- Geographic areas: assatigue, popolar, sunset
- Dataset types: mini, original, small
- Feature extraction methods: advanced_stats, hybrid, wst
- Feature selection (k values): 2, 5, 10, 20

## AVERAGE ACCURACY BY FEATURE EXTRACTION METHOD
- **Advanced Stats**: 0.914 ± 0.100 (144.0 experiments)
- **Hybrid**: 0.929 ± 0.096 (144.0 experiments)
- **WST**: 0.926 ± 0.062 (144.0 experiments)

## AVERAGE ACCURACY BY NOISE CONDITION
- **Clean**: 0.959 ± 0.046 (108.0 experiments)
- **Speckle ν=0.15**: 0.924 ± 0.070 (108.0 experiments)
- **Speckle ν=0.35**: 0.904 ± 0.119 (108.0 experiments)
- **Speckle ν=0.55**: 0.904 ± 0.088 (108.0 experiments)

## TOP 10 GLOBAL PERFORMANCES
- 1.000 | Hybrid | popolar | Mini | k=5 | Speckle ν=0.15
- 1.000 | Advanced Stats | popolar | Mini | k=10 | Speckle ν=0.15
- 1.000 | Hybrid | popolar | Mini | k=10 | Speckle ν=0.15
- 1.000 | Hybrid | popolar | Mini | k=20 | Speckle ν=0.15
- 1.000 | Advanced Stats | popolar | Small | k=10 | Speckle ν=0.15
- 1.000 | Hybrid | popolar | Small | k=10 | Speckle ν=0.15
- 1.000 | Advanced Stats | popolar | Original | k=2 | Speckle ν=0.15
- 1.000 | Hybrid | popolar | Original | k=2 | Speckle ν=0.15
- 1.000 | Advanced Stats | popolar | Original | k=5 | Speckle ν=0.15
- 1.000 | Hybrid | popolar | Original | k=5 | Speckle ν=0.15

## PERFORMANCE BY GEOGRAPHIC AREA
### ASSATIGUE
  - Advanced Stats: 0.895 ± 0.070
  - Hybrid: 0.918 ± 0.059
  - WST: 0.914 ± 0.043

### POPOLAR
  - Advanced Stats: 0.945 ± 0.072
  - Hybrid: 0.949 ± 0.113
  - WST: 0.951 ± 0.037

### SUNSET
  - Advanced Stats: 0.900 ± 0.137
  - Hybrid: 0.919 ± 0.107
  - WST: 0.912 ± 0.087

## SPECKLE NOISE ROBUSTNESS ANALYSIS
### Average degradation per method (Clean → Speckle ν=0.15)
- Advanced Stats: 0.048 ± 0.078
- Hybrid: 0.037 ± 0.076
- WST: 0.018 ± 0.063

### Average degradation per method (Clean → Speckle ν=0.35)
- Advanced Stats: 0.071 ± 0.125
- Hybrid: 0.073 ± 0.153
- WST: 0.021 ± 0.105

### Average degradation per method (Clean → Speckle ν=0.55)
- Advanced Stats: 0.105 ± 0.124
- Hybrid: 0.042 ± 0.072
- WST: 0.016 ± 0.083
