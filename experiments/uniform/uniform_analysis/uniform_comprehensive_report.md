# COMPARATIVE REPORT: RANDOM FOREST EXPERIMENTS - UNIFORM NOISE
======================================================================

## EXECUTIVE SUMMARY

This report presents a comprehensive analysis of Random Forest classification
experiments comparing feature extraction methods (WST, Advanced Stats, Hybrid)
under different uniform noise conditions and dataset sizes.

## EXPERIMENTAL SETUP
- Total experiments: 432
- Noise conditions: clean, uniform10, uniform25, uniform40
- Geographic areas: assatigue, popolar, sunset
- Dataset types: mini, original, small
- Feature extraction methods: advanced_stats, hybrid, wst
- Feature selection (k values): 2, 5, 10, 20

## AVERAGE ACCURACY BY FEATURE EXTRACTION METHOD
- **Advanced Stats**: 0.938 ± 0.074 (144.0 experiments)
- **Hybrid**: 0.941 ± 0.057 (144.0 experiments)
- **WST**: 0.930 ± 0.051 (144.0 experiments)

## AVERAGE ACCURACY BY NOISE CONDITION
- **Clean**: 0.959 ± 0.046 (108.0 experiments)
- **Uniform ±10**: 0.942 ± 0.055 (108.0 experiments)
- **Uniform ±25**: 0.926 ± 0.062 (108.0 experiments)
- **Uniform ±40**: 0.919 ± 0.072 (108.0 experiments)

## TOP 10 GLOBAL PERFORMANCES
- 1.000 | Advanced Stats | assatigue | Mini | k=2 | Uniform ±10
- 1.000 | Hybrid | assatigue | Mini | k=2 | Uniform ±10
- 1.000 | WST | popolar | Mini | k=5 | Uniform ±10
- 1.000 | Advanced Stats | popolar | Mini | k=10 | Uniform ±10
- 1.000 | Hybrid | popolar | Mini | k=10 | Uniform ±10
- 1.000 | Advanced Stats | popolar | Mini | k=20 | Uniform ±10
- 1.000 | Hybrid | popolar | Mini | k=20 | Uniform ±10
- 1.000 | Advanced Stats | popolar | Small | k=2 | Uniform ±10
- 1.000 | Hybrid | popolar | Small | k=2 | Uniform ±10
- 1.000 | Advanced Stats | popolar | Small | k=5 | Uniform ±10

## PERFORMANCE BY GEOGRAPHIC AREA
### ASSATIGUE
  - Advanced Stats: 0.947 ± 0.054
  - Hybrid: 0.945 ± 0.056
  - WST: 0.921 ± 0.050

### POPOLAR
  - Advanced Stats: 0.964 ± 0.052
  - Hybrid: 0.962 ± 0.046
  - WST: 0.939 ± 0.049

### SUNSET
  - Advanced Stats: 0.904 ± 0.095
  - Hybrid: 0.914 ± 0.060
  - WST: 0.930 ± 0.053

## UNIFORM NOISE ROBUSTNESS ANALYSIS
### Average degradation per method (Clean → Uniform ±10)
- Advanced Stats: 0.030 ± 0.063
- Hybrid: 0.016 ± 0.033
- WST: 0.005 ± 0.039

### Average degradation per method (Clean → Uniform ±25)
- Advanced Stats: 0.042 ± 0.074
- Hybrid: 0.038 ± 0.052
- WST: 0.019 ± 0.059

### Average degradation per method (Clean → Uniform ±40)
- Advanced Stats: 0.055 ± 0.103
- Hybrid: 0.052 ± 0.064
- WST: 0.015 ± 0.061
