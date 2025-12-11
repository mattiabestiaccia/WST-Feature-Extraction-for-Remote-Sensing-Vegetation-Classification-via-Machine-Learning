# COMPARATIVE REPORT: RANDOM FOREST EXPERIMENTS - POISSON NOISE
======================================================================

## EXECUTIVE SUMMARY

This report presents a comprehensive analysis of Random Forest classification
experiments comparing feature extraction methods (WST, Advanced Stats, Hybrid)
under different poisson noise conditions and dataset sizes.

## EXPERIMENTAL SETUP
- Total experiments: 324
- Noise conditions: clean, poisson40, poisson60
- Geographic areas: assatigue, popolar, sunset
- Dataset types: mini, original, small
- Feature extraction methods: advanced_stats, hybrid, wst
- Feature selection (k values): 2, 5, 10, 20

## AVERAGE ACCURACY BY FEATURE EXTRACTION METHOD
- **Advanced Stats**: 0.915 ± 0.103 (108.0 experiments)
- **Hybrid**: 0.930 ± 0.087 (108.0 experiments)
- **WST**: 0.928 ± 0.053 (108.0 experiments)

## AVERAGE ACCURACY BY NOISE CONDITION
- **Clean**: 0.959 ± 0.046 (108.0 experiments)
- **Poisson λ=40**: 0.899 ± 0.111 (108.0 experiments)
- **Poisson λ=60**: 0.916 ± 0.069 (108.0 experiments)

## TOP 10 GLOBAL PERFORMANCES
- 1.000 | WST | popolar | Mini | k=2 | Poisson λ=40
- 1.000 | WST | popolar | Mini | k=5 | Poisson λ=40
- 1.000 | Hybrid | popolar | Mini | k=10 | Poisson λ=40
- 1.000 | Advanced Stats | popolar | Small | k=2 | Poisson λ=40
- 1.000 | Hybrid | popolar | Small | k=2 | Poisson λ=40
- 1.000 | Advanced Stats | popolar | Original | k=2 | Poisson λ=40
- 1.000 | Hybrid | popolar | Original | k=2 | Poisson λ=40
- 1.000 | WST | sunset | Mini | k=20 | Poisson λ=40
- 1.000 | WST | popolar | Mini | k=10 | Poisson λ=60
- 1.000 | WST | popolar | Mini | k=20 | Poisson λ=60

## PERFORMANCE BY GEOGRAPHIC AREA
### ASSATIGUE
  - Advanced Stats: 0.894 ± 0.125
  - Hybrid: 0.904 ± 0.089
  - WST: 0.893 ± 0.052

### POPOLAR
  - Advanced Stats: 0.955 ± 0.061
  - Hybrid: 0.950 ± 0.101
  - WST: 0.949 ± 0.041

### SUNSET
  - Advanced Stats: 0.898 ± 0.105
  - Hybrid: 0.937 ± 0.059
  - WST: 0.941 ± 0.047

## POISSON NOISE ROBUSTNESS ANALYSIS
### Average degradation per method (Clean → Poisson λ=40)
- Advanced Stats: 0.101 ± 0.146
- Hybrid: 0.059 ± 0.105
- WST: 0.020 ± 0.066

### Average degradation per method (Clean → Poisson λ=60)
- Advanced Stats: 0.063 ± 0.068
- Hybrid: 0.051 ± 0.082
- WST: 0.015 ± 0.060
