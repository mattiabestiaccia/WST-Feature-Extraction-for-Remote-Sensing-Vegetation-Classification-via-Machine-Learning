# COMPARATIVE REPORT: RANDOM FOREST EXPERIMENTS - SALT & PEPPER NOISE
======================================================================

## EXECUTIVE SUMMARY

This report presents a comprehensive analysis of Random Forest classification
experiments comparing feature extraction methods (WST, Advanced Stats, Hybrid)
under different salt and pepper noise conditions and dataset sizes.

## EXPERIMENTAL SETUP
- Total experiments: 432
- Noise conditions: clean, saltpepper15, saltpepper25, saltpepper5
- Geographic areas: assatigue, popolar, sunset
- Dataset types: mini, original, small
- Feature extraction methods: advanced_stats, hybrid, wst
- Feature selection (k values): 2, 5, 10, 20

## AVERAGE ACCURACY BY FEATURE EXTRACTION METHOD
- **Advanced Stats**: 0.935 ± 0.077 (144.0 experiments)
- **Hybrid**: 0.938 ± 0.075 (144.0 experiments)
- **WST**: 0.888 ± 0.087 (144.0 experiments)

## AVERAGE ACCURACY BY NOISE CONDITION
- **Clean**: 0.945 ± 0.059 (108.0 experiments)
- **S&P 15%**: 0.929 ± 0.083 (108.0 experiments)
- **S&P 25%**: 0.870 ± 0.090 (108.0 experiments)
- **S&P 5%**: 0.938 ± 0.073 (108.0 experiments)

## TOP 10 GLOBAL PERFORMANCES
- 1.000 | WST | assatigue | Mini | k=5 | S&P 5%
- 1.000 | Advanced Stats | assatigue | Small | k=2 | S&P 5%
- 1.000 | Hybrid | assatigue | Small | k=2 | S&P 5%
- 1.000 | Advanced Stats | assatigue | Small | k=5 | S&P 5%
- 1.000 | Hybrid | assatigue | Small | k=5 | S&P 5%
- 1.000 | Advanced Stats | assatigue | Small | k=10 | S&P 5%
- 1.000 | Hybrid | assatigue | Small | k=10 | S&P 5%
- 1.000 | Advanced Stats | assatigue | Small | k=20 | S&P 5%
- 1.000 | Hybrid | assatigue | Small | k=20 | S&P 5%
- 1.000 | Advanced Stats | assatigue | Original | k=5 | S&P 5%

## PERFORMANCE BY GEOGRAPHIC AREA
### ASSATIGUE
  - Advanced Stats: 0.942 ± 0.072
  - Hybrid: 0.944 ± 0.072
  - WST: 0.861 ± 0.082

### POPOLAR
  - Advanced Stats: 0.960 ± 0.063
  - Hybrid: 0.962 ± 0.054
  - WST: 0.900 ± 0.081

### SUNSET
  - Advanced Stats: 0.902 ± 0.083
  - Hybrid: 0.907 ± 0.085
  - WST: 0.904 ± 0.092

## SALT & PEPPER NOISE ROBUSTNESS ANALYSIS
### Average degradation per method (Clean → S&P 5%)
- Advanced Stats: -0.003 ± 0.046
- Hybrid: 0.011 ± 0.073
- WST: 0.014 ± 0.086

### Average degradation per method (Clean → S&P 15%)
- Advanced Stats: 0.000 ± 0.047
- Hybrid: 0.012 ± 0.086
- WST: 0.038 ± 0.090

### Average degradation per method (Clean → S&P 25%)
- Advanced Stats: 0.076 ± 0.099
- Hybrid: 0.082 ± 0.091
- WST: 0.068 ± 0.109
