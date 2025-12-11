# SALT & PEPPER NOISE ANALYSIS SUMMARY
==================================================

Analysis generated on: 2025-07-22 17:47:09
Total experiments analyzed: 432

## GENERATED FILES

### Reports
- `saltpepper_comprehensive_report.md`: Complete statistical analysis
- `saltpepper_qualitative_analysis.md`: In-depth qualitative interpretation
- `saltpepper_experiments_summary.csv`: Raw data export

### Plots
- `comparisons/`: 4 comparison plots
- `detailed/`: 40 detailed plots (averaged over geographic areas only)

## KEY FINDINGS

- **Best performing method**: Hybrid (0.938 average accuracy)
- **Salt & pepper noise impact**: 7.5% performance loss (Clean → S&P 25%)

## METHODOLOGY

- **Geographic areas**: Results averaged over assatigue, popolar, sunset
- **Noise conditions**: clean, saltpepper5 (5%), saltpepper15 (15%), saltpepper25 (25%)
- **Dataset sizes**: mini, small, original
- **Feature selection**: k ∈ {2, 5, 10, 20}
- **Methods**: WST, Advanced RGB Statistics, Hybrid
