# COMPLETE ANALYSIS SUMMARY
==================================================

Analysis generated on: 2025-07-20 18:52:15
Total experiments analyzed: 324

## GENERATED FILES

### Reports
- `comprehensive_report.md`: Complete statistical analysis
- `qualitative_analysis.md`: In-depth qualitative interpretation
- `experiments_summary_english.csv`: Raw data export

### Plots
- `comparisons/`: 4 comparison plots (averaged over multiple dimensions)
- `detailed/`: 33 detailed plots (averaged over geographic areas only)

## KEY FINDINGS

- **Best performing method**: WST (0.913 average accuracy)
- **Noise impact**: 11.4% performance loss (Clean → Gaussian σ=50)
- **Dataset size effect**: 7.4% improvement (Mini → Original)

## METHODOLOGY

- **Geographic areas**: Results averaged over assatigue, popolar, sunset
- **Noise conditions**: clean, gaussian30 (σ=30), gaussian50 (σ=50)
- **Dataset sizes**: mini, small, original
- **Feature selection**: k ∈ {2, 5, 10, 20}
- **Methods**: WST, Advanced RGB Statistics, Hybrid
