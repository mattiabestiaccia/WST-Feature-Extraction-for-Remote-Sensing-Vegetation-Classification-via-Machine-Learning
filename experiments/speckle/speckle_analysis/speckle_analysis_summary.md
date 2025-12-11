# SPECKLE NOISE ANALYSIS SUMMARY
==================================================

Analysis generated on: 2025-07-22 11:28:11
Total experiments analyzed: 432

## GENERATED FILES

### Reports
- `speckle_comprehensive_report.md`: Complete statistical analysis
- `speckle_qualitative_analysis.md`: In-depth qualitative interpretation
- `speckle_experiments_summary.csv`: Raw data export

### Plots
- `comparisons/`: 4 comparison plots
- `detailed/`: 40 detailed plots (averaged over geographic areas only)

## KEY FINDINGS

- **Best performing method**: Hybrid (0.929 average accuracy)
- **Speckle noise impact**: 5.5% performance loss (Clean → Speckle ν=0.55)

## METHODOLOGY

- **Geographic areas**: Results averaged over assatigue, popolar, sunset
- **Noise conditions**: clean, speckle15 (ν=0.15), speckle35 (ν=0.35), speckle55 (ν=0.55)
- **Dataset sizes**: mini, small, original
- **Feature selection**: k ∈ {2, 5, 10, 20}
- **Methods**: WST, Advanced RGB Statistics, Hybrid
