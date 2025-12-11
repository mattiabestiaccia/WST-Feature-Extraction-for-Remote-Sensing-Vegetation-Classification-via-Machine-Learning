# UNIFORM NOISE ANALYSIS SUMMARY
==================================================

Analysis generated on: 2025-07-22 11:28:36
Total experiments analyzed: 432

## GENERATED FILES

### Reports
- `uniform_comprehensive_report.md`: Complete statistical analysis
- `uniform_qualitative_analysis.md`: In-depth qualitative interpretation
- `uniform_experiments_summary.csv`: Raw data export

### Plots
- `comparisons/`: 4 comparison plots
- `detailed/`: 40 detailed plots (averaged over geographic areas only)

## KEY FINDINGS

- **Best performing method**: Hybrid (0.941 average accuracy)
- **Uniform noise impact**: 4.0% performance loss (Clean → Uniform ±40)

## METHODOLOGY

- **Geographic areas**: Results averaged over assatigue, popolar, sunset
- **Noise conditions**: clean, uniform10 (±10), uniform25 (±25), uniform40 (±40)
- **Dataset sizes**: mini, small, original
- **Feature selection**: k ∈ {2, 5, 10, 20}
- **Methods**: WST, Advanced RGB Statistics, Hybrid
