# POISSON NOISE ANALYSIS SUMMARY
==================================================

Analysis generated on: 2025-07-22 12:03:38
Total experiments analyzed: 324

## GENERATED FILES

### Reports
- `poisson_comprehensive_report.md`: Complete statistical analysis
- `poisson_qualitative_analysis.md`: In-depth qualitative interpretation
- `poisson_experiments_summary.csv`: Raw data export

### Plots
- `comparisons/`: 4 comparison plots
- `detailed/`: 33 detailed plots (averaged over geographic areas only)

## KEY FINDINGS

- **Best performing method**: Hybrid (0.930 average accuracy)
- **Poisson noise impact**: 4.3% performance loss (Clean → Poisson λ=60)

## METHODOLOGY

- **Geographic areas**: Results averaged over assatigue, popolar, sunset
- **Noise conditions**: clean, poisson40 (λ=40), poisson60 (λ=60)
- **Dataset sizes**: mini, small, original
- **Feature selection**: k ∈ {2, 5, 10, 20}
- **Methods**: WST, Advanced RGB Statistics, Hybrid
