# Vegetation Classification & Methodology Comparison Study Context

## 1. Scientific Overview & Objectives
**Role**: Researcher (Vegetation Image Classification & Species Recognition).
**Goal**: Evaluate and validatestatistically the effectiveness of different feature extraction methods for classifying vegetation species/land cover types using Random Forest.

### Key Research Questions
- How does **Wavelet Scattering Transform (WST)** compare to traditional **RGB Statistical Features** in terms of classification accuracy?
- Can **Hybrid methods** (RGB + WST) offer superior performance?
- How robust are these methods against various types of image noise (Gaussian, Poisson, Speckle, etc.)?
- What is the impact of training dataset size (Data Scarcity) on performance, and is there a saturation point?

## 2. Methodology & Experimental Design

### Classification Target
Classes identified (Land Cover Areas):
1.  **Assatigue** (Agricultural)
2.  **Popolar** (Forest/Woodland)
3.  **Sunset** (Urban/Built-up)

### Feature Extraction Candidates
1.  **RGB Statistics**: Baseline, low complexity.
2.  **Wavelet Scattering Transform (WST)**: Capture texture/invariance, higher complexity.
3.  **Hybrid**: Concatenation of RGB and WST features.

### Experimental Variables (Grid Search)
The study relies on a massive grid search (~1,500 experiments) exploring:
-   **Noise Conditions (Robustness)**:
    -   Clean (Baseline)
    -   Gaussian (varied $\sigma$)
    -   Poisson
    -   Salt & Pepper
    -   Speckle
    -   Uniform
-   **Dataset Size (Data Efficiency)**:
    -   Mini (~10 samples/class)
    -   Small (~50 samples/class)
    -   Original (~200 samples/class)
    -   *(Augmented was tested but excluded due to diminishing returns)*
-   **Feature Selection**: Top-$k$ features ($k \in \{2, 5, 10, 20\}$).

### Algorithm
-   **Classifier**: Random Forest (robust, interpretable).
-   **Validation**: Stratified Cross-Validation (likely 5-fold).

## 3. Project Structure & State

### Phase 1: Experiment Generation (Completed)
-   **Location**: `experiments/` (organized by noise type $\rightarrow$ area $\rightarrow$ size $\rightarrow$ k $\rightarrow$ method).
-   **Output**: JSON reports containing metrics (Accuracy, F1, etc.) for each permutation.
-   **Scale**: Systematically generated experiments covering the defined grid.

### Phase 2: Statistical Analysis (Current Focus)
-   **Goal**: Move beyond raw metrics to statistical validation.
-   **Requirement**: Assert dominance of one method over others with statistical significance.
-   **Tools/Scripts**: Located in `scripts/analysis` or `analisi`.

### Phase 3: Publication (In Progress)
-   **Location**: `article/article_latex/`.
-   **Format**: LaTeX.
-   **Notes**: The LaTeX source contains "redundancies" (likely `main_old.tex` or structural repetitions) intentionally kept to facilitate the writing process.
-   **Content**: "Materials and Methods" presumably drafted.

## 4. Key Preliminary Findings (Contextual Knowledge)
-   **WST Dominance**: Generally outperforms RGB, especially under noise.
-   **Data Plateau**: Performance gains level off at ~200 images/class (Original dataset), rendering massive augmentation less critical for this specific task/resolution.
-   **Feature Selection**: $k=10-20$ features often sufficient.
