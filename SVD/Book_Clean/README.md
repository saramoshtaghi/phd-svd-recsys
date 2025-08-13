# Book Recommendation System - Bias Injection Study

This project investigates bias propagation in SVD-based collaborative filtering systems by injecting synthetic users with strong genre preferences (Adventure/Mystery).

## 🗂️ Project Structure

```
Book_Clean/
├── data/
│   ├── original/           # Clean original Goodbooks 10K dataset
│   ├── biased_experiments/ # Biased datasets (generated)
│   └── processed/         # Preprocessed clean data
├── notebooks/             # Analysis notebooks (run in order)
│   ├── 01_data_exploration.ipynb      # Dataset exploration & genre identification
│   ├── 02_baseline_svd.ipynb          # Baseline SVD model (no bias)
│   ├── 03_bias_injection.ipynb        # Bias injection experiments
│   └── 04_bias_analysis.ipynb         # Results analysis
├── results/
│   ├── baseline/          # Baseline model results
│   └── biased/           # Bias injection experiment results
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-surprise scipy
```

### Running the Analysis

**IMPORTANT: Run notebooks in order!**

1. **Data Exploration** (`01_data_exploration.ipynb`)
   - Explores the Goodbooks 10K dataset
   - Identifies Adventure and Mystery books using tags
   - Creates clean processed datasets

2. **Baseline SVD** (`02_baseline_svd.ipynb`)
   - Trains SVD model on original data only
   - Establishes performance baselines
   - Saves model and test set for consistent comparison

3. **Bias Injection** (`03_bias_injection.ipynb`)
   - Creates synthetic biased users (50-2000 users)
   - Trains SVD models with bias injection
   - Tests: Adventure bias, Mystery bias
   - Measures recommendation changes

4. **Results Analysis** (`04_bias_analysis.ipynb`)
   - Analyzes bias propagation effects
   - Identifies threshold effects
   - Quantifies cross-genre contamination
   - Generates comprehensive report

## 🧪 Experimental Design

### Key Principles
- **Original data integrity**: Test set never contaminated
- **Consistent evaluation**: Same test set used for all experiments
- **Controlled bias injection**: Only training data receives synthetic users
- **Multiple bias levels**: 50, 100, 200, 500, 1000, 2000 biased users

### Bias Creation Process
1. Generate synthetic user IDs (beyond original range)
2. Assign 20-50 ratings per biased user
3. Focus ratings on target genre books
4. Use positive rating distribution (3-5 stars, weighted toward 4-5)
5. Add to training data only (never test data)

### Metrics Tracked
- **Performance**: RMSE, MAE overall and by genre
- **Recommendation bias**: % of recommendations by genre
- **Bias propagation**: Changes in recommendation patterns for original users
- **Cross-genre effects**: How Adventure bias affects Mystery recommendations

## 📊 Expected Outcomes

This study will reveal:
- **Robustness**: How resistant SVD is to biased user injection
- **Propagation thresholds**: Minimum biased users needed for measurable impact
- **Amplification effects**: How small biases get amplified through collaborative filtering
- **Attack feasibility**: Practical requirements for recommendation manipulation

## ⚠️ Important Notes

- This is a **clean restart** of the bias injection experiment
- Original dataset remains **uncontaminated** throughout
- All bias injection happens only in **training data**
- **Same test set** used for all experiments to ensure fair comparison
- Results are **reproducible** with fixed random seeds

## 🔬 Research Applications

- Understanding bias propagation in recommender systems
- Developing bias detection and mitigation strategies
- Informing robustness requirements for production systems
- Contributing to fairness and transparency research in ML

---

**Next Steps**: Start with `01_data_exploration.ipynb` and follow the numbered sequence!
