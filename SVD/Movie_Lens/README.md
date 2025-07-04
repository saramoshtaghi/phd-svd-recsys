
# 🎬 Movie Recommendation System with Biased Data Comparison

This project investigates how **modifying a training dataset** (injecting bias) affects the performance of a **Singular Value Decomposition (SVD)**-based movie recommender system. We use the **MovieLens 100k** dataset and a several modified version to evaluate recommendation quality through multiple Top-N metrics.

---

## 📁 Project Structure

```
.to be annonced

---

## 📊 Objectives

- Train and evaluate **SVD models** on both **original** and **biased** data.
- Generate Top-10 recommendations per user.
- Compare the models using:
  - **Hit Rate**
  - **Precision@10**
  - **Recommendation Overlap**
  - **Diversity**

---

## 🧪 Datasets

- **df_0**: The original MovieLens 100k dataset.
- **df_40**: A modified version of the dataset with additional 40 user interactions or injected bias.
- **df_80**: A modified version of the dataset with additional 80 user interactions or injected bias.
- **df_120**: A modified version of the dataset with additional 120 user interactions or injected bias.

---

## 🛠️ Technologies Used

- Python 3.x
- [Surprise](http://surpriselib.com/) (for building recommender models)
- Pandas, NumPy
- Matplotlib (optional for visualizations)

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install scikit-surprise pandas
```

2. Run the notebook:
```bash
jupyter notebook SVD.ipynb
```

3. It will:
   - Train SVD models
   - Generate Top-10 recommendations for users for each datasets
   - Compute evaluation metrics
   - Save results to CSV files

---

## 📈 Evaluation Metrics

| Metric                | Description |
|-----------------------|-------------|
| **Hit Rate**          | Measures how many users had at least 1 correct recommendation. |
| **Precision@10**      | Measures the proportion of recommended items that were relevant. |
| **Recommendation Overlap** | Compares the overlap between models A and B’s recommendations. |
| **Diversity**         | Measures how varied the recommendations are (less repetition = more diverse). |

---

## 📌 Example Output

```plaintext
📊 Evaluation Summary
Model A (100k)     → Hit Rate: 0.1230, Precision@10: 0.0452, Diversity: 0.6021  
Model B (101k bias)→ Hit Rate: 0.1700, Precision@10: 0.0634, Diversity: 0.5254  
🔁 Recommendation Overlap Between A & B: 0.3421
```
---

## 🤔 Why This Matters

This experiment explores a **data-centric approach** to recommender systems. Instead of changing the model, we study how **adding or altering data** affects performance, helping us understand:

- How biased signals impact generalization
- Whether synthetic interactions improve accuracy
- How to balance diversity vs. precision

---

## 📬 Contact

For questions or contributions, reach out to [Your Name] at [moshtasa@mail.uc.edu]




# diagram
                           EXPERIMENTAL DESIGN PIPELINE

+--------------------------------------------------------+
|                 Original Dataset (df_final)            |
+--------------------------------------------------------+
                |        
                | Add synthetic users (poisoning)  
                v
+--------------------------------------------------------+
|         Synthetic Datasets for Each Decade             |
|  e.g. df_40_1980, df_80_1920, df_120_1990, etc.        |
+--------------------------------------------------------+
                |
                | Train separate SVD for each dataset
                v
+--------------------------------------------------------+
|             SVD Training on each dataset               |
+--------------------------------------------------------+
                |
                | Generate recommendations 
                | Always test on same df_final users
                v
+--------------------------------------------------------+
|      Generate Top-25 Recommendations (per dataset)     |
+--------------------------------------------------------+
                |
                | Merge recommendation results with decade info
                v
+--------------------------------------------------------+
|      Recommendation Output Files (per dataset):        |
|  e.g. top25_df_40_1980_with_decade.csv, etc.           |
+--------------------------------------------------------+
                |
                | Perform full analysis:
                v
+--------------------------------------------------------+
|     Entropy Calculation & Cluster Analysis:            |
|  - Decade distributions (count & %)                    |
|  - Recommendation entropy per user                     |
|  - Merge with cluster info                             |
|  - Plot: histograms, distributions, cluster comparisons|
+--------------------------------------------------------+

