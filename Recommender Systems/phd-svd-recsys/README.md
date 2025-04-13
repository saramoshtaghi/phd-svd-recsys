
# 🎬 Movie Recommendation System with Biased Data Comparison

This project investigates how **modifying a training dataset** (injecting bias) affects the performance of a **Singular Value Decomposition (SVD)**-based movie recommender system. We use the **MovieLens 100k** dataset and a modified version (**101k**) to evaluate recommendation quality through multiple Top-N metrics.

---

## 📁 Project Structure

```
.
├── A-top10_recommendations_100k.csv     # Top-10 recommendations from Model A (100k)
├── B-top10_recommendations_101k.csv     # Top-10 recommendations from Model B (101k biased)
├── df_final.csv                         # Original 100k dataset
├── df101k.csv                           # Modified/biased dataset (101k)
├── model_training_and_evaluation.ipynb  # Main notebook (training, evaluation)
├── README.md                            # This file
```

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

- **df_final**: The original MovieLens 100k dataset.
- **df101k**: A modified version of the dataset with additional user interactions or injected bias.

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
jupyter notebook model_training_and_evaluation.ipynb
```

3. It will:
   - Train two SVD models
   - Generate Top-10 recommendations for users
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

---
