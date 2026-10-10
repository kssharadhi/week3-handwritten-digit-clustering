# Handwritten Digit Pattern Discovery and Recognition

**Week 3: Unsupervised Learning and Model Evaluation**

A small Python machine learning project that compares how digit images are grouped **without labels** with how well a model recognizes digits **with labels**.

## What the project does

1. Loads the genuine 1,797 handwritten-digit samples included in scikit-learn (8 x 8 images, 64 pixel values per image).
2. Scales pixel data and runs **K-Means** for k = 6, 8, 10 and 12. Measures silhouette scores.
3. Runs **Hierarchical Clustering (Ward linkage)** with 10 groups and draws an 80-image dendrogram.
4. Uses **PCA** to display the 64-dimensional data in 2D and compares clusters with true digit labels. These labels are *not used to create clusters*.
5. Splits data into training (80%) and testing (20%) with stratification.
6. Uses **5-fold stratified cross-validation** and **GridSearchCV** to find the best number of neighbors for a **K-Nearest Neighbors (KNN)** classifier.
7. Reports a confusion matrix, accuracy, macro precision, macro recall, macro F1, and multiclass one-vs-rest ROC-AUC.

**Why KNN?** It is simple to understand and explain: it predicts a digit by looking at the most similar labeled training images. That makes it appropriate for a week-three learning project.

## Run locally

From a terminal in this directory:

```bash
python -m pip install -r requirements.txt
python main.py
```

For Anaconda Prompt on Windows, you can make a separate environment first:

```bash
conda create -n week3digits python=3.11 -y
conda activate week3digits
pip install -r requirements.txt
python main.py
```

No dataset download, API key, or GPU is required.

## Actual measured results

Results obtained by running the included code with `random_state=42`:

| Metric | Result |
|---|---:|
| Dataset size | 1,797 |
| Training / test images | 1,437 / 360 |
| K-Means silhouette (k=10) | 0.1468 |
| Hierarchical silhouette (k=10) | 0.1376 |
| Best KNN neighbors | 3 |
| Best 5-fold CV macro F1 | 0.9762 |
| Held-out test accuracy | 0.9667 |
| Test macro precision | 0.9678 |
| Test macro recall | 0.9664 |
| Test macro F1 | 0.9663 |
| Test macro ROC-AUC (OvR) | 0.9907 |

The classifier beats a 10% most-frequent-label baseline. The K-Means silhouette score remains modest, which shows that handwritten digits do not always form clear-cut natural clusters. The best silhouette among the *four tested values* was k=12, even though there are 10 digit labels; these are different ideas and neither alone proves the ideal number of clusters.

## Output files

Run the code to regenerate the `outputs/` folder:

- `01_digit_examples.png` — example images
- `02_silhouette_by_k.png` — K-Means k comparison
- `03_pca_cluster_comparison.png` — cluster labels vs. true labels (used only after clustering)
- `04_digit_cluster_heatmap.png` — counts of true digits inside K-Means clusters
- `05_hierarchical_dendrogram.png` — sampled hierarchical tree
- `06_confusion_matrix.png` — holdout predictions
- `07_roc_curve.png` — micro-average one-vs-rest ROC curve
- `kmeans_scores.csv`, `kmeans_cluster_digit_counts.csv`, `classification_report.txt`, `results.json`

## What I noticed

- Clusters containing most 0s and 6s were fairly clear.
- Some 1s, 8s and 9s overlapped in the unsupervised groups.
- The two PCA axes retain only about 21.59% of standardized feature variance, so the scatterplot does **not** preserve every difference between the original images.
- KNN classified digits much better than K-Means grouped them because it had access to labels during training.

## Limitations

This is a classroom experiment using a fixed public dataset, not a deployed OCR system. Digits were already normalized to 8 x 8. Accuracy on new camera or phone photos can be very different. Testing a new set of writers and trying more clustering settings would be useful follow-up tasks.

## Dataset credit

Scikit-learn `load_digits()` (a copy of the UCI handwritten digits test dataset). See the dataset documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
