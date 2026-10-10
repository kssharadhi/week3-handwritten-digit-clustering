"""Week 3 project: grouping handwritten digit images and evaluating a classifier.
Run: python main.py
Uses the real digits dataset packaged with scikit-learn. No internet required.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Save figures, even when run without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

RANDOM_STATE = 42
OUT = Path(__file__).resolve().parent / 'outputs'
OUT.mkdir(exist_ok=True)


def save_chart(filename):
    plt.savefig(OUT / filename, dpi=155, bbox_inches='tight')
    plt.close()


def main():
    digits = load_digits()
    X, y = digits.data, digits.target
    assert X.shape == (1797, 64)
    print(f'Dataset: {len(y)} handwritten images, {X.shape[1]} pixel features, 10 digits')

    # Preview the dataset to see what the 8x8 images actually look like.
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for label, ax in enumerate(axes.flat):
        idx = np.flatnonzero(y == label)[0]
        ax.imshow(digits.images[idx], cmap='gray_r')
        ax.set_title(f'Digit {label}')
        ax.axis('off')
    fig.suptitle('Sample handwritten images from the dataset')
    fig.tight_layout()
    save_chart('01_digit_examples.png')

    # UNsupervised learning: clustering does not use the true digit labels.
    X_scaled = StandardScaler().fit_transform(X)
    cluster_rows = []
    for k in [6, 8, 10, 12]:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=3, max_iter=150)
        cluster_labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels, sample_size=350, random_state=RANDOM_STATE)
        cluster_rows.append({'number_of_clusters': k, 'silhouette_score': round(float(score), 4)})
        print(f'K-Means k={k}: silhouette={score:.4f}')

    cluster_scores = pd.DataFrame(cluster_rows)
    cluster_scores.to_csv(OUT / 'kmeans_scores.csv', index=False)
    # Digits 0-9 imply 10 expected visual categories; silhouette may prefer a different k.
    kmeans = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=3, max_iter=150)
    km_labels = kmeans.fit_predict(X_scaled)
    kmeans_sil = silhouette_score(X_scaled, km_labels, sample_size=350, random_state=RANDOM_STATE)

    hierarchical = AgglomerativeClustering(n_clusters=10, linkage='ward')
    hier_labels = hierarchical.fit_predict(X_scaled)
    hier_sil = silhouette_score(X_scaled, hier_labels, sample_size=350, random_state=RANDOM_STATE)

    plt.figure(figsize=(7.2, 4.1))
    plt.plot(cluster_scores.number_of_clusters, cluster_scores.silhouette_score, marker='o')
    plt.xticks(cluster_scores.number_of_clusters)
    plt.xlabel('Number of groups (k)')
    plt.ylabel('Silhouette score (higher is better)')
    plt.title('K-Means: silhouette score for different k values')
    plt.grid(alpha=.25)
    save_chart('02_silhouette_by_k.png')

    print(f'Hierarchical k=10: silhouette={hier_sil:.4f}')
    reduced = PCA(n_components=2, svd_solver='randomized', random_state=RANDOM_STATE).fit_transform(X_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
    axes[0].scatter(reduced[:, 0], reduced[:, 1], c=km_labels, s=10, cmap='tab10', alpha=.75)
    axes[0].set_title('K-Means groups (without labels)')
    axes[1].scatter(reduced[:, 0], reduced[:, 1], c=y, s=10, cmap='tab10', alpha=.75)
    axes[1].set_title('Actual digit labels (for comparison only)')
    for ax in axes:
        ax.set_xlabel('Principal component 1')
        ax.set_ylabel('Principal component 2')
    fig.suptitle('PCA projection of 64 pixel features into 2 dimensions')
    fig.tight_layout()
    save_chart('03_pca_cluster_comparison.png')

    # The heatmap is a simple way to discuss mixed clusters, not training labels.
    cross = pd.crosstab(pd.Series(km_labels, name='Cluster'), pd.Series(y, name='Digit'))
    cross.to_csv(OUT / 'kmeans_cluster_digit_counts.csv')
    fig, ax = plt.subplots(figsize=(8.3, 5))
    heat = ax.imshow(cross.values, aspect='auto', cmap='Blues')
    fig.colorbar(heat, ax=ax, label='Number of images')
    ax.set_xticks(np.arange(10), labels=np.arange(10))
    ax.set_yticks(np.arange(10), labels=np.arange(10))
    ax.set_xlabel('True handwritten digit')
    ax.set_ylabel('K-Means cluster ID (arbitrary)')
    ax.set_title('Which actual digits ended up in each K-Means group?')
    fig.tight_layout()
    save_chart('04_digit_cluster_heatmap.png')

    # Dendrogram for a small random sample, to keep the picture understandable.
    selected = np.random.default_rng(RANDOM_STATE).choice(len(y), size=80, replace=False)
    links = linkage(X_scaled[selected], method='ward')
    plt.figure(figsize=(10, 4.2))
    dendrogram(links, no_labels=True, color_threshold=None)
    plt.title('Hierarchical clustering: Ward dendrogram (80 sampled images)')
    plt.xlabel('Sampled images')
    plt.ylabel('Merge distance')
    save_chart('05_hierarchical_dendrogram.png')

    # SUPERVISED evaluation: labels are needed to check prediction accuracy.
    # Divide FIRST, then scale inside the pipeline to prevent leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier()),
    ])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tune = GridSearchCV(
        pipe, {'model__n_neighbors': [1, 3, 5, 7]},
        cv=folds, scoring='f1_macro', n_jobs=1, refit=True
    )
    tune.fit(X_train, y_train)
    pred = tune.predict(X_test)
    probability = tune.predict_proba(X_test)

    cm = confusion_matrix(y_test, pred, labels=np.arange(10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title('Final model: confusion matrix on the held-out test set')
    fig.tight_layout()
    save_chart('06_confusion_matrix.png')

    binary_y = label_binarize(y_test, classes=np.arange(10))
    false_positive_rate, true_positive_rate, _ = roc_curve(binary_y.ravel(), probability.ravel())
    auc_micro = roc_auc_score(binary_y.ravel(), probability.ravel())
    auc_macro = roc_auc_score(y_test, probability, multi_class='ovr', average='macro')
    plt.figure(figsize=(6.5, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f'Micro-average ROC (AUC = {auc_micro:.3f})')
    plt.plot([0, 1], [0, 1], '--', label='Chance baseline')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('One-vs-rest ROC, with all 10 digits combined')
    plt.legend(loc='lower right')
    plt.grid(alpha=.2)
    save_chart('07_roc_curve.png')

    report = classification_report(y_test, pred, digits=3)
    (OUT / 'classification_report.txt').write_text(report, encoding='utf-8')
    metrics = {
        'dataset_size': int(len(y)),
        'pixel_features': int(X.shape[1]),
        'training_samples': int(len(y_train)),
        'test_samples': int(len(y_test)),
        'kmeans_k10_silhouette': round(float(kmeans_sil), 4),
        'hierarchical_k10_silhouette': round(float(hier_sil), 4),
        'baseline_test_accuracy': round(float(baseline_acc), 4),
        'best_knn_neighbors': int(tune.best_params_['model__n_neighbors']),
        'best_cv_macro_f1': round(float(tune.best_score_), 4),
        'test_accuracy': round(float(accuracy_score(y_test, pred)), 4),
        'test_macro_precision': round(float(precision_score(y_test, pred, average='macro', zero_division=0)), 4),
        'test_macro_recall': round(float(recall_score(y_test, pred, average='macro', zero_division=0)), 4),
        'test_macro_f1': round(float(f1_score(y_test, pred, average='macro', zero_division=0)), 4),
        'test_macro_roc_auc_ovr': round(float(auc_macro), 4),
        'test_micro_roc_auc_ovr': round(float(auc_micro), 4),
        'pca_2d_variance_explained': round(float(PCA(n_components=2, svd_solver='randomized', random_state=RANDOM_STATE).fit(X_scaled).explained_variance_ratio_.sum()), 4),
        'random_state': RANDOM_STATE,
    }
    (OUT / 'results.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('\nClassification report on 360 held-out examples:\n' + report)
    print('Results saved to:', OUT)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()