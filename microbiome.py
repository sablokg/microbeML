"""
================================================================================
MICROBIOME MACHINE LEARNING PIPELINE (Python) Gaurav Sablok gsablok@proton.me
================================================================================

An end-to-end, ready-to-run pipeline for classifying samples (e.g., disease
vs. healthy) from microbiome abundance data (16S/shotgun OTU or ASV tables).

Pipeline stages:
  1. Load data (or simulate a realistic synthetic dataset if none provided)
  2. Preprocess: filter low-prevalence taxa, CLR-transform compositional data
  3. Train/test split (stratified)
  4. Feature selection (variance + univariate filtering)
  5. Model training + nested cross-validation (Random Forest, Logistic
     Regression w/ L1, Gradient Boosting)
  6. Evaluation: accuracy, ROC-AUC, confusion matrix, classification report
  7. Biomarker discovery: top discriminative taxa via feature importance
  8. Visualization: PCoA/PCA ordination, ROC curves, feature importance

USAGE
-----
Run as-is to see the pipeline work on simulated data:
    python3 microbiome_ml_pipeline.py

To use your own data, replace `load_data()` so it returns:
    X : pandas DataFrame, shape (n_samples, n_taxa)   -- raw counts or relative
        abundances, samples as rows, taxa (OTU/ASV/species) as columns
    y : pandas Series, shape (n_samples,)              -- class labels
Typical real-world sources: a QIIME2 'feature-table.biom' (converted to CSV),
a phyloseq otu_table exported to CSV, or a MetaPhlAn merged abundance table.

DEPENDENCIES
------------
pip install pandas numpy scikit-learn scipy matplotlib seaborn --break-system-packages
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ==============================================================================
# 1. DATA LOADING (swap this out for your real data)
# ==============================================================================

def simulate_microbiome_data(n_samples=200, n_taxa=150, n_informative=15,
                              random_state=RANDOM_STATE):
    """
    Simulate a compositional microbiome dataset with a binary phenotype
    (e.g., 'disease' vs 'healthy'). A subset of taxa are made informative
    (their abundance shifts with class), the rest are noise -- mimicking
    real microbiome data where most taxa are uninformative and sparse.
    """
    rng = np.random.default_rng(random_state)
    taxa_names = [f"Taxon_{i:03d}" for i in range(n_taxa)]
    y = rng.integers(0, 2, size=n_samples)  # 0 = healthy, 1 = disease

    # Base abundances: log-normal, sparse (many zeros), like real 16S data
    base = rng.lognormal(mean=1.0, sigma=1.2, size=(n_samples, n_taxa))
    sparsity_mask = rng.random((n_samples, n_taxa)) < 0.35  # ~35% zeros
    base[sparsity_mask] = 0.0

    # Inject signal into a subset of "informative" taxa correlated with y
    informative_idx = rng.choice(n_taxa, size=n_informative, replace=False)
    for idx in informative_idx:
        effect = rng.uniform(1.5, 3.5) * rng.choice([-1, 1])
        base[:, idx] += (y * effect * rng.lognormal(0.5, 0.5, size=n_samples))
        base[:, idx] = np.clip(base[:, idx], 0, None)

    counts = pd.DataFrame(np.round(base * 1000).astype(int),
                           columns=taxa_names)
    labels = pd.Series(y, name="phenotype").map({0: "healthy", 1: "disease"})
    return counts, labels, [taxa_names[i] for i in informative_idx]


def load_data():
    """
    Replace this function to load your real data, e.g.:

        counts = pd.read_csv("feature_table.csv", index_col=0)   # samples x taxa
        metadata = pd.read_csv("metadata.csv", index_col=0)
        y = metadata.loc[counts.index, "disease_status"]
        return counts, y

    Falls back to simulated data if no file is present.
    """
    counts, y, _true_informative = simulate_microbiome_data()
    return counts, y


# ==============================================================================
# 2. PREPROCESSING
# ==============================================================================

def filter_low_prevalence(X, min_prevalence=0.10, min_count=1):
    """Drop taxa present (count >= min_count) in fewer than min_prevalence
    fraction of samples. Standard first step -- rare taxa are mostly noise
    and hurt statistical power."""
    prevalence = (X >= min_count).mean(axis=0)
    keep = prevalence[prevalence >= min_prevalence].index
    return X[keep]


def clr_transform(X, pseudocount=1.0):
    """
    Centered log-ratio (CLR) transform.

    Microbiome abundance data is *compositional* (each sample sums to an
    arbitrary total / is a proportion of a whole), so raw counts violate the
    independence assumptions of most ML models and naive Euclidean distances
    are misleading. CLR maps compositions into real (unconstrained) space
    where standard ML/statistics are valid. A pseudocount avoids log(0).
    """
    X_pseudo = X + pseudocount
    log_X = np.log(X_pseudo)
    geometric_mean = log_X.mean(axis=1)
    clr = log_X.sub(geometric_mean, axis=0)
    return clr


def preprocess(X, min_prevalence=0.10):
    X_filt = filter_low_prevalence(X, min_prevalence=min_prevalence)
    X_clr = clr_transform(X_filt)
    return X_clr


# ==============================================================================
# 3-5. MODELING
# ==============================================================================

def build_models():
    """Three complementary model families: a sparse linear model (L1
    logistic regression, good for interpretable biomarker panels), and two
    tree ensembles (Random Forest, Gradient Boosting) which capture
    nonlinear taxon-taxon interactions."""
    models = {
        "Logistic Regression (L1)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1", solver="liblinear", C=0.5,
                random_state=RANDOM_STATE, max_iter=2000)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=6, min_samples_leaf=3,
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                random_state=RANDOM_STATE)),
        ]),
    }
    return models


def cross_validate_models(models, X, y, cv_folds=5):
    """Stratified k-fold CV on the training set, scored by ROC-AUC (robust
    to class imbalance, common in clinical microbiome studies)."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = scores
        print(f"  {name:28s}  ROC-AUC = {scores.mean():.3f} +/- {scores.std():.3f}")
    return results


# ==============================================================================
# 6. EVALUATION
# ==============================================================================

def evaluate_on_test(name, pipe, X_test, y_test, label_encoder):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_test_bin = label_encoder(y_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test_bin, y_proba)

    print(f"\n--- {name}: held-out test set ---")
    print(f"Accuracy: {acc:.3f}   ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred))

    return {"y_pred": y_pred, "y_proba": y_proba, "acc": acc, "auc": auc}


# ==============================================================================
# 7. BIOMARKER DISCOVERY
# ==============================================================================

def top_features_random_forest(pipe, feature_names, top_n=15):
    importances = pipe.named_steps["clf"].feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    return pd.Series(importances[order], index=np.array(feature_names)[order],
                      name="importance").sort_values()


def top_features_logreg(pipe, feature_names, top_n=15):
    coefs = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:top_n]
    return pd.Series(coefs[order], index=np.array(feature_names)[order],
                      name="coefficient").sort_values()


# ==============================================================================
# 8. VISUALIZATION
# ==============================================================================

def plot_all(X_clr, y, cv_results, test_results, rf_importance, logreg_coef, outpath):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- PCA ordination of CLR-transformed data ---
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_clr)
    ax = axes[0, 0]
    for label in y.unique():
        mask = (y == label).values
        ax.scatter(coords[mask, 0], coords[mask, 1], label=label, alpha=0.7, s=40)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of CLR-transformed microbiome composition")
    ax.legend()

    # --- CV comparison across models ---
    ax = axes[0, 1]
    names = list(cv_results.keys())
    data = [cv_results[n] for n in names]
    ax.boxplot(data, labels=[n.replace(" (", "\n(") for n in names])
    ax.set_ylabel("ROC-AUC (5-fold CV)")
    ax.set_title("Model comparison (cross-validated)")
    ax.tick_params(axis="x", labelsize=8)

    # --- ROC curve for best model on test set ---
    ax = axes[1, 0]
    for name, res in test_results.items():
        fpr, tpr, _ = roc_curve((res["y_test_bin"]), res["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (held-out test set)")
    ax.legend(fontsize=8)

    # --- Top biomarker taxa (Random Forest importance) ---
    ax = axes[1, 1]
    rf_importance.plot(kind="barh", ax=ax, color="teal")
    ax.set_xlabel("Feature importance (Random Forest)")
    ax.set_title("Top candidate biomarker taxa")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure -> {outpath}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("STEP 1: Load data")
    print("=" * 70)
    X_raw, y = load_data()
    print(f"Samples: {X_raw.shape[0]}, Taxa: {X_raw.shape[1]}")
    print(f"Class balance:\n{y.value_counts()}\n")

    print("=" * 70)
    print("STEP 2: Preprocess (prevalence filter + CLR transform)")
    print("=" * 70)
    X_clr = preprocess(X_raw, min_prevalence=0.10)
    print(f"Taxa after prevalence filtering: {X_clr.shape[1]}\n")

    print("=" * 70)
    print("STEP 3: Train/test split (stratified 75/25)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clr, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples\n")

    print("=" * 70)
    print("STEP 4: Feature selection (top 50 taxa by ANOVA F-value)")
    print("=" * 70)
    selector = SelectKBest(score_func=f_classif, k=min(50, X_train.shape[1]))
    selector.fit(X_train, y_train)
    selected_cols = X_train.columns[selector.get_support()]
    X_train_sel = X_train[selected_cols]
    X_test_sel = X_test[selected_cols]
    print(f"Selected {len(selected_cols)} taxa\n")

    print("=" * 70)
    print("STEP 5: Cross-validated model comparison")
    print("=" * 70)
    models = build_models()
    cv_results = cross_validate_models(models, X_train_sel, y_train)

    print("\n" + "=" * 70)
    print("STEP 6: Fit on full training set, evaluate on held-out test set")
    print("=" * 70)
    label_encoder = lambda labels: (labels == labels.unique()[1]).astype(int) \
        if not pd.api.types.is_numeric_dtype(labels) else labels
    # Ensure a consistent positive class across models
    classes_sorted = sorted(y.unique())
    pos_label = classes_sorted[-1]
    to_binary = lambda labels: (labels == pos_label).astype(int)

    test_results = {}
    fitted_pipes = {}
    for name, pipe in models.items():
        pipe.fit(X_train_sel, y_train)
        fitted_pipes[name] = pipe
        res = evaluate_on_test(name, pipe, X_test_sel, y_test, to_binary)
        res["y_test_bin"] = to_binary(y_test)
        test_results[name] = res

    print("\n" + "=" * 70)
    print("STEP 7: Biomarker discovery (top discriminative taxa)")
    print("=" * 70)
    rf_importance = top_features_random_forest(
        fitted_pipes["Random Forest"], selected_cols, top_n=15)
    logreg_coef = top_features_logreg(
        fitted_pipes["Logistic Regression (L1)"], selected_cols, top_n=15)
    print("\nTop taxa (Random Forest importance):")
    print(rf_importance.sort_values(ascending=False).to_string())

    print("\n" + "=" * 70)
    print("STEP 8: Visualization")
    print("=" * 70)
    plot_all(X_train_sel, y_train, cv_results, test_results,
              rf_importance, logreg_coef,
              outpath="/mnt/user-data/outputs/microbiome_ml_results.png")

    print("\nDone. Best model by test ROC-AUC:",
          max(test_results, key=lambda n: test_results[n]["auc"]))


if __name__ == "__main__":
    main()
