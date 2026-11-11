
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

TREE_DEPTHS = [2, 3, 5, 10, None]
NEIGHBOR_COUNTS = [1, 3, 5, 7, 11]
REPRESENTATIONS = ["unscaled", "scaled"]


def make_models(seed):
    
    models = []
    for depth in TREE_DEPTHS:
        model = DecisionTreeClassifier(max_depth=depth, random_state=seed)
        models.append(("DecisionTree", f"max_depth={depth}", f"dt_depth{depth}", model))
    for k in NEIGHBOR_COUNTS:
        model = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")
        models.append(("KNN", f"n_neighbors={k}", f"knn_k{k}", model))
    models.append(("GaussianNB", "default", "gnb_default", GaussianNB()))
    return models


def undefined_metric_classes(y_true, y_pred, classes):
    
    notes = []
    for c in classes:
        if np.sum(y_pred == c) == 0:
            notes.append(f"precision undefined for class {c}")
        if np.sum(y_true == c) == 0:
            notes.append(f"recall undefined for class {c}")
    return "; ".join(notes)


def classification_metrics(y_true, y_pred, proba, classes, dataset_name):
    
    y_true = np.asarray(y_true)
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if dataset_name == "breast_cancer":
        malignant_target = (y_true == 0).astype(int)        # 1 = malignant, 0 = benign
        malignant_column = list(classes).index(0)           # P(orijinal sınıf 0)
        result["roc_auc"] = roc_auc_score(malignant_target, proba[:, malignant_column])
    else:
        result["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr",
                                          average="macro", labels=list(classes))
    return result


def run_classifier_experiments(dataset_name, splits, seeds):
    
    rows = []
    seed42_models = {}
    for seed in seeds:
        y_train = splits[seed]["y_train"]
        y_val = splits[seed]["y_val"]
        for representation in REPRESENTATIONS:
            X_train = splits[seed][representation]["train"]
            X_val = splits[seed][representation]["val"]
            for algorithm, parameter, short_name, model in make_models(seed):
                algo_part, param_part = short_name.split("_")
                config_id = f"{algo_part}_{representation}_{param_part}"
                run_id = f"{dataset_name}_{config_id}_s{seed}"

                model.fit(X_train, y_train)          # sadece eğitim verisiyle fit

                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                val_proba = model.predict_proba(X_val)
                val_scores = classification_metrics(y_val, val_pred, val_proba,
                                                    model.classes_, dataset_name)
                rows.append({
                    "dataset": dataset_name,
                    "run_id": run_id,
                    "config_id": config_id,
                    "seed": seed,
                    "algorithm": algorithm,
                    "representation": representation,
                    "parameter": parameter,
                    "train_accuracy": accuracy_score(y_train, train_pred),
                    "train_f1_macro": f1_score(y_train, train_pred, average="macro", zero_division=0),
                    "val_accuracy": val_scores["accuracy"],
                    "val_precision_macro": val_scores["precision_macro"],
                    "val_recall_macro": val_scores["recall_macro"],
                    "val_f1_macro": val_scores["f1_macro"],
                    "val_roc_auc": val_scores["roc_auc"],
                    "val_undefined_before_zero_division":
                        undefined_metric_classes(np.asarray(y_val), val_pred, model.classes_),
                })
                if seed == 42:
                    seed42_models[config_id] = model
    return pd.DataFrame(rows), seed42_models


def cluster_scores(X, y_true, clusters):
    """ARI ve silhouette hesaplar. Silhouette tanımsızsa NaN (N/A) döner."""
    ari = adjusted_rand_score(y_true, clusters)
    n_clusters_found = len(np.unique(clusters))
    if 2 <= n_clusters_found < len(clusters):
        silhouette = silhouette_score(X, clusters)
    else:
        silhouette = np.nan    # N/A: en az 2 farklı küme gerekir
    return ari, silhouette


def run_kmeans_experiments(dataset_name, splits, seeds, n_clusters):
    
    rows = []
    seed42_models = {}
    for seed in seeds:
        for representation in REPRESENTATIONS:
            X_train = splits[seed][representation]["train"]
            X_val = splits[seed][representation]["val"]
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            kmeans.fit(X_train)                       # etiket yok: unsupervised
            val_clusters = kmeans.predict(X_val)
            ari, silhouette = cluster_scores(X_val, splits[seed]["y_val"], val_clusters)
            rows.append({
                "dataset": dataset_name,
                "run_id": f"{dataset_name}_kmeans_{representation}_s{seed}",
                "seed": seed,
                "representation": representation,
                "n_clusters": n_clusters,
                "n_init": 10,
                "train_inertia": kmeans.inertia_,
                "val_ari": ari,
                "val_silhouette": silhouette,
            })
            if seed == 42:
                seed42_models[representation] = kmeans
    return pd.DataFrame(rows), seed42_models


def save_combined_csv(table, file_name, dataset_name, folder="results"):
   
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file_name)
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old["dataset"] != dataset_name]
        table = pd.concat([old, table], ignore_index=True)
    table = table.sort_values("dataset", kind="stable").reset_index(drop=True)
    table.to_csv(path, index=False)
    return path
