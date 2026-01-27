"""
Metrics - Fonctions d'evaluation pour le Credit Scoring
Projet: Credit Scoring - Home Credit Default Risk

Fonctions:
- evaluate_model: Calcule toutes les metriques
- print_metrics: Affiche les metriques formatees
- plot_roc_curve: Trace la courbe ROC
- plot_confusion_matrix: Trace la matrice de confusion
- plot_feature_importance: Trace l'importance des features
- compare_models: Compare plusieurs modeles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from typing import Dict, List, Optional, Tuple, Any


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0
) -> Dict[str, float]:
    """
    Calcule les metriques d'evaluation pour un modele de classification binaire.

    Args:
        y_true: Labels reels
        y_pred_proba: Probabilites predites (classe positive)
        threshold: Seuil de classification (default: 0.5)
        cost_fp: Cout d'un faux positif (default: 1)
        cost_fn: Cout d'un faux negatif (default: 10)

    Returns:
        Dict avec toutes les metriques
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Metriques de base
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'auc_pr': average_precision_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }

    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['tp'] = int(tp)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)

    # Cout metier
    metrics['business_cost'] = fp * cost_fp + fn * cost_fn

    # Taux
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    return metrics


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset") -> None:
    """
    Affiche les metriques de maniere formatee.

    Args:
        metrics: Dictionnaire des metriques
        dataset_name: Nom du dataset (Train, Test, etc.)
    """
    print(f"\n{'='*50}")
    print(f"METRIQUES {dataset_name.upper()}")
    print(f"{'='*50}")
    print(f"  AUC-ROC:       {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:        {metrics['auc_pr']:.4f}")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  F1-Score:      {metrics['f1']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  Business Cost: {metrics['business_cost']:,.0f}")
    print(f"  Confusion:     TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
    color: str = 'blue'
) -> plt.Figure:
    """
    Trace la courbe ROC.

    Args:
        y_true: Labels reels
        y_pred_proba: Probabilites predites
        model_name: Nom du modele
        ax: Axes matplotlib (optionnel)
        color: Couleur de la courbe

    Returns:
        Figure matplotlib
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.plot(fpr, tpr, color=color, label=f'{model_name} (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Trace la matrice de confusion.

    Args:
        y_true: Labels reels
        y_pred_proba: Probabilites predites
        threshold: Seuil de classification
        model_name: Nom du modele
        ax: Axes matplotlib (optionnel)

    Returns:
        Figure matplotlib
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=['Pred 0', 'Pred 1'],
        yticklabels=['True 0', 'True 1'],
        ylabel='True label',
        xlabel='Predicted label'
    )
    ax.set_title(f'Confusion Matrix - {model_name}')

    # Ajouter les valeurs
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Figure]:
    """
    Trace l'importance des features.

    Args:
        model: Modele entraine (doit avoir feature_importances_ ou coef_)
        feature_names: Liste des noms de features
        top_n: Nombre de features a afficher
        model_name: Nom du modele
        ax: Axes matplotlib (optionnel)

    Returns:
        Figure matplotlib ou None si pas d'importance disponible
    """
    # Extraire importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None

    # Top N features
    idx = np.argsort(importance)[-top_n:]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    ax.barh(range(len(idx)), importance[idx], color='steelblue')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance - {model_name}')

    return fig


def compare_models(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metrics_to_compare: List[str] = None
) -> pd.DataFrame:
    """
    Compare les resultats de plusieurs modeles.

    Args:
        results: Dict structure comme:
            {
                'LogReg': {'baseline_train': {...}, 'baseline_test': {...}, 'tuned_test': {...}},
                'LightGBM': {'baseline_train': {...}, 'baseline_test': {...}, 'tuned_test': {...}}
            }
        metrics_to_compare: Liste des metriques a inclure

    Returns:
        DataFrame de comparaison
    """
    if metrics_to_compare is None:
        metrics_to_compare = ['auc_roc', 'f1', 'precision', 'recall', 'business_cost']

    rows = []
    for model_name, stages in results.items():
        for stage_name, metrics in stages.items():
            row = {'model': model_name, 'stage': stage_name}
            for m in metrics_to_compare:
                row[m] = metrics.get(m, None)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_models_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Trace un graphique comparatif des modeles.

    Args:
        results: Dict des resultats par modele et etape
        figsize: Taille de la figure

    Returns:
        Figure matplotlib
    """
    df = compare_models(results)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # AUC-ROC comparison
    models = df['model'].unique()
    stages = df['stage'].unique()
    x = np.arange(len(stages))
    width = 0.35

    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        values = [model_data[model_data['stage'] == s]['auc_roc'].values[0] for s in stages]
        axes[0].bar(x + i*width, values, width, label=model)

    axes[0].set_ylabel('AUC-ROC')
    axes[0].set_title('AUC-ROC par modele et etape')
    axes[0].set_xticks(x + width/2)
    axes[0].set_xticklabels(stages, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Business Cost comparison
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        values = [model_data[model_data['stage'] == s]['business_cost'].values[0] for s in stages]
        axes[1].bar(x + i*width, values, width, label=model)

    axes[1].set_ylabel('Business Cost')
    axes[1].set_title('Business Cost par modele et etape')
    axes[1].set_xticks(x + width/2)
    axes[1].set_xticklabels(stages, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    """
    Test des fonctions.
    """
    print("Test des fonctions de metrics")

    # Donnees fictives
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred_proba = np.random.rand(1000)

    # Test evaluate_model
    metrics = evaluate_model(y_true, y_pred_proba)
    print_metrics(metrics, "Test")

    # Test plot
    fig = plot_roc_curve(y_true, y_pred_proba, "Test Model")
    plt.close()

    print("\nTests termines avec succes!")
