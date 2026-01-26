"""
Exemples d'utilisation du Pipeline - Mode Single vs Multi
=========================================================

Ce fichier montre comment utiliser le pipeline avec:
1. Mode SINGLE: une seule méthode outliers
2. Mode MULTI: toutes les méthodes outliers en parallèle
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine du projet au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.pipeline import PreprocessingPipeline, preprocess_data

# ==================================================================
# PRÉPARATION DES DONNÉES (exemple fictif)
# ==================================================================

print("="*80)
print("PRÉPARATION DES DONNÉES")
print("="*80)

# Créer des données de test
np.random.seed(42)
n_samples = 1000

train_data = pd.DataFrame({
    'SK_ID_CURR': range(n_samples),
    'TARGET': np.random.binomial(1, 0.08, n_samples),
    'AMT_INCOME_TOTAL': np.random.normal(150000, 50000, n_samples),
    'AMT_CREDIT': np.random.normal(500000, 200000, n_samples),
    'AMT_ANNUITY': np.random.normal(25000, 10000, n_samples),
    'AMT_GOODS_PRICE': np.random.normal(450000, 180000, n_samples),
    'DAYS_BIRTH': np.random.randint(-25000, -7000, n_samples),
    'DAYS_EMPLOYED': np.random.randint(-10000, 0, n_samples),
    'EXT_SOURCE_1': np.random.uniform(0.2, 0.8, n_samples),
    'EXT_SOURCE_2': np.random.uniform(0.1, 0.9, n_samples),
    'EXT_SOURCE_3': np.random.uniform(0.3, 0.7, n_samples),
    'CODE_GENDER': np.random.choice(['M', 'F'], n_samples),
    'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], n_samples),
})

# Introduire valeurs manquantes et outliers
train_data.loc[np.random.choice(n_samples, 100), 'EXT_SOURCE_1'] = np.nan
train_data.loc[np.random.choice(n_samples, 50), 'AMT_INCOME_TOTAL'] = np.random.uniform(500000, 1000000, 50)

test_data = train_data.copy()
test_data['SK_ID_CURR'] = range(n_samples, n_samples * 2)
test_data = test_data.drop('TARGET', axis=1)

print(f"\nTrain shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")


# ==================================================================
# MODE 1: SINGLE - UNE SEULE MÉTHODE OUTLIERS
# ==================================================================

print("\n" + "="*80)
print("MODE 1: SINGLE - WINSORIZE UNIQUEMENT")
print("="*80)

# Méthode 1: Utiliser la fonction wrapper
train_single, test_single, pipeline_single = preprocess_data(
    train_data, 
    test_data,
    outlier_method='winsorize',  # Une seule méthode
    apply_all_outlier_methods=False,  # Mode single
    save_artifacts=True,
    artifacts_dir='artifacts/pipeline_single'
)

print(f"\nRésultat Mode Single:")
print(f"  - Train shape: {train_single.shape}")
print(f"  - Test shape: {test_single.shape}")
print(f"  - Nombre de colonnes AMT_INCOME_TOTAL: 1 (originale remplacée)")


# ==================================================================
# MODE 2: MULTI - TOUTES LES MÉTHODES OUTLIERS
# ==================================================================

print("\n" + "="*80)
print("MODE 2: MULTI - TOUTES LES MÉTHODES EN PARALLÈLE")
print("="*80)

# Méthode 2: Utiliser la fonction wrapper en mode multi
train_multi, test_multi, pipeline_multi = preprocess_data(
    train_data, 
    test_data,
    outlier_method='winsorize',  # Ignoré en mode multi
    apply_all_outlier_methods=True,  # Mode multi activé
    save_artifacts=True,
    artifacts_dir='artifacts/pipeline_multi'
)

print(f"\nRésultat Mode Multi:")
print(f"  - Train shape: {train_multi.shape}")
print(f"  - Test shape: {test_multi.shape}")
print(f"  - Nombre de colonnes pour AMT_INCOME_TOTAL:")
print(f"    • AMT_INCOME_TOTAL (originale)")
print(f"    • AMT_INCOME_TOTAL_WINS (winsorisée)")
print(f"    • AMT_INCOME_TOTAL_CAP (cappée)")
print(f"    • AMT_INCOME_TOTAL_LOG (log transformée)")


# ==================================================================
# COMPARAISON DES DEUX MODES
# ==================================================================

print("\n" + "="*80)
print("COMPARAISON DES MODES")
print("="*80)

print("\nMode SINGLE:")
print(f"  - Colonnes finales: {train_single.shape[1]}")
print(f"  - Avantage: Dataset plus compact")
print(f"  - Usage: Choisir la meilleure méthode après analyse")
print(f"  - Stockage: ~X Mo")

print("\nMode MULTI:")
print(f"  - Colonnes finales: {train_multi.shape[1]}")
print(f"  - Avantage: Tester toutes les méthodes simultanément")
print(f"  - Usage: Laisser le modèle choisir les meilleures features")
print(f"  - Stockage: ~3X Mo (environ 3x plus)")

print("\n" + "="*80)
print("RECOMMANDATION")
print("="*80)
print("""
Mode SINGLE (recommandé pour production):
  → Utiliser après avoir identifié la meilleure méthode
  → Plus économe en mémoire/stockage
  → Plus rapide à traîner

Mode MULTI (recommandé pour expérimentation):
  → Utiliser pour comparer les méthodes
  → Le modèle ML fera la sélection de features
  → Utile avec LightGBM/XGBoost (gèrent bien haute dimensionnalité)
""")


# ==================================================================
# UTILISATION AVANCÉE: CRÉER MANUELLEMENT LE PIPELINE
# ==================================================================

print("\n" + "="*80)
print("UTILISATION AVANCÉE")
print("="*80)

# Exemple 1: Pipeline single avec configuration personnalisée
print("\nExemple 1: Pipeline personnalisé mode single")
pipeline_custom_single = PreprocessingPipeline(
    use_outlier_handler=True,
    outlier_method='log',  # Transformation log
    apply_all_outlier_methods=False,
    use_feature_engineering=True,
    use_target_encoding=True,
    use_categorical_encoding=False  # Désactiver encodage catégoriel
)
train_custom_single = pipeline_custom_single.fit_transform(train_data, train_data['TARGET'])
print(f"  Shape: {train_custom_single.shape}")

# Exemple 2: Pipeline multi
print("\nExemple 2: Pipeline mode multi")
pipeline_custom_multi = PreprocessingPipeline(
    use_outlier_handler=True,
    outlier_method='winsorize',  # Ignoré
    apply_all_outlier_methods=True,  # Mode multi
    use_feature_engineering=True,
    use_target_encoding=True,
    use_categorical_encoding=True
)
train_custom_multi = pipeline_custom_multi.fit_transform(train_data, train_data['TARGET'])
print(f"  Shape: {train_custom_multi.shape}")

# Exemple 3: Pipeline sans traitement outliers
print("\nExemple 3: Pipeline sans traitement outliers")
pipeline_no_outliers = PreprocessingPipeline(
    use_outlier_handler=False,  # Désactivé
    use_feature_engineering=True,
    use_target_encoding=True,
    use_categorical_encoding=True
)
train_no_outliers = pipeline_no_outliers.fit_transform(train_data, train_data['TARGET'])
print(f"  Shape: {train_no_outliers.shape}")


# ==================================================================
# CHARGEMENT ET RÉUTILISATION
# ==================================================================

print("\n" + "="*80)
print("CHARGEMENT ET RÉUTILISATION")
print("="*80)

# Charger pipeline sauvegardé
pipeline_loaded = PreprocessingPipeline.load('artifacts/pipeline_single')
print("Pipeline chargé avec succès")

# Afficher résumé
pipeline_loaded.summary()

# Appliquer sur nouvelles données
new_data_processed = pipeline_loaded.transform(test_data)
print(f"\nNouvelles données transformées: {new_data_processed.shape}")


print("\n" + "="*80)
print("FIN DES EXEMPLES")
print("="*80)