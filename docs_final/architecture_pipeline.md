---
output:
  pdf_document: default
  html_document: default
---
# Architecture Pipeline Preprocessing - Projet Credit Scoring

## 🎯 Principe : Pipeline en Mémoire (Sans Sauvegarde Intermédiaire)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DONNÉES BRUTES (Kaggle)                         │
│                          application_train.csv                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING PIPELINE (src/)                       │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  1. MissingValuesHandler (preprocessors/missing_handler.py)    │   │
│  │     • Indicateurs binaires                                     │   │
│  │     • Imputation par catégorie                                 │   │
│  └────────────────────────┬───────────────────────────────────────┘   │
│                           │                                             │
│                           ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  2. OutlierHandler (preprocessors/outlier_handler.py)          │   │
│  │     • Détection IQR/percentiles                                │   │
│  │     • Winsorization/Log transform                              │   │
│  └────────────────────────┬───────────────────────────────────────┘   │
│                           │                                             │
│                           ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  3. FeatureEngineer (preprocessors/feature_engineer.py)        │   │
│  │     • Ratios financiers                                        │   │
│  │     • Variables temporelles                                    │   │
│  │     • Agrégations scores                                       │   │
│  └────────────────────────┬───────────────────────────────────────┘   │
│                           │                                             │
│                           ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  4. CategoricalEncoder (preprocessors/encoder.py)              │   │
│  │     • Target Encoding                                          │   │
│  │     • One-Hot Encoding                                         │   │
│  └────────────────────────┬───────────────────────────────────────┘   │
│                           │                                             │
│  ┌────────────────────────▼───────────────────────────────────────┐   │
│  │  5. PreprocessingPipeline (pipeline.py)                        │   │
│  │     • Orchestration de toutes les étapes                       │   │
│  │     • fit_transform() : exécution séquentielle                 │   │
│  └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ✅ DONNÉES PREPROCESSÉES (EN MÉMOIRE)                │
│                           df_processed                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      EXPÉRIMENTATION MODÈLES                            │
│                                                                         │
│  ┌──────────────────────┐   ┌──────────────────────┐                  │
│  │  Modèles à tester    │   │  Techniques testées  │                  │
│  │  • Logistic Reg      │   │  • SMOTE             │                  │
│  │  • Random Forest     │   │  • class_weight      │                  │
│  │  • XGBoost           │   │  • Threshold tuning  │                  │
│  │  • LightGBM          │   │  • Undersampling     │                  │
│  └──────────────────────┘   └──────────────────────┘                  │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  Boucle d'expérimentation (MLflow tracking)                    │   │
│  │                                                                 │   │
│  │  FOR model IN [LogReg, RF, XGB, LGBM]:                         │   │
│  │      FOR technique IN [SMOTE, weight, threshold]:              │   │
│  │          • Split train/validation (80/20)                      │   │
│  │          • Entraînement modèle                                 │   │
│  │          • Évaluation (AUC-ROC, F-beta, coût métier)          │   │
│  │          • Log MLflow (params + metrics)                       │   │
│  └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    💾 SAUVEGARDES STRATÉGIQUES                          │
│                                                                         │
│  1. Données finales :     data/processed/df_processed.csv              │
│  2. Artefacts ML :        artifacts/encoders/, artifacts/scalers/      │
│  3. Meilleur modèle :     models/best_model.pkl                        │
│  4. Tracking MLflow :     mlruns/                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📊 Avantages de cette Architecture

### ✅ Efficacité
- **Pas de sauvegarde intermédiaire** → gain de stockage
- **Pipeline en mémoire** → exécution rapide
- **Transformations enchaînées** → workflow fluide

### ✅ MLOps Ready
- **Modulaire** : chaque classe = une responsabilité
- **Testable** : tests unitaires par composant
- **Réutilisable** : même pipeline train/production
- **Versionnable** : artefacts sauvegardés pour production

### ✅ Expérimentation
- **Flexibilité** : tester multiples combinaisons modèles/techniques
- **Traçabilité** : MLflow log toutes les expériences
- **Comparabilité** : métriques standardisées

## 🔧 Structure Fichiers

```
src/
├── preprocessors/
│   ├── __init__.py
│   ├── missing_handler.py      ← Classe gestion valeurs manquantes
│   ├── outlier_handler.py      ← Classe gestion outliers
│   ├── feature_engineer.py     ← Classe création features
│   └── encoder.py              ← Classe encodage catégorielles
├── pipeline.py                  ← Orchestration complète
├── utils.py                     ← Fonctions utilitaires
└── models.py                    ← Classes modèles ML

notebooks/
├── 01_analyse.ipynb            ← EDA
├── 02_preprocessing.ipynb      ← Test pipeline
├── 03_modeling.ipynb           ← Expérimentation modèles
└── 04_interpretation.ipynb     ← SHAP/LIME

data/
├── raw/                        ← Données Kaggle brutes
└── processed/                  ← df_processed final uniquement

artifacts/
├── encoders/                   ← Target/OneHot encoders
└── scalers/                    ← StandardScaler si nécessaire

models/
└── best_model.pkl              ← Meilleur modèle final

mlruns/                         ← Tracking MLflow
```

## 🚀 Workflow d'Utilisation

### Phase Développement (Notebooks)
```python
# notebook 02_preprocessing.ipynb
from src.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()
df_processed = pipeline.fit_transform(df_raw)

# Sauvegarde uniquement à la fin
df_processed.to_csv('data/processed/df_processed.csv')
```

### Phase Expérimentation (Notebooks + MLflow)
```python
# notebook 03_modeling.ipynb
import mlflow

for model in [LogisticRegression(), RandomForest(), XGBoost(), LightGBM()]:
    for technique in ['SMOTE', 'class_weight', 'threshold']:
        with mlflow.start_run():
            # Entraînement
            results = train_evaluate(df_processed, model, technique)
            
            # Logging
            mlflow.log_params({"model": model.__class__.__name__, 
                              "technique": technique})
            mlflow.log_metrics(results)
```

### Phase Production (API)
```python
# api/main.py
from src.pipeline import PreprocessingPipeline
import joblib

pipeline = PreprocessingPipeline.load('artifacts/')
model = joblib.load('models/best_model.pkl')

@app.post("/predict")
def predict(data):
    processed = pipeline.transform(data)
    prediction = model.predict_proba(processed)
    return {"probability": prediction}
```

## 📝 Notes Importantes

1. **Pas de sauvegarde entre les étapes** : gain stockage + simplicité
2. **Artefacts sauvegardés** : encoders, scalers pour production
3. **MLflow tracking** : toutes expériences tracées et comparables
4. **Reproductibilité** : mêmes transformations train/production
5. **Scalabilité** : ajout facile de nouvelles étapes

---
**Date** : Janvier 2026  
**Projet** : CLF02 - Crédit Scoring "Prêt à dépenser"
