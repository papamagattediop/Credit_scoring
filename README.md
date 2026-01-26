# Credit Scoring - Home Credit Default Risk

Projet de Machine Learning pour la prediction du risque de defaut de paiement.

---

## Objectif

Developper un modele de scoring credit capable de predire la probabilite qu'un client ne rembourse pas son pret. Ce projet s'appuie sur les donnees du challenge Kaggle "Home Credit Default Risk".

**Problematique metier:** Identifier les clients a risque tout en minimisant le refus de clients solvables.

**Metrique cible:** AUC-ROC > 0.80 avec optimisation du cout metier (ratio FN/FP).

---

## Plan du Projet

| Phase | Description | Statut |
|-------|-------------|--------|
| 1. Analyse exploratoire | Comprendre les donnees, distributions, correlations | Termine |
| 2. Preprocessing | Gestion valeurs manquantes, outliers, encoding | Termine |
| 3. Feature Selection | Selection des top 100 features | Termine |
| 4. Modelisation | Entrainement modeles individuels (par membre) | En cours |
| 5. Stacking | Ensemble des meilleurs modeles | A faire |
| 6. Interpretation | SHAP, feature importance | A faire |
| 7. Deploiement | API FastAPI + Dashboard Streamlit | A faire |

---

## Architecture du Projet

```
Credit_scoring/
│
├── data/
│   ├── raw/                    # Donnees Kaggle originales (non versionnees)
│   ├── merged/                 # Donnees fusionnees (non versionnees)
│   ├── processed/              # Donnees preprocessees
│   │   ├── train_selected.csv  # Train avec features selectionnees
│   │   └── test_selected.csv   # Test avec features selectionnees
│   └── outputs/                # Analyses et resultats
│
├── src/
│   ├── pipeline.py             # Pipeline de preprocessing principal
│   ├── preprocessors/          # Modules de preprocessing
│   │   ├── missing_handler.py  # Gestion valeurs manquantes
│   │   ├── outlier_handler.py  # Gestion outliers
│   │   ├── encoder.py          # Encodage categoriel
│   │   └── feature_engineer.py # Creation de features
│   ├── selection/              # Selection de features
│   │   └── feature_selector.py # Selecteur multi-methodes
│   ├── models/                 # Classes de modeles
│   ├── evaluation/             # Metriques et evaluation
│   └── utils/                  # Utilitaires (MLflow, config)
│
├── notebooks/
│   ├── 01_analyse.ipynb              # EDA
│   ├── 02_preprocessing.ipynb        # Preprocessing
│   ├── 03_0_modeling_template.ipynb  # Template modelisation
│   ├── 03_1_feature_selection.ipynb  # Selection features (COMMUN)
│   ├── 03_2_modeling_papa.ipynb      # Modeles Papa Magatte
│   ├── 03_3_modeling_moise.ipynb     # Modeles Moise
│   ├── 03_4_modeling_awa.ipynb       # Modeles Awa
│   ├── 03_5_modeling_khary.ipynb     # Modeles Khary
│   ├── 03_6_modeling_cos.ipynb       # Modeles COS
│   ├── 03_modeling_stacking_final.ipynb  # Stacking
│   └── 04_interpretation.ipynb       # SHAP
│
├── artifacts/
│   ├── selected_features/      # Selecteur et liste features
│   ├── pipeline_*/             # Pipelines sauvegardes
│   └── models/                 # Modeles entraines par membre
│       ├── papa/
│       ├── moise/
│       ├── awa/
│       ├── khary/
│       ├── cos/
│       └── stacking/
│
├── dashboard/                  # Application Streamlit
├── tests/                      # Tests unitaires
├── mlruns/                     # MLflow tracking (local)
└── requirements.txt            # Dependances Python
```

---

## Attribution des Modeles

| Membre | Modele 1 | Modele 2 | Notebook |
|--------|----------|----------|----------|
| Papa Magatte | Logistic Regression | LightGBM | `03_2_modeling_papa.ipynb` |
| Moise | Logistic (ElasticNet) | Decision Tree | `03_3_modeling_moise.ipynb` |
| Awa | XGBoost | Random Forest | `03_4_modeling_awa.ipynb` |
| Khary | ExtraTrees | HistGradientBoosting | `03_5_modeling_khary.ipynb` |
| COS | CatBoost | TabNet/MLP | `03_6_modeling_cos.ipynb` |

---

## Installation

```bash
# Cloner le repo
git clone https://github.com/VOTRE_USERNAME/Credit_scoring.git
cd Credit_scoring

# Creer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer dependances
pip install -r requirements.txt

# Telecharger donnees Kaggle (optionnel - si besoin de regenerer)
kaggle competitions download -c home-credit-default-risk -p data/raw/
```

---

## Utilisation

### Pour la modelisation (chaque membre)

```python
import pandas as pd

# Charger les donnees selectionnees
train = pd.read_csv('data/processed/train_selected.csv')
test = pd.read_csv('data/processed/test_selected.csv')

# Separer X et y
X_train = train.drop('TARGET', axis=1)
y_train = train['TARGET']

# Entrainer votre modele...
```

### Tracking MLflow

```python
import mlflow

mlflow.set_experiment("credit_scoring_VOTRE_NOM")

with mlflow.start_run(run_name="model_v1"):
    # Entrainement
    model.fit(X_train, y_train)

    # Log metriques
    mlflow.log_metric("auc", auc_score)
    mlflow.log_metric("f1", f1_score)

    # Log modele
    mlflow.sklearn.log_model(model, "model")
```

---

## Donnees

Les donnees proviennent du challenge Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).

**Fichiers inclus dans le repo:**
- `train_selected.csv` - 100 features selectionnees + TARGET
- `test_selected.csv` - 100 features selectionnees

**Fichiers a telecharger depuis Kaggle:**
- `application_train.csv` / `application_test.csv`
- `bureau.csv` / `bureau_balance.csv`
- `previous_application.csv`
- `POS_CASH_balance.csv`
- `credit_card_balance.csv`
- `installments_payments.csv`

---

## Technologies

- **Python 3.9+**
- **Preprocessing:** pandas, numpy, scikit-learn
- **Modeles:** LightGBM, XGBoost, CatBoost, scikit-learn
- **Tracking:** MLflow
- **Interpretation:** SHAP
- **API:** FastAPI
- **Dashboard:** Streamlit

---

## Auteurs

| Nom | Role | Contact |
|-----|------|---------|
| Papa Magatte Diop | Chef de projet / LogReg + LightGBM | - |
| Moise | Modelisation / LogReg + DecisionTree | - |
| Awa Gueye | Modelisation / XGBoost + RandomForest | - |
| Khary | Modelisation / ExtraTrees + GradientBoosting | - |
| COS | Modelisation / CatBoost + TabNet | - |

**Promotion:** AS3 - Machine Learning 2025/2026

---

## License

Ce projet est realise dans un cadre academique.
