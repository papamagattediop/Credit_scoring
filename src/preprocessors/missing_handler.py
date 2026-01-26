"""
Missing Values Handler - Preprocessing Pipeline
Projet: Crédit Scoring - Home Credit Default Risk

Stratégie basée sur les solutions Kaggle gagnantes:
- NoxMoon (17ème, AUC 0.80265)
- deepsense.ai (multiple gold medals)
- TheCyPhy team (Top 3%)
- pklauke (Top 4%, 248ème)

Approche validée:
1. Création d'indicateurs binaires de missingness
2. Imputation simple par catégorie de variable
3. Exploitation du traitement natif LightGBM/XGBoost
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class MissingValuesHandler:
    """
    Gestionnaire des valeurs manquantes pour le pipeline de preprocessing.
    
    Cette classe suit le pattern scikit-learn avec fit() et transform().
    
    Attributes:
        medians_ (dict): Médianes calculées sur le train set
        new_features (list): Liste des nouvelles features créées
        fitted_ (bool): Indique si fit() a été appelé
    """
    
    def __init__(self):
        """Initialise le handler."""
        self.medians_ = {}
        self.new_features = []
        self.fitted_ = False
        
    def fit(self, X, y=None):
        """
        Apprend les paramètres nécessaires (médianes) sur le train set.
        
        Args:
            X (pd.DataFrame): Dataset d'entraînement
            y: Non utilisé (compatibilité scikit-learn)
            
        Returns:
            self: Instance fitted
        """
        print("\n" + "="*60)
        print("FIT: Apprentissage des paramètres (médianes)")
        print("="*60)
        
        # Calculer les médianes sur le train set AVANT imputation
        self.medians_['EXT_SOURCE_1'] = X['EXT_SOURCE_1'].median()
        self.medians_['EXT_SOURCE_2'] = X['EXT_SOURCE_2'].median()
        self.medians_['EXT_SOURCE_3'] = X['EXT_SOURCE_3'].median()
        
        # DAYS_EMPLOYED: calculer médiane après correction anomalie
        days_employed_clean = X['DAYS_EMPLOYED'].replace(365243, np.nan)
        self.medians_['DAYS_EMPLOYED'] = days_employed_clean.median()
        
        # Variables immobilières
        real_estate_keywords = ['AREA', 'FLOORS', 'ELEVATORS', 'ENTRANCES', 'APARTMENTS', 
                                'YEARS_BUILD', 'COMMONAREA', 'LIVINGAREA', 'NONLIVINGAREA',
                                'BASEMENTAREA', 'LANDAREA', 'TOTALAREA']
        
        real_estate_cols = [col for col in X.columns 
                           if any(keyword in col for keyword in real_estate_keywords)]
        
        for col in real_estate_cols:
            if col in X.columns:
                self.medians_[col] = X[col].median()
        
        # Autres variables numériques (calculées AVANT imputation)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # Exclure les colonnes déjà traitées
        excluded_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_EMPLOYED'] + real_estate_cols
        
        for col in numeric_cols:
            if col not in excluded_cols and col not in self.medians_:
                if X[col].isnull().any():
                    self.medians_[col] = X[col].median()
        
        print(f"\nMédianes calculées pour {len(self.medians_)} variables")
        print(f"Exemples:")
        for i, (col, val) in enumerate(list(self.medians_.items())[:5]):
            print(f"   - {col}: {val:.6f}")
        
        self.fitted_ = True
        return self
        
    def transform(self, X):
        """
        Applique les transformations (indicateurs + imputation).
        
        Args:
            X (pd.DataFrame): Dataset à transformer
            
        Returns:
            pd.DataFrame: Dataset transformé
        """
        if not self.fitted_:
            raise ValueError("Le handler doit être fitted avant transform(). Utilisez fit() ou fit_transform().")
        
        print("\n" + "="*60)
        print("TRANSFORM: Application des transformations")
        print("="*60)
        
        # Copie pour éviter modification inplace
        X = X.copy()
        
        # Etape 1: Créer indicateurs binaires
        X = self._create_binary_indicators(X)
        
        # Etape 2: Imputer les valeurs manquantes
        X = self._impute_missing_values(X)
        
        # Validation
        n_missing = X.isnull().sum().sum()
        print(f"\nValeurs manquantes restantes: {n_missing}")
        
        if n_missing > 0:
            print("ATTENTION: Il reste des valeurs manquantes!")
            remaining = X.isnull().sum()
            remaining = remaining[remaining > 0].sort_values(ascending=False)
            print(f"Colonnes concernées: {len(remaining)}")
            print(remaining.head(10))
        else:
            print("Succès: Aucune valeur manquante restante")
        
        return X
        
    def fit_transform(self, X, y=None):
        """
        Fit et transform en une seule étape (train set uniquement).
        
        Args:
            X (pd.DataFrame): Dataset d'entraînement
            y: Non utilisé
            
        Returns:
            pd.DataFrame: Dataset transformé
        """
        return self.fit(X, y).transform(X)
        
    def _create_binary_indicators(self, X):
        """
        Crée les indicateurs binaires de missingness.
        
        Ces features capturent le pattern de valeurs manquantes qui est 
        informatif pour prédire le défaut de paiement.
        
        Gain attendu: +0.001 à +0.002 AUC
        """
        print("\n1. Création des indicateurs binaires...")
        
        # 1.1 Indicateurs pour tables auxiliaires
        bureau_cols = [col for col in X.columns if col.startswith('BUREAU_')]
        cc_cols = [col for col in X.columns if col.startswith('CC_')]
        prev_cols = [col for col in X.columns if col.startswith('PREV_')]
        pos_cols = [col for col in X.columns if col.startswith('POS_')]
        inst_cols = [col for col in X.columns if col.startswith('INST_')]
        
        X['HAS_BUREAU'] = X[bureau_cols].notna().any(axis=1).astype(int) if bureau_cols else 0
        X['HAS_CC'] = X[cc_cols].notna().any(axis=1).astype(int) if cc_cols else 0
        X['HAS_PREV'] = X[prev_cols].notna().any(axis=1).astype(int) if prev_cols else 0
        X['HAS_POS'] = X[pos_cols].notna().any(axis=1).astype(int) if pos_cols else 0
        X['HAS_INST'] = X[inst_cols].notna().any(axis=1).astype(int) if inst_cols else 0
        
        print(f"   - Tables auxiliaires: 5 indicateurs créés")
        
        # 1.2 Indicateurs pour EXT_SOURCE (features critiques)
        X['HAS_EXT_SOURCE_1'] = X['EXT_SOURCE_1'].notna().astype(int)
        X['HAS_EXT_SOURCE_2'] = X['EXT_SOURCE_2'].notna().astype(int)
        X['HAS_EXT_SOURCE_3'] = X['EXT_SOURCE_3'].notna().astype(int)
        
        print(f"   - EXT_SOURCE: 3 indicateurs créés")
        
        # 1.3 Anomalie DAYS_EMPLOYED
        X['DAYS_EMPLOYED_ANOM'] = (X['DAYS_EMPLOYED'] == 365243).astype(int)
        X['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        
        n_anom = X['DAYS_EMPLOYED_ANOM'].sum()
        print(f"   - DAYS_EMPLOYED_ANOM: {n_anom:,} anomalies détectées ({n_anom/len(X)*100:.2f}%)")
        
        # Mise à jour de la liste des nouvelles features (une seule fois)
        if not self.new_features:
            self.new_features = [
                'HAS_BUREAU', 'HAS_CC', 'HAS_PREV', 'HAS_POS', 'HAS_INST',
                'HAS_EXT_SOURCE_1', 'HAS_EXT_SOURCE_2', 'HAS_EXT_SOURCE_3',
                'DAYS_EMPLOYED_ANOM'
            ]
        
        print(f"\nTotal: {len(self.new_features)} nouvelles features créées")
        
        return X
        
    def _impute_missing_values(self, X):
        """
        Impute les valeurs manquantes par catégorie.
        
        Stratégies validées par les solutions gagnantes Kaggle.
        """
        print("\n2. Imputation par catégorie...")
        
        # 2.1 Tables auxiliaires -> 0
        bureau_cols = [col for col in X.columns if col.startswith('BUREAU_')]
        cc_cols = [col for col in X.columns if col.startswith('CC_')]
        prev_cols = [col for col in X.columns if col.startswith('PREV_')]
        pos_cols = [col for col in X.columns if col.startswith('POS_')]
        inst_cols = [col for col in X.columns if col.startswith('INST_')]
        
        auxiliary_cols = bureau_cols + cc_cols + prev_cols + pos_cols + inst_cols
        
        if auxiliary_cols:
            X[auxiliary_cols] = X[auxiliary_cols].fillna(0)
            print(f"   - Tables auxiliaires: {len(auxiliary_cols)} colonnes → 0")
        
        # 2.2 EXT_SOURCE -> Médiane (learned from train)
        for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
            if col in X.columns and col in self.medians_:
                X[col] = X[col].fillna(self.medians_[col])
        print(f"   - EXT_SOURCE: 3 colonnes → médiane")
        
        # 2.3 Variables conditionnelles -> -1
        conditional_cols = ['OWN_CAR_AGE', 'CNT_FAM_MEMBERS']
        for col in conditional_cols:
            if col in X.columns:
                X[col] = X[col].fillna(-1)
        print(f"   - Variables conditionnelles: {len([c for c in conditional_cols if c in X.columns])} colonnes → -1")
        
        # 2.4 Variables catégorielles -> "Unknown"
        cat_cols = ['OCCUPATION_TYPE', 'NAME_TYPE_SUITE', 'FONDKAPREMONT_MODE', 
                    'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
        
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].fillna('Unknown')
        print(f"   - Variables catégorielles: {len([c for c in cat_cols if c in X.columns])} colonnes → 'Unknown'")
        
        # 2.5 Requêtes Credit Bureau -> 0
        bureau_req_cols = [col for col in X.columns if 'AMT_REQ_CREDIT_BUREAU' in col]
        if bureau_req_cols:
            X[bureau_req_cols] = X[bureau_req_cols].fillna(0)
            print(f"   - Requêtes Bureau: {len(bureau_req_cols)} colonnes → 0")
        
        # 2.6 Variables immobilières -> Médiane
        real_estate_keywords = ['AREA', 'FLOORS', 'ELEVATORS', 'ENTRANCES', 'APARTMENTS', 
                                'YEARS_BUILD', 'COMMONAREA', 'LIVINGAREA', 'NONLIVINGAREA',
                                'BASEMENTAREA', 'LANDAREA', 'TOTALAREA']
        
        real_estate_cols = [col for col in X.columns 
                           if any(keyword in col for keyword in real_estate_keywords)]
        
        for col in real_estate_cols:
            if col in self.medians_:
                X[col] = X[col].fillna(self.medians_[col])
        
        if real_estate_cols:
            print(f"   - Variables immobilières: {len(real_estate_cols)} colonnes → médiane")
        
        # 2.7 DAYS_EMPLOYED -> Médiane
        if 'DAYS_EMPLOYED' in self.medians_:
            X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].fillna(self.medians_['DAYS_EMPLOYED'])
            print(f"   - DAYS_EMPLOYED: → médiane ({self.medians_['DAYS_EMPLOYED']:.0f} jours)")
        
        # 2.8 Autres variables numériques -> Médiane
        for col, median_val in self.medians_.items():
            if col in X.columns and X[col].isnull().any():
                X[col] = X[col].fillna(median_val)
        
        print(f"   - Autres numériques: imputées par médiane")
        
        return X
        
    def save(self, filepath):
        """
        Sauvegarde le handler fitted (médianes).
        
        Args:
            filepath (str): Chemin du fichier de sauvegarde
        """
        if not self.fitted_:
            raise ValueError("Le handler doit être fitted avant sauvegarde.")
        
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder uniquement les paramètres nécessaires
        params = {
            'medians': self.medians_,
            'new_features': self.new_features,
            'fitted': self.fitted_
        }
        
        joblib.dump(params, save_path)
        print(f"Handler sauvegardé: {save_path}")
        
    @classmethod
    def load(cls, filepath):
        """
        Charge un handler depuis un fichier.
        
        Args:
            filepath (str): Chemin du fichier
            
        Returns:
            MissingValuesHandler: Instance chargée
        """
        params = joblib.load(filepath)
        
        handler = cls()
        handler.medians_ = params['medians']
        handler.new_features = params['new_features']
        handler.fitted_ = params['fitted']
        
        print(f"Handler chargé: {filepath}")
        return handler


# Fonction utilitaire pour compatibilité avec ancien code
def process_missing_values(train_df, test_df=None):
    """
    Fonction wrapper pour traiter les valeurs manquantes.
    
    Args:
        train_df (pd.DataFrame): Dataset d'entraînement
        test_df (pd.DataFrame, optional): Dataset de test
        
    Returns:
        tuple: (train_processed, test_processed) ou train_processed si test_df=None
        
    Exemple:
        >>> handler = MissingValuesHandler()
        >>> train_processed = handler.fit_transform(train_df)
        >>> test_processed = handler.transform(test_df)
    """
    handler = MissingValuesHandler()
    
    # Fit sur train
    train_processed = handler.fit_transform(train_df)
    
    # Transform sur test si fourni
    if test_df is not None:
        test_processed = handler.transform(test_df)
        return train_processed, test_processed, handler
    
    return train_processed, handler


if __name__ == "__main__":
    """
    Test du handler avec des données fictives.
    """
    print("Test du MissingValuesHandler")
    print("="*60)
    
    # Créer des données de test
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'EXT_SOURCE_1': np.random.randn(n_samples),
        'EXT_SOURCE_2': np.random.randn(n_samples),
        'EXT_SOURCE_3': np.random.randn(n_samples),
        'DAYS_EMPLOYED': np.random.randint(-10000, 0, n_samples),
        'BUREAU_DAYS_CREDIT_mean': np.random.randn(n_samples),
        'CC_AMT_BALANCE_mean': np.random.randn(n_samples),
        'OWN_CAR_AGE': np.random.randint(0, 20, n_samples),
    })
    
    # Introduire des valeurs manquantes
    test_data.loc[np.random.choice(n_samples, 100), 'EXT_SOURCE_1'] = np.nan
    test_data.loc[np.random.choice(n_samples, 50), 'BUREAU_DAYS_CREDIT_mean'] = np.nan
    test_data.loc[np.random.choice(n_samples, 10), 'DAYS_EMPLOYED'] = 365243  # Anomalie
    
    print(f"\nDonnées test: {test_data.shape}")
    print(f"Valeurs manquantes: {test_data.isnull().sum().sum()}")
    
    # Test du handler
    handler = MissingValuesHandler()
    test_processed = handler.fit_transform(test_data)
    
    print(f"\nDonnées traitées: {test_processed.shape}")
    print(f"Nouvelles features: {handler.new_features}")
    
    # Test sauvegarde/chargement
    handler.save('/tmp/missing_handler.pkl')
    handler_loaded = MissingValuesHandler.load('/tmp/missing_handler.pkl')
    
    print("\nTests terminés avec succès!")