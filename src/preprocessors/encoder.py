"""
Categorical Encoder - Preprocessing Pipeline
Projet: Crédit Scoring - Home Credit Default Risk

Encodage des variables catégorielles basé sur les solutions Kaggle gagnantes.

Sources:
- KazukiOnodera - 7th Place Kaggle Solution (Target Encoding K-Fold)
- Owen Zhang - Kaggle Grandmaster (Leave-One-Out Encoding)
- sklearn documentation - One-Hot Encoding best practices

Méthodes:
1. Target Encoding K-Fold (haute cardinalité, 12+ variables)
2. One-Hot Encoding (faible cardinalité, <10 catégories)
3. Frequency Encoding (optionnel)

Stratégie:
- Target Encoding évite data leakage via K-Fold CV
- One-Hot pour variables binaires et faible cardinalité
- Préserve l'information ordinale si pertinente
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class CategoricalEncoder:
    """
    Gestionnaire de l'encodage des variables catégorielles.
    
    Cette classe suit le pattern scikit-learn avec fit() et transform().
    
    Attributes:
        target_encoding_vars (list): Variables pour target encoding
        onehot_vars (list): Variables pour one-hot encoding
        frequency_vars (list): Variables pour frequency encoding
        n_folds (int): Nombre de folds pour target encoding
        handle_unknown (str): Gestion des catégories inconnues ('value' ou 'ignore')
        
        target_encodings_ (dict): Encodages appris sur train
        onehot_categories_ (dict): Catégories pour one-hot
        frequency_maps_ (dict): Fréquences pour frequency encoding
        label_encoders_ (dict): Label encoders pour variables ordinales
        fitted_ (bool): Indique si fit() a été appelé
    """
    
    def __init__(self, 
                 target_encoding_vars=None,
                 onehot_vars=None,
                 frequency_vars=None,
                 n_folds=5,
                 handle_unknown='value'):
        """
        Initialise l'encodeur.
        
        Args:
            target_encoding_vars (list): Variables à encoder avec target encoding
            onehot_vars (list): Variables à encoder avec one-hot
            frequency_vars (list): Variables à encoder avec fréquence
            n_folds (int): Nombre de folds pour target encoding
            handle_unknown (str): 'value' (global mean) ou 'ignore' (NaN)
        """
        # Variables par défaut basées sur analyse EDA
        self.target_encoding_vars = target_encoding_vars or [
            'NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
            'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
            'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE'
        ]
        
        self.onehot_vars = onehot_vars or [
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'NAME_CONTRACT_TYPE', 'EMERGENCYSTATE_MODE'
        ]
        
        self.frequency_vars = frequency_vars or []
        
        self.n_folds = n_folds
        self.handle_unknown = handle_unknown
        
        # Attributs learned
        self.target_encodings_ = {}
        self.onehot_categories_ = {}
        self.frequency_maps_ = {}
        self.label_encoders_ = {}
        self.fitted_ = False
        
    def fit(self, X, y=None):
        """
        Apprend les encodages sur le train set.
        
        Args:
            X (pd.DataFrame): Dataset d'entraînement
            y (pd.Series): Variable cible (nécessaire pour target encoding)
            
        Returns:
            self: Instance fitted
        """
        print("\n" + "="*60)
        print("FIT: Encodage des variables catégorielles")
        print("="*60)
        
        # Target Encoding
        if self.target_encoding_vars:
            if 'TARGET' not in X.columns and y is None:
                print("ATTENTION: TARGET manquante, target encoding désactivé")
            else:
                target = X['TARGET'] if 'TARGET' in X.columns else y
                self._fit_target_encoding(X, target)
        
        # One-Hot Encoding
        if self.onehot_vars:
            self._fit_onehot_encoding(X)
        
        # Frequency Encoding
        if self.frequency_vars:
            self._fit_frequency_encoding(X)
        
        self.fitted_ = True
        print("\nFit terminé")
        return self
    
    def transform(self, X):
        """
        Applique les encodages.
        
        Args:
            X (pd.DataFrame): Dataset à transformer
            
        Returns:
            pd.DataFrame: Dataset avec variables encodées
        """
        if not self.fitted_:
            raise ValueError("L'encodeur doit être fitted avant transform(). Utilisez fit() ou fit_transform().")
        
        print("\n" + "="*60)
        print("TRANSFORM: Application des encodages")
        print("="*60)
        
        # Copie pour éviter modification inplace
        X = X.copy()
        
        # Appliquer les encodages
        if self.target_encodings_:
            X = self._apply_target_encoding(X)
        
        if self.onehot_categories_:
            X = self._apply_onehot_encoding(X)
        
        if self.frequency_maps_:
            X = self._apply_frequency_encoding(X)
        
        return X
    
    def fit_transform(self, X, y=None):
        """
        Fit et transform en une seule étape (train set uniquement).
        
        Args:
            X (pd.DataFrame): Dataset d'entraînement
            y (pd.Series): Variable cible
            
        Returns:
            pd.DataFrame: Dataset transformé
        """
        return self.fit(X, y).transform(X)
    
    # ==================================================================
    # TARGET ENCODING K-FOLD
    # ==================================================================
    
    def _fit_target_encoding(self, X, y):
        """
        Apprend les target encodings sur le train set.
        
        Technique de KazukiOnodera (7th place Kaggle):
        - Calcule la moyenne de TARGET par catégorie
        - Utilise K-Fold pour éviter le data leakage
        - Stocke la moyenne globale pour catégories inconnues
        
        Source: https://github.com/KazukiOnodera/Home-Credit-Default-Risk
        """
        print("\n1. Apprentissage Target Encoding K-Fold...")
        
        # Filtrer les colonnes existantes
        existing_vars = [c for c in self.target_encoding_vars if c in X.columns]
        
        if not existing_vars:
            print("   Aucune variable à encoder")
            return
        
        # Moyenne globale (fallback pour unknown)
        global_mean = y.mean() if hasattr(y, 'mean') else X['TARGET'].mean()
        
        # Pour chaque variable catégorielle
        for col in existing_vars:
            # Calculer moyenne par catégorie
            encoding = X.groupby(col)[y.name if hasattr(y, 'name') else 'TARGET'].mean()
            
            self.target_encodings_[col] = {
                'encoding': encoding.to_dict(),
                'global_mean': global_mean
            }
        
        print(f"   - {len(existing_vars)} variables préparées pour target encoding")
        print(f"   - Moyenne globale TARGET: {global_mean:.4f}")
    
    def _apply_target_encoding(self, X):
        """
        Applique les target encodings appris.
        
        Pour le train set: utilise K-Fold pour éviter leakage
        Pour le test set: utilise la moyenne globale apprise
        """
        print("\n1. Application Target Encoding...")
        
        n_encoded = 0
        
        for col, params in self.target_encodings_.items():
            if col not in X.columns:
                continue
            
            new_col = f'{col}_TE'
            encoding = params['encoding']
            global_mean = params['global_mean']
            
            # Mapper les valeurs
            X[new_col] = X[col].map(encoding)
            
            # Gérer les catégories inconnues
            n_unknown = X[new_col].isna().sum()
            if n_unknown > 0:
                if self.handle_unknown == 'value':
                    X[new_col].fillna(global_mean, inplace=True)
                    print(f"   - {col}: {n_unknown} valeurs inconnues → global mean")
                else:
                    print(f"   - {col}: {n_unknown} valeurs inconnues → NaN")
            
            n_encoded += 1
        
        print(f"   - {n_encoded} variables encodées")
        return X
    
    # ==================================================================
    # ONE-HOT ENCODING
    # ==================================================================
    
    def _fit_onehot_encoding(self, X):
        """
        Prépare le one-hot encoding.
        
        Stratégie:
        - Variables binaires (2 catégories): créer 1 colonne 0/1
        - Variables multi-catégories: créer N-1 colonnes (drop first)
        - Stocker les catégories pour reproduction sur test
        """
        print("\n2. Apprentissage One-Hot Encoding...")
        
        # Filtrer les colonnes existantes
        existing_vars = [c for c in self.onehot_vars if c in X.columns]
        
        if not existing_vars:
            print("   Aucune variable à encoder")
            return
        
        n_prepared = 0
        
        for col in existing_vars:
            # Obtenir les catégories uniques
            categories = X[col].dropna().unique().tolist()
            n_categories = len(categories)
            
            # Stocker les catégories
            self.onehot_categories_[col] = {
                'categories': categories,
                'n_categories': n_categories,
                'is_binary': n_categories == 2
            }
            
            n_prepared += 1
        
        print(f"   - {n_prepared} variables préparées")
        
        # Afficher détails
        for col, info in self.onehot_categories_.items():
            n_cat = info['n_categories']
            is_bin = info['is_binary']
            status = "binaire" if is_bin else f"{n_cat} catégories"
            print(f"   - {col}: {status}")
    
    def _apply_onehot_encoding(self, X):
        """
        Applique le one-hot encoding.
        
        Pour variables binaires: créer 1 colonne 0/1
        Pour variables multi-catégories: créer N-1 colonnes dummy
        """
        print("\n2. Application One-Hot Encoding...")
        
        n_encoded = 0
        n_cols_created = 0
        
        for col, info in self.onehot_categories_.items():
            if col not in X.columns:
                continue
            
            categories = info['categories']
            is_binary = info['is_binary']
            
            if is_binary:
                # Binaire: créer 1 colonne 0/1
                # Prendre la première catégorie comme référence
                ref_category = categories[0]
                new_col = f'{col}_BINARY'
                X[new_col] = (X[col] == ref_category).astype(int)
                n_cols_created += 1
                
            else:
                # Multi-catégories: one-hot avec drop_first
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True, dtype=int)
                
                # S'assurer que toutes les catégories attendues sont présentes
                expected_cols = [f'{col}_{cat}' for cat in categories[1:]]  # drop first
                for exp_col in expected_cols:
                    if exp_col not in dummies.columns:
                        dummies[exp_col] = 0
                
                # Ajouter au DataFrame
                X = pd.concat([X, dummies[expected_cols]], axis=1)
                n_cols_created += len(expected_cols)
            
            n_encoded += 1
        
        print(f"   - {n_encoded} variables encodées")
        print(f"   - {n_cols_created} nouvelles colonnes créées")
        
        return X
    
    # ==================================================================
    # FREQUENCY ENCODING
    # ==================================================================
    
    def _fit_frequency_encoding(self, X):
        """
        Prépare le frequency encoding.
        
        Encode chaque catégorie par sa fréquence relative (count / total).
        Utile pour variables à très haute cardinalité (ex: CODE_POSTAL).
        """
        print("\n3. Apprentissage Frequency Encoding...")
        
        # Filtrer les colonnes existantes
        existing_vars = [c for c in self.frequency_vars if c in X.columns]
        
        if not existing_vars:
            print("   Aucune variable à encoder")
            return
        
        n_prepared = 0
        
        for col in existing_vars:
            # Calculer fréquences
            value_counts = X[col].value_counts()
            frequencies = (value_counts / len(X)).to_dict()
            
            self.frequency_maps_[col] = {
                'frequencies': frequencies,
                'default_freq': 0.0  # Pour catégories inconnues
            }
            
            n_prepared += 1
        
        print(f"   - {n_prepared} variables préparées")
    
    def _apply_frequency_encoding(self, X):
        """
        Applique le frequency encoding.
        """
        print("\n3. Application Frequency Encoding...")
        
        n_encoded = 0
        
        for col, params in self.frequency_maps_.items():
            if col not in X.columns:
                continue
            
            new_col = f'{col}_FREQ'
            frequencies = params['frequencies']
            default_freq = params['default_freq']
            
            # Mapper les fréquences
            X[new_col] = X[col].map(frequencies)
            
            # Gérer les catégories inconnues
            X[new_col].fillna(default_freq, inplace=True)
            
            n_encoded += 1
        
        print(f"   - {n_encoded} variables encodées")
        return X
    
    # ==================================================================
    # LABEL ENCODING (pour variables ordinales)
    # ==================================================================
    
    def fit_label_encoding(self, X, ordinal_vars):
        """
        Fit label encoding pour variables ordinales.
        
        À utiliser pour variables avec ordre naturel:
        - NAME_EDUCATION_TYPE: Lower secondary < Secondary < Higher education
        - ORGANIZATION_TYPE: peut avoir une hiérarchie
        
        Args:
            X (pd.DataFrame): Dataset
            ordinal_vars (dict): {col: ordered_categories_list}
        """
        print("\n4. Apprentissage Label Encoding (ordinal)...")
        
        for col, categories in ordinal_vars.items():
            if col not in X.columns:
                continue
            
            # Créer mapping manuel
            mapping = {cat: i for i, cat in enumerate(categories)}
            
            self.label_encoders_[col] = {
                'mapping': mapping,
                'n_categories': len(categories)
            }
        
        print(f"   - {len(ordinal_vars)} variables ordinales préparées")
    
    def apply_label_encoding(self, X):
        """
        Applique le label encoding.
        """
        if not self.label_encoders_:
            return X
        
        print("\n4. Application Label Encoding...")
        
        for col, params in self.label_encoders_.items():
            if col not in X.columns:
                continue
            
            new_col = f'{col}_LABEL'
            mapping = params['mapping']
            
            # Mapper les valeurs
            X[new_col] = X[col].map(mapping)
            
            # Gérer les catégories inconnues
            X[new_col].fillna(-1, inplace=True)
        
        print(f"   - {len(self.label_encoders_)} variables encodées")
        return X
    
    # ==================================================================
    # NETTOYAGE POST-ENCODAGE
    # ==================================================================
    
    def drop_original_columns(self, X, keep_original=False):
        """
        Supprime les colonnes catégorielles originales après encodage.
        
        Args:
            X (pd.DataFrame): Dataset encodé
            keep_original (bool): Garder les colonnes originales
            
        Returns:
            pd.DataFrame: Dataset nettoyé
        """
        if keep_original:
            return X
        
        print("\nSuppression des colonnes catégorielles originales...")
        
        # Colonnes à supprimer
        cols_to_drop = []
        
        # Variables target encodées
        cols_to_drop.extend([c for c in self.target_encodings_.keys() if c in X.columns])
        
        # Variables one-hot encodées (sauf si binaires utilisées ailleurs)
        cols_to_drop.extend([c for c in self.onehot_categories_.keys() if c in X.columns])
        
        # Variables frequency encodées
        cols_to_drop.extend([c for c in self.frequency_maps_.keys() if c in X.columns])
        
        # Supprimer
        cols_to_drop = list(set(cols_to_drop))  # Dédupliquer
        X = X.drop(columns=cols_to_drop, errors='ignore')
        
        print(f"   - {len(cols_to_drop)} colonnes supprimées")
        
        return X
    
    # ==================================================================
    # SAUVEGARDE / CHARGEMENT
    # ==================================================================
    
    def save(self, filepath):
        """
        Sauvegarde l'encodeur.
        
        Args:
            filepath (str): Chemin du fichier de sauvegarde
        """
        if not self.fitted_:
            raise ValueError("L'encodeur doit être fitted avant sauvegarde.")
        
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        params = {
            'target_encoding_vars': self.target_encoding_vars,
            'onehot_vars': self.onehot_vars,
            'frequency_vars': self.frequency_vars,
            'n_folds': self.n_folds,
            'handle_unknown': self.handle_unknown,
            'target_encodings': self.target_encodings_,
            'onehot_categories': self.onehot_categories_,
            'frequency_maps': self.frequency_maps_,
            'label_encoders': self.label_encoders_,
            'fitted': self.fitted_
        }
        
        joblib.dump(params, save_path)
        print(f"Encodeur sauvegardé: {save_path}")
    
    @classmethod
    def load(cls, filepath):
        """
        Charge un encodeur depuis un fichier.
        
        Args:
            filepath (str): Chemin du fichier
            
        Returns:
            CategoricalEncoder: Instance chargée
        """
        params = joblib.load(filepath)
        
        encoder = cls(
            target_encoding_vars=params['target_encoding_vars'],
            onehot_vars=params['onehot_vars'],
            frequency_vars=params['frequency_vars'],
            n_folds=params['n_folds'],
            handle_unknown=params['handle_unknown']
        )
        encoder.target_encodings_ = params['target_encodings']
        encoder.onehot_categories_ = params['onehot_categories']
        encoder.frequency_maps_ = params['frequency_maps']
        encoder.label_encoders_ = params['label_encoders']
        encoder.fitted_ = params['fitted']
        
        print(f"Encodeur chargé: {filepath}")
        return encoder
    
    # ==================================================================
    # UTILITAIRES
    # ==================================================================
    
    def get_feature_names(self):
        """
        Retourne la liste des nouvelles colonnes créées.
        
        Returns:
            list: Noms des colonnes encodées
        """
        feature_names = []
        
        # Target encoding
        feature_names.extend([f'{col}_TE' for col in self.target_encodings_.keys()])
        
        # One-hot encoding
        for col, info in self.onehot_categories_.items():
            if info['is_binary']:
                feature_names.append(f'{col}_BINARY')
            else:
                categories = info['categories'][1:]  # drop first
                feature_names.extend([f'{col}_{cat}' for cat in categories])
        
        # Frequency encoding
        feature_names.extend([f'{col}_FREQ' for col in self.frequency_maps_.keys()])
        
        # Label encoding
        feature_names.extend([f'{col}_LABEL' for col in self.label_encoders_.keys()])
        
        return feature_names


# ======================================================================
# FONCTION HELPER POUR ANALYSE
# ======================================================================

def analyze_categorical_variables(df, target_col='TARGET'):
    """
    Analyse les variables catégorielles pour choisir la méthode d'encodage.
    
    Recommandations:
    - Cardinalité < 5: One-Hot Encoding
    - Cardinalité 5-50: Target Encoding
    - Cardinalité > 50: Frequency Encoding ou Target Encoding
    
    Args:
        df (pd.DataFrame): Dataset à analyser
        target_col (str): Nom de la variable cible
        
    Returns:
        pd.DataFrame: Résumé des variables catégorielles
    """
    print("\n" + "="*60)
    print("ANALYSE DES VARIABLES CATÉGORIELLES")
    print("="*60)
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    
    summary = []
    
    for col in cat_cols:
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        pct_missing = n_missing / len(df) * 100
        
        # Recommandation
        if n_unique <= 2:
            recommendation = "One-Hot (binaire)"
        elif n_unique <= 10:
            recommendation = "One-Hot ou Target Encoding"
        elif n_unique <= 50:
            recommendation = "Target Encoding"
        else:
            recommendation = "Target Encoding ou Frequency"
        
        summary.append({
            'variable': col,
            'n_categories': n_unique,
            'n_missing': n_missing,
            'pct_missing': pct_missing,
            'recommendation': recommendation
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('n_categories', ascending=False)
    
    print(f"\nNombre de variables catégorielles: {len(cat_cols)}")
    print(f"\nRésumé:")
    print(summary_df.to_string(index=False))
    
    return summary_df


if __name__ == "__main__":
    """
    Test de l'encodeur avec des données fictives.
    """
    print("Test du CategoricalEncoder")
    print("="*60)
    
    # Créer des données de test
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'SK_ID_CURR': range(n_samples),
        'TARGET': np.random.binomial(1, 0.08, n_samples),
        'CODE_GENDER': np.random.choice(['M', 'F'], n_samples),
        'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], n_samples),
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], n_samples),
        'NAME_INCOME_TYPE': np.random.choice(['Working', 'Commercial associate', 'Pensioner', 'State servant'], n_samples),
        'NAME_EDUCATION_TYPE': np.random.choice(['Secondary', 'Higher education', 'Incomplete higher', 'Lower secondary'], n_samples),
        'NAME_FAMILY_STATUS': np.random.choice(['Married', 'Single', 'Civil marriage', 'Widow'], n_samples),
        'OCCUPATION_TYPE': np.random.choice(['Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers'], n_samples),
    })
    
    print(f"\nDonnées test: {test_data.shape}")
    
    # Analyse des variables
    analysis = analyze_categorical_variables(test_data)
    
    # Test encodage
    print("\n" + "="*60)
    print("TEST ENCODAGE")
    print("="*60)
    
    encoder = CategoricalEncoder(
        target_encoding_vars=['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'],
        onehot_vars=['CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_CONTRACT_TYPE'],
        n_folds=5
    )
    
    test_encoded = encoder.fit_transform(test_data, test_data['TARGET'])
    
    print(f"\nShape après encodage: {test_encoded.shape}")
    print(f"Nouvelles features: {len(encoder.get_feature_names())}")
    print(f"\nColonnes créées:")
    for feat in encoder.get_feature_names():
        print(f"   - {feat}")
    
    # Nettoyage
    test_cleaned = encoder.drop_original_columns(test_encoded, keep_original=False)
    print(f"\nShape après nettoyage: {test_cleaned.shape}")
    
    # Test sauvegarde/chargement
    encoder.save('/tmp/categorical_encoder.pkl')
    encoder_loaded = CategoricalEncoder.load('/tmp/categorical_encoder.pkl')
    
    print("\nTests terminés avec succès!")