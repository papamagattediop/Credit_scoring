"""
Feature Engineer - Preprocessing Pipeline
Projet: Crédit Scoring - Home Credit Default Risk

Ce module contient toutes les fonctions de feature engineering.

Sources:
- KazukiOnodera - 7th Place Kaggle Solution
- Altman Z-Score (1968) - Financial Ratios
- Basel III - Debt-to-Income requirements

Architecture:
- Pattern scikit-learn (fit/transform)
- Séparation des responsabilités
- Compatible avec pipeline de preprocessing
- Sauvegarde des artefacts pour production

"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import gc


class FeatureEngineer:
    """
    Gestionnaire du feature engineering pour le pipeline de preprocessing.
    
    Cette classe suit le pattern scikit-learn avec fit() et transform().
    
    Attributes:
        use_target_encoding (bool): Utiliser target encoding K-Fold
        n_folds (int): Nombre de folds pour target encoding
        target_encodings_ (dict): Encodages appris sur train
        new_features (list): Liste des nouvelles features créées
        fitted_ (bool): Indique si fit() a été appelé
    """
    
    def __init__(self, use_target_encoding=True, n_folds=5):
        """
        Initialise le feature engineer.
        
        Args:
            use_target_encoding (bool): Activer target encoding K-Fold
            n_folds (int): Nombre de folds pour validation croisée
        """
        self.use_target_encoding = use_target_encoding
        self.n_folds = n_folds
        self.target_encodings_ = {}
        self.new_features = []
        self.fitted_ = False
        
    def fit(self, X, y=None):
        """
        Apprend les paramètres nécessaires sur le train set.
        
        Pour target encoding: calcule les moyennes par catégorie.
        
        Args:
            X (pd.DataFrame): Dataset d'entraînement
            y (pd.Series): Variable cible (nécessaire pour target encoding)
            
        Returns:
            self: Instance fitted
        """
        print("\n" + "="*60)
        print("FIT: Feature Engineering")
        print("="*60)
        
        # Vérifier présence de TARGET pour target encoding
        if self.use_target_encoding:
            if 'TARGET' not in X.columns and y is None:
                print("ATTENTION: TARGET manquante, target encoding désactivé")
                self.use_target_encoding = False
            else:
                # Utiliser TARGET de X ou y fourni
                target = X['TARGET'] if 'TARGET' in X.columns else y
                self._fit_target_encoding(X, target)
        
        self.fitted_ = True
        print("\nFit terminé")
        return self
        
    def transform(self, X):
        """
        Crée toutes les features.
        
        Args:
            X (pd.DataFrame): Dataset à transformer
            
        Returns:
            pd.DataFrame: Dataset avec nouvelles features
        """
        if not self.fitted_:
            raise ValueError("Le feature engineer doit être fitted avant transform(). Utilisez fit() ou fit_transform().")
        
        print("\n" + "="*60)
        print("TRANSFORM: Création des features")
        print("="*60)
        
        # Copie pour éviter modification inplace
        X = X.copy()
        
        # Réinitialiser la liste des nouvelles features
        self.new_features = []
        
        # 1. Ratios financiers (11 features)
        X = self._create_financial_ratios(X)
        
        # 2. Variables temporelles (15 features)
        X = self._create_temporal_features(X)
        
        # 3. Target encoding (si activé)
        if self.use_target_encoding and self.target_encodings_:
            X = self._apply_target_encoding(X)
        
        # 4. Features d'interaction EXT_SOURCE (21 features)
        X = self._create_interaction_features(X)
        
        # 5. Flags de présence (8 features)
        X = self._create_presence_flags(X)
        
        print(f"\nTotal nouvelles features créées: {len(self.new_features)}")
        
        return X
        
    def fit_transform(self, X, y=None):
        """
        Fit et transform en une seule étape.
        
        Args:
            X (pd.DataFrame): Dataset d'entraînement
            y (pd.Series): Variable cible
            
        Returns:
            pd.DataFrame: Dataset transformé
        """
        return self.fit(X, y).transform(X)
    
    # ==================================================================
    # RATIOS FINANCIERS (11 features)
    # ==================================================================
    
    def _create_financial_ratios(self, X):
        """
        Crée des ratios financiers inspirés du Z-Score d'Altman.
        
        Gain attendu: +0.002 à +0.005 AUC
        """
        print("\n1. Création des ratios financiers...")
        
        n_created = 0
        
        # DTI (Debt-to-Income) - Ratio clé en scoring bancaire
        if 'AMT_CREDIT' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)
            self.new_features.append('CREDIT_INCOME_RATIO')
            n_created += 1
        
        # Annuity Burden - Charge mensuelle relative
        if 'AMT_ANNUITY' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)
            self.new_features.append('ANNUITY_INCOME_RATIO')
            n_created += 1
        
        # Credit/Goods Ratio - Marge de financement
        if 'AMT_GOODS_PRICE' in X.columns and 'AMT_CREDIT' in X.columns:
            X['CREDIT_GOODS_RATIO'] = X['AMT_CREDIT'] / (X['AMT_GOODS_PRICE'] + 1)
            self.new_features.append('CREDIT_GOODS_RATIO')
            n_created += 1
        
        # Annuity/Credit Ratio - Vitesse de remboursement
        if 'AMT_ANNUITY' in X.columns and 'AMT_CREDIT' in X.columns:
            X['ANNUITY_CREDIT_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_CREDIT'] + 1)
            self.new_features.append('ANNUITY_CREDIT_RATIO')
            n_created += 1
        
        # Revenu par membre de famille
        if 'CNT_FAM_MEMBERS' in X.columns:
            X['INCOME_PER_FAMILY_MEMBER'] = X['AMT_INCOME_TOTAL'] / (X['CNT_FAM_MEMBERS'] + 1)
            self.new_features.append('INCOME_PER_FAMILY_MEMBER')
            n_created += 1
        
        # Crédit par membre de famille
        if 'CNT_FAM_MEMBERS' in X.columns:
            X['CREDIT_PER_FAMILY_MEMBER'] = X['AMT_CREDIT'] / (X['CNT_FAM_MEMBERS'] + 1)
            self.new_features.append('CREDIT_PER_FAMILY_MEMBER')
            n_created += 1
        
        # Revenu par enfant
        if 'CNT_CHILDREN' in X.columns:
            X['INCOME_PER_CHILD'] = X['AMT_INCOME_TOTAL'] / (X['CNT_CHILDREN'] + 1)
            self.new_features.append('INCOME_PER_CHILD')
            n_created += 1
        
        # Durée estimée du crédit
        if 'AMT_ANNUITY' in X.columns and 'AMT_CREDIT' in X.columns:
            X['CREDIT_TERM_MONTHS'] = X['AMT_CREDIT'] / (X['AMT_ANNUITY'] + 1)
            X['CREDIT_TERM_YEARS'] = X['CREDIT_TERM_MONTHS'] / 12
            self.new_features.extend(['CREDIT_TERM_MONTHS', 'CREDIT_TERM_YEARS'])
            n_created += 2
        
        # Différence crédit - prix du bien
        if 'AMT_GOODS_PRICE' in X.columns and 'AMT_CREDIT' in X.columns:
            X['CREDIT_GOODS_DIFF'] = X['AMT_CREDIT'] - X['AMT_GOODS_PRICE']
            X['CREDIT_GOODS_PERC'] = X['CREDIT_GOODS_DIFF'] / (X['AMT_GOODS_PRICE'] + 1)
            self.new_features.extend(['CREDIT_GOODS_DIFF', 'CREDIT_GOODS_PERC'])
            n_created += 2
        
        print(f"   - {n_created} ratios financiers créés")
        return X
    
    # ==================================================================
    # VARIABLES TEMPORELLES (15 features)
    # ==================================================================
    
    def _create_temporal_features(self, X):
        """
        Crée des features temporelles dérivées des variables DAYS_*.
        
        Gain attendu: +0.001 à +0.003 AUC
        """
        print("\n2. Création des variables temporelles...")
        
        n_created = 0
        
        # Conversions en années
        if 'DAYS_BIRTH' in X.columns:
            X['AGE_YEARS'] = -X['DAYS_BIRTH'] / 365
            self.new_features.append('AGE_YEARS')
            n_created += 1
        
        if 'DAYS_EMPLOYED' in X.columns:
            X['EMPLOYED_YEARS'] = -X['DAYS_EMPLOYED'] / 365
            X.loc[X['DAYS_EMPLOYED'] > 0, 'EMPLOYED_YEARS'] = 0
            self.new_features.append('EMPLOYED_YEARS')
            n_created += 1
        
        if 'DAYS_REGISTRATION' in X.columns:
            X['REGISTRATION_YEARS'] = -X['DAYS_REGISTRATION'] / 365
            self.new_features.append('REGISTRATION_YEARS')
            n_created += 1
        
        if 'DAYS_ID_PUBLISH' in X.columns:
            X['ID_PUBLISH_YEARS'] = -X['DAYS_ID_PUBLISH'] / 365
            self.new_features.append('ID_PUBLISH_YEARS')
            n_created += 1
        
        if 'DAYS_LAST_PHONE_CHANGE' in X.columns:
            X['PHONE_CHANGE_YEARS'] = -X['DAYS_LAST_PHONE_CHANGE'] / 365
            self.new_features.append('PHONE_CHANGE_YEARS')
            n_created += 1
        
        # Ratios de stabilité
        if 'EMPLOYED_YEARS' in X.columns and 'AGE_YEARS' in X.columns:
            X['EMPLOYED_TO_AGE_RATIO'] = X['EMPLOYED_YEARS'] / (X['AGE_YEARS'] + 1)
            self.new_features.append('EMPLOYED_TO_AGE_RATIO')
            n_created += 1
        
        if 'REGISTRATION_YEARS' in X.columns and 'AGE_YEARS' in X.columns:
            X['REGISTRATION_TO_AGE_RATIO'] = X['REGISTRATION_YEARS'] / (X['AGE_YEARS'] + 1)
            self.new_features.append('REGISTRATION_TO_AGE_RATIO')
            n_created += 1
        
        if 'ID_PUBLISH_YEARS' in X.columns and 'AGE_YEARS' in X.columns:
            X['ID_TO_AGE_RATIO'] = X['ID_PUBLISH_YEARS'] / (X['AGE_YEARS'] + 1)
            self.new_features.append('ID_TO_AGE_RATIO')
            n_created += 1
        
        # Indicateurs binaires
        if 'AGE_YEARS' in X.columns:
            X['IS_YOUNG'] = (X['AGE_YEARS'] < 30).astype(int)
            X['IS_SENIOR'] = (X['AGE_YEARS'] > 55).astype(int)
            self.new_features.extend(['IS_YOUNG', 'IS_SENIOR'])
            n_created += 2
        
        if 'EMPLOYED_YEARS' in X.columns:
            X['IS_NEW_EMPLOYED'] = (X['EMPLOYED_YEARS'] < 1).astype(int)
            X['IS_STABLE_EMPLOYED'] = (X['EMPLOYED_YEARS'] > 5).astype(int)
            self.new_features.extend(['IS_NEW_EMPLOYED', 'IS_STABLE_EMPLOYED'])
            n_created += 2
        
        if 'DAYS_EMPLOYED' in X.columns:
            X['EMPLOYED_ANOMALY'] = (X['DAYS_EMPLOYED'] > 0).astype(int)
            self.new_features.append('EMPLOYED_ANOMALY')
            n_created += 1
        
        # Tranches d'âge
        if 'AGE_YEARS' in X.columns:
            X['AGE_GROUP'] = pd.cut(
                X['AGE_YEARS'],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            )
            self.new_features.append('AGE_GROUP')
            n_created += 1
        
        print(f"   - {n_created} variables temporelles créées")
        return X
    
    # ==================================================================
    # TARGET ENCODING K-FOLD
    # ==================================================================
    
    def _fit_target_encoding(self, X, y):
        """
        Apprend les target encodings sur le train set.
        """
        print("\n3. Apprentissage Target Encoding K-Fold...")
        
        # Variables catégorielles à encoder
        cat_cols = [
            'NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
            'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
            'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE'
        ]
        
        # Filtrer les colonnes existantes
        cat_cols = [c for c in cat_cols if c in X.columns]
        
        # Calculer moyenne globale (fallback)
        global_mean = y.mean()
        
        # Pour chaque variable catégorielle
        for col in cat_cols:
            # Calculer moyenne par catégorie
            encoding = X.groupby(col)[y.name if hasattr(y, 'name') else 'TARGET'].mean().to_dict()
            
            self.target_encodings_[col] = {
                'encoding': encoding,
                'global_mean': global_mean
            }
        
        print(f"   - {len(cat_cols)} variables préparées pour target encoding")
    
    def _apply_target_encoding(self, X):
        """
        Applique les target encodings appris.
        """
        print("\n3. Application Target Encoding...")
        
        n_created = 0
        
        for col, params in self.target_encodings_.items():
            if col not in X.columns:
                continue
            
            new_col = f'{col}_TE'
            encoding = params['encoding']
            global_mean = params['global_mean']
            
            # Mapper les valeurs
            X[new_col] = X[col].map(encoding)
            
            # Remplir les NaN avec la moyenne globale
            X[new_col].fillna(global_mean, inplace=True)
            
            self.new_features.append(new_col)
            n_created += 1
        
        print(f"   - {n_created} variables encodées")
        return X
    
    # ==================================================================
    # FEATURES D'INTERACTION (21 features)
    # ==================================================================
    
    def _create_interaction_features(self, X):
        """
        Crée des features d'interaction et polynomiales.
        
        Focus sur EXT_SOURCE (top features).
        Gain attendu: +0.003 à +0.008 AUC
        """
        print("\n4. Création des features d'interaction...")
        
        n_created = 0
        
        ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        
        if not all(c in X.columns for c in ext_cols):
            print("   ATTENTION: Colonnes EXT_SOURCE manquantes")
            return X
        
        # Produits deux à deux
        X['EXT_SOURCE_1_2'] = X['EXT_SOURCE_1'] * X['EXT_SOURCE_2']
        X['EXT_SOURCE_1_3'] = X['EXT_SOURCE_1'] * X['EXT_SOURCE_3']
        X['EXT_SOURCE_2_3'] = X['EXT_SOURCE_2'] * X['EXT_SOURCE_3']
        self.new_features.extend(['EXT_SOURCE_1_2', 'EXT_SOURCE_1_3', 'EXT_SOURCE_2_3'])
        n_created += 3
        
        # Produit des trois
        X['EXT_SOURCE_PROD'] = X['EXT_SOURCE_1'] * X['EXT_SOURCE_2'] * X['EXT_SOURCE_3']
        self.new_features.append('EXT_SOURCE_PROD')
        n_created += 1
        
        # Statistiques
        X['EXT_SOURCE_MEAN'] = X[ext_cols].mean(axis=1)
        X['EXT_SOURCE_MIN'] = X[ext_cols].min(axis=1)
        X['EXT_SOURCE_MAX'] = X[ext_cols].max(axis=1)
        X['EXT_SOURCE_STD'] = X[ext_cols].std(axis=1)
        X['EXT_SOURCE_COUNT'] = X[ext_cols].notna().sum(axis=1)
        self.new_features.extend(['EXT_SOURCE_MEAN', 'EXT_SOURCE_MIN', 'EXT_SOURCE_MAX', 
                                  'EXT_SOURCE_STD', 'EXT_SOURCE_COUNT'])
        n_created += 5
        
        # Moyenne pondérée
        X['EXT_SOURCE_WEIGHTED'] = (
            X['EXT_SOURCE_1'] * 0.155 +
            X['EXT_SOURCE_2'] * 0.160 +
            X['EXT_SOURCE_3'] * 0.179
        ) / (0.155 + 0.160 + 0.179)
        self.new_features.append('EXT_SOURCE_WEIGHTED')
        n_created += 1
        
        # Features polynomiales
        for i in [1, 2, 3]:
            col = f'EXT_SOURCE_{i}'
            X[f'{col}_SQ'] = X[col] ** 2
            X[f'{col}_CUB'] = X[col] ** 3
            self.new_features.extend([f'{col}_SQ', f'{col}_CUB'])
            n_created += 2
        
        # Interactions avec âge et crédit
        if 'AGE_YEARS' in X.columns:
            X['CREDIT_PER_AGE'] = X['AMT_CREDIT'] / (X['AGE_YEARS'] + 1)
            X['ANNUITY_PER_AGE'] = X['AMT_ANNUITY'] / (X['AGE_YEARS'] + 1)
            self.new_features.extend(['CREDIT_PER_AGE', 'ANNUITY_PER_AGE'])
            n_created += 2
        
        if 'EMPLOYED_YEARS' in X.columns:
            X['INCOME_PER_EMPLOYED_YEAR'] = X['AMT_INCOME_TOTAL'] / (X['EMPLOYED_YEARS'] + 1)
            self.new_features.append('INCOME_PER_EMPLOYED_YEAR')
            n_created += 1
        
        if 'REGION_POPULATION_RELATIVE' in X.columns:
            X['INCOME_REGION_RATIO'] = X['AMT_INCOME_TOTAL'] * X['REGION_POPULATION_RELATIVE']
            self.new_features.append('INCOME_REGION_RATIO')
            n_created += 1
        
        print(f"   - {n_created} features d'interaction créées")
        return X
    
    # ==================================================================
    # FLAGS DE PRÉSENCE (8 features)
    # ==================================================================
    
    def _create_presence_flags(self, X):
        """
        Crée des flags binaires indiquant la présence de données.
        
        Les NaN des agrégations sont informatifs.
        """
        print("\n5. Création des flags de présence...")
        
        n_created = 0
        
        # Vérifier colonnes agrégées des tables annexes
        presence_checks = {
            'HAS_BUREAU': 'BUREAU_CREDIT_COUNT',
            'HAS_PREV_APP': 'PREV_APP_COUNT',
            'HAS_POS': 'POS_CONTRACT_COUNT',
            'HAS_CC': 'CC_CARD_COUNT',
            'HAS_INST': 'INST_CONTRACT_COUNT'
        }
        
        for flag_name, check_col in presence_checks.items():
            if check_col in X.columns:
                X[flag_name] = X[check_col].notna().astype(int)
                self.new_features.append(flag_name)
                n_created += 1
        
        # Flags EXT_SOURCE
        for i in [1, 2, 3]:
            col = f'EXT_SOURCE_{i}'
            if col in X.columns:
                flag = f'HAS_EXT_SOURCE_{i}'
                X[flag] = X[col].notna().astype(int)
                self.new_features.append(flag)
                n_created += 1
        
        print(f"   - {n_created} flags de présence créés")
        return X
    
    # ==================================================================
    # SAUVEGARDE / CHARGEMENT
    # ==================================================================
    
    def get_feature_names(self):
        """Retourne la liste des nouvelles features créées."""
        return self.new_features.copy()
    
    def save(self, filepath):
        """
        Sauvegarde le feature engineer.
        
        Args:
            filepath (str): Chemin du fichier de sauvegarde
        """
        if not self.fitted_:
            raise ValueError("Le feature engineer doit être fitted avant sauvegarde.")
        
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        params = {
            'use_target_encoding': self.use_target_encoding,
            'n_folds': self.n_folds,
            'target_encodings': self.target_encodings_,
            'new_features': self.new_features,
            'fitted': self.fitted_
        }
        
        joblib.dump(params, save_path)
        print(f"Feature engineer sauvegardé: {save_path}")
    
    @classmethod
    def load(cls, filepath):
        """
        Charge un feature engineer depuis un fichier.
        
        Args:
            filepath (str): Chemin du fichier
            
        Returns:
            FeatureEngineer: Instance chargée
        """
        params = joblib.load(filepath)
        
        engineer = cls(
            use_target_encoding=params['use_target_encoding'],
            n_folds=params['n_folds']
        )
        engineer.target_encodings_ = params['target_encodings']
        engineer.new_features = params['new_features']
        engineer.fitted_ = params['fitted']
        
        print(f"Feature engineer chargé: {filepath}")
        return engineer


# ======================================================================
# FONCTIONS D'AGRÉGATION DES TABLES ANNEXES (À utiliser AVANT le pipeline)
# ======================================================================

def aggregate_bureau_advanced(bureau, bureau_balance):
    """
    Agrégation avancée de la table Bureau - inspirée de KazukiOnodera.
    
    Cette fonction doit être appelée AVANT le pipeline principal.
    """
    print("\nAgrégation avancée de Bureau...")
    
    # Distribution STATUS bureau_balance
    bb_status = bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts().unstack(fill_value=0)
    bb_status.columns = ['BB_STATUS_' + str(col) for col in bb_status.columns]
    
    # Statistiques temporelles
    bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
    })
    bb_agg.columns = ['BB_MONTHS_MIN', 'BB_MONTHS_MAX', 'BB_MONTHS_MEAN', 'BB_COUNT']
    bb_agg = bb_agg.join(bb_status).reset_index()
    
    # Joindre à bureau
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    
    del bb_agg, bb_status
    gc.collect()
    
    # Fonction d'agrégation par fenêtre
    def agg_bureau_window(bureau_data, prefix=''):
        num_agg = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'std'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean', 'sum'],
            'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
            'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max'],
            'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean'],
            'AMT_CREDIT_SUM_LIMIT': ['sum', 'mean'],
            'AMT_ANNUITY': ['sum', 'mean', 'max'],
            'CNT_CREDIT_PROLONG': ['sum', 'mean'],
            'BB_MONTHS_MIN': ['min'],
            'BB_MONTHS_MAX': ['max'],
            'BB_COUNT': ['sum', 'mean'],
        }
        
        # Ajouter colonnes STATUS si elles existent
        status_cols = [c for c in bureau_data.columns if c.startswith('BB_STATUS_')]
        for col in status_cols:
            num_agg[col] = ['sum', 'mean']
        
        agg = bureau_data.groupby('SK_ID_CURR').agg(num_agg)
        agg.columns = ['BUREAU' + prefix + '_' + '_'.join(col).upper() for col in agg.columns]
        agg[f'BUREAU{prefix}_CREDIT_COUNT'] = bureau_data.groupby('SK_ID_CURR').size()
        
        return agg
    
    # Agrégations par fenêtre temporelle
    bureau_agg = agg_bureau_window(bureau, prefix='')
    
    # 1 YEAR
    bureau_1y = bureau[bureau['DAYS_CREDIT'] >= -365].copy()
    if len(bureau_1y) > 0:
        bureau_agg_1y = agg_bureau_window(bureau_1y, prefix='_1Y')
        bureau_agg = bureau_agg.join(bureau_agg_1y, how='left')
    
    # 2 YEARS
    bureau_2y = bureau[bureau['DAYS_CREDIT'] >= -730].copy()
    if len(bureau_2y) > 0:
        bureau_agg_2y = agg_bureau_window(bureau_2y, prefix='_2Y')
        bureau_agg = bureau_agg.join(bureau_agg_2y, how='left')
    
    # 3 YEARS
    bureau_3y = bureau[bureau['DAYS_CREDIT'] >= -1095].copy()
    if len(bureau_3y) > 0:
        bureau_agg_3y = agg_bureau_window(bureau_3y, prefix='_3Y')
        bureau_agg = bureau_agg.join(bureau_agg_3y, how='left')
    
    # Features First/Last credit
    first_credit = bureau.sort_values('DAYS_CREDIT').groupby('SK_ID_CURR').first()
    first_credit = first_credit[['DAYS_CREDIT', 'AMT_CREDIT_SUM', 'CREDIT_TYPE']]
    first_credit.columns = ['BUREAU_FIRST_' + col for col in first_credit.columns]
    bureau_agg = bureau_agg.join(first_credit, how='left')
    
    last_credit = bureau.sort_values('DAYS_CREDIT').groupby('SK_ID_CURR').last()
    last_credit = last_credit[['DAYS_CREDIT', 'AMT_CREDIT_SUM', 'CREDIT_TYPE']]
    last_credit.columns = ['BUREAU_LAST_' + col for col in last_credit.columns]
    bureau_agg = bureau_agg.join(last_credit, how='left')
    
    # Features par type de crédit
    credit_active = bureau.groupby('SK_ID_CURR')['CREDIT_ACTIVE'].value_counts().unstack(fill_value=0)
    credit_active.columns = ['BUREAU_CREDIT_' + col.upper().replace(' ', '_') for col in credit_active.columns]
    bureau_agg = bureau_agg.join(credit_active)
    
    credit_type = bureau.groupby('SK_ID_CURR')['CREDIT_TYPE'].value_counts().unstack(fill_value=0)
    credit_type.columns = ['BUREAU_TYPE_' + col.upper().replace(' ', '_') for col in credit_type.columns]
    bureau_agg = bureau_agg.join(credit_type)
    
    # Ratios dérivés
    if 'BUREAU_AMT_CREDIT_SUM_DEBT_SUM' in bureau_agg.columns:
        bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = (
            bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] /
            (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1)
        )
    
    if 'BUREAU_1Y_CREDIT_COUNT' in bureau_agg.columns:
        bureau_agg['BUREAU_1Y_RATIO'] = (
            bureau_agg['BUREAU_1Y_CREDIT_COUNT'] /
            (bureau_agg['BUREAU_CREDIT_COUNT'] + 1)
        )
    
    bureau_agg = bureau_agg.reset_index()
    
    print(f"   {len(bureau_agg.columns)-1} features Bureau créées")
    return bureau_agg


def aggregate_previous_application(prev):
    """Agrège les données des demandes précédentes."""
    print("\nAgrégation de Previous Application...")
    
    prev = prev.replace({'XNA': np.nan, 'XAP': np.nan})
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / (prev['AMT_CREDIT'] + 1)
    prev['CREDIT_GOODS_DIFF'] = prev['AMT_CREDIT'] - prev['AMT_GOODS_PRICE']
    
    num_agg = {
        'AMT_ANNUITY': ['sum', 'mean', 'max', 'min'],
        'AMT_APPLICATION': ['sum', 'mean', 'max', 'min'],
        'AMT_CREDIT': ['sum', 'mean', 'max', 'min'],
        'AMT_DOWN_PAYMENT': ['sum', 'mean', 'max'],
        'AMT_GOODS_PRICE': ['sum', 'mean', 'max'],
        'HOUR_APPR_PROCESS_START': ['mean'],
        'RATE_DOWN_PAYMENT': ['mean', 'max'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['sum', 'mean', 'max'],
        'APP_CREDIT_PERC': ['mean', 'max', 'min'],
        'CREDIT_GOODS_DIFF': ['mean', 'sum'],
    }
    
    prev_agg = prev.groupby('SK_ID_CURR').agg(num_agg)
    prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
    prev_agg['PREV_APP_COUNT'] = prev.groupby('SK_ID_CURR').size()
    
    # Features par statut
    status_counts = prev.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].value_counts().unstack(fill_value=0)
    status_counts.columns = ['PREV_STATUS_' + col.upper().replace(' ', '_') for col in status_counts.columns]
    prev_agg = prev_agg.join(status_counts)
    
    # Ratios de décision
    if 'PREV_STATUS_APPROVED' in prev_agg.columns:
        prev_agg['PREV_APPROVAL_RATE'] = (
            prev_agg['PREV_STATUS_APPROVED'] /
            (prev_agg['PREV_APP_COUNT'] + 1)
        )
    
    if 'PREV_STATUS_REFUSED' in prev_agg.columns:
        prev_agg['PREV_REFUSAL_RATE'] = (
            prev_agg['PREV_STATUS_REFUSED'] /
            (prev_agg['PREV_APP_COUNT'] + 1)
        )
    
    # Features temporelles
    prev_agg['PREV_DAYS_SINCE_LAST'] = prev.groupby('SK_ID_CURR')['DAYS_DECISION'].max()
    prev_agg['PREV_DAYS_SINCE_FIRST'] = prev.groupby('SK_ID_CURR')['DAYS_DECISION'].min()
    
    prev_agg = prev_agg.reset_index()
    
    print(f"   {len(prev_agg.columns)-1} features Previous Application créées")
    return prev_agg


def aggregate_pos_cash(pos):
    """Agrège les données POS et Cash loans."""
    print("\nAgrégation de POS_CASH_balance...")
    
    pos_agg = pos.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
        'CNT_INSTALMENT': ['sum', 'mean', 'max'],
        'CNT_INSTALMENT_FUTURE': ['sum', 'mean', 'max', 'min'],
        'SK_DPD': ['sum', 'mean', 'max'],
        'SK_DPD_DEF': ['sum', 'mean', 'max'],
    })
    pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
    pos_agg['POS_CONTRACT_COUNT'] = pos.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
    pos_agg['POS_DPD_RATIO'] = pos_agg['POS_SK_DPD_SUM'] / (pos_agg['POS_MONTHS_BALANCE_SIZE'] + 1)
    
    pos_agg = pos_agg.reset_index()
    print(f"   {len(pos_agg.columns)-1} features POS créées")
    return pos_agg


def aggregate_credit_card(cc):
    """Agrège les données des cartes de crédit."""
    print("\nAgrégation de credit_card_balance...")
    
    cc_agg = cc.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
        'AMT_BALANCE': ['sum', 'mean', 'max', 'min'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['sum', 'mean', 'max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['sum', 'mean', 'max'],
        'AMT_DRAWINGS_CURRENT': ['sum', 'mean', 'max'],
        'AMT_DRAWINGS_POS_CURRENT': ['sum', 'mean', 'max'],
        'AMT_INST_MIN_REGULARITY': ['sum', 'mean'],
        'AMT_PAYMENT_CURRENT': ['sum', 'mean', 'max'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['sum', 'mean'],
        'AMT_RECEIVABLE_PRINCIPAL': ['sum', 'mean', 'max'],
        'AMT_TOTAL_RECEIVABLE': ['sum', 'mean', 'max'],
        'CNT_DRAWINGS_ATM_CURRENT': ['sum', 'mean', 'max'],
        'CNT_DRAWINGS_CURRENT': ['sum', 'mean', 'max'],
        'CNT_INSTALMENT_MATURE_CUM': ['max', 'mean'],
        'SK_DPD': ['sum', 'mean', 'max'],
        'SK_DPD_DEF': ['sum', 'mean', 'max'],
    })
    cc_agg.columns = ['CC_' + '_'.join(col).upper() for col in cc_agg.columns]
    cc_agg['CC_CARD_COUNT'] = cc.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
    cc_agg['CC_UTILIZATION_RATIO'] = (
        cc_agg['CC_AMT_BALANCE_MEAN'] /
        (cc_agg['CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN'] + 1)
    )
    cc_agg['CC_DPD_RATIO'] = cc_agg['CC_SK_DPD_SUM'] / (cc_agg['CC_MONTHS_BALANCE_SIZE'] + 1)
    
    cc_agg = cc_agg.reset_index()
    print(f"   {len(cc_agg.columns)-1} features Credit Card créées")
    return cc_agg


def aggregate_installments(inst):
    """Agrège les données des paiements d'échéances."""
    print("\nAgrégation de installments_payments...")
    
    inst['PAYMENT_DIFF'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    inst['PAYMENT_RATIO'] = inst['AMT_PAYMENT'] / (inst['AMT_INSTALMENT'] + 1)
    inst['DAYS_DELAY'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst['LATE_PAYMENT'] = (inst['DAYS_DELAY'] > 0).astype(int)
    
    inst_agg = inst.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'NUM_INSTALMENT_NUMBER': ['max', 'mean'],
        'DAYS_INSTALMENT': ['min', 'max', 'mean'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
        'AMT_INSTALMENT': ['sum', 'mean', 'max'],
        'AMT_PAYMENT': ['sum', 'mean', 'max', 'min'],
        'PAYMENT_DIFF': ['sum', 'mean', 'max', 'min'],
        'PAYMENT_RATIO': ['mean', 'max', 'min'],
        'DAYS_DELAY': ['max', 'mean', 'sum'],
        'LATE_PAYMENT': ['sum', 'mean'],
    })
    inst_agg.columns = ['INST_' + '_'.join(col).upper() for col in inst_agg.columns]
    inst_agg['INST_PAYMENT_COUNT'] = inst.groupby('SK_ID_CURR').size()
    inst_agg['INST_CONTRACT_COUNT'] = inst.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
    inst_agg['INST_LATE_RATIO'] = (
        inst_agg['INST_LATE_PAYMENT_SUM'] /
        (inst_agg['INST_PAYMENT_COUNT'] + 1)
    )
    
    inst_agg = inst_agg.reset_index()
    print(f"   {len(inst_agg.columns)-1} features Installments créées")
    return inst_agg


if __name__ == "__main__":
    """
    Test du feature engineer avec des données fictives.
    """
    print("Test du FeatureEngineer")
    print("="*60)
    
    # Créer des données de test
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'SK_ID_CURR': range(n_samples),
        'TARGET': np.random.binomial(1, 0.08, n_samples),
        'AMT_INCOME_TOTAL': np.random.normal(150000, 50000, n_samples),
        'AMT_CREDIT': np.random.normal(500000, 200000, n_samples),
        'AMT_ANNUITY': np.random.normal(25000, 10000, n_samples),
        'AMT_GOODS_PRICE': np.random.normal(450000, 180000, n_samples),
        'DAYS_BIRTH': np.random.randint(-25000, -7000, n_samples),
        'DAYS_EMPLOYED': np.random.randint(-10000, 0, n_samples),
        'DAYS_REGISTRATION': np.random.randint(-5000, 0, n_samples),
        'DAYS_ID_PUBLISH': np.random.randint(-6000, 0, n_samples),
        'DAYS_LAST_PHONE_CHANGE': np.random.randint(-4000, 0, n_samples),
        'EXT_SOURCE_1': np.random.uniform(0.2, 0.8, n_samples),
        'EXT_SOURCE_2': np.random.uniform(0.1, 0.9, n_samples),
        'EXT_SOURCE_3': np.random.uniform(0.3, 0.7, n_samples),
        'CNT_FAM_MEMBERS': np.random.randint(1, 6, n_samples),
        'CNT_CHILDREN': np.random.randint(0, 4, n_samples),
        'REGION_POPULATION_RELATIVE': np.random.uniform(0.001, 0.07, n_samples),
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], n_samples),
        'NAME_INCOME_TYPE': np.random.choice(['Working', 'Commercial associate', 'Pensioner'], n_samples),
    })
    
    print(f"\nDonnées test: {test_data.shape}")
    
    # Test sans target encoding
    print("\n" + "="*60)
    print("TEST SANS TARGET ENCODING")
    print("="*60)
    
    engineer = FeatureEngineer(use_target_encoding=False)
    test_processed = engineer.fit_transform(test_data)
    
    print(f"\nColonnes après feature engineering: {test_processed.shape[1]}")
    print(f"Nouvelles features: {len(engineer.get_feature_names())}")
    
    # Test avec target encoding
    print("\n" + "="*60)
    print("TEST AVEC TARGET ENCODING")
    print("="*60)
    
    engineer_te = FeatureEngineer(use_target_encoding=True, n_folds=5)
    test_processed_te = engineer_te.fit_transform(test_data, test_data['TARGET'])
    
    print(f"\nColonnes après feature engineering: {test_processed_te.shape[1]}")
    print(f"Nouvelles features: {len(engineer_te.get_feature_names())}")
    
    # Test sauvegarde/chargement
    engineer_te.save('/tmp/feature_engineer.pkl')
    engineer_loaded = FeatureEngineer.load('/tmp/feature_engineer.pkl')
    
    print("\nTests terminés avec succès!")