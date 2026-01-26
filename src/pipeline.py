"""
Preprocessing Pipeline - Orchestration Complète
Projet: Crédit Scoring - Home Credit Default Risk

Pipeline modulaire qui orchestre tous les preprocessors:
1. MissingValuesHandler - Gestion des valeurs manquantes
2. OutlierHandler - Traitement des outliers
3. FeatureEngineer - Feature engineering avancé
4. CategoricalEncoder - Encodage des variables catégorielles

Architecture:
- Pattern scikit-learn (fit/transform)
- Pipeline fluide en mémoire (pas de sauvegarde intermédiaire)
- Sauvegarde des artefacts pour production
- Compatible avec train/test split

Auteur: Credit Scoring Project
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import gc
from datetime import datetime


class PreprocessingPipeline:
    """
    Pipeline de preprocessing complet pour le crédit scoring.
    
    Cette classe orchestre tous les preprocessors dans l'ordre correct.
    
    Attributes:
        use_outlier_handler (bool): Activer traitement des outliers
        outlier_method (str): Méthode pour outliers ('winsorize', 'cap', 'log', 'remove')
        use_feature_engineering (bool): Activer feature engineering
        use_target_encoding (bool): Activer target encoding
        use_categorical_encoding (bool): Activer encodage catégoriel
        
        missing_handler_ (MissingValuesHandler): Handler des valeurs manquantes
        outlier_handler_ (OutlierHandler): Handler des outliers
        feature_engineer_ (FeatureEngineer): Feature engineer
        categorical_encoder_ (CategoricalEncoder): Encodeur catégoriel
        
        fitted_ (bool): Indique si le pipeline a été fitted
    """
    
    def __init__(self,
                 use_outlier_handler=True,
                 outlier_method='winsorize',
                 apply_all_outlier_methods=False,
                 use_feature_engineering=True,
                 use_target_encoding=True,
                 use_categorical_encoding=True):
        """
        Initialise le pipeline.
        
        Args:
            use_outlier_handler (bool): Activer traitement des outliers
            outlier_method (str): Méthode pour outliers (si apply_all_outlier_methods=False)
            apply_all_outlier_methods (bool): Si True, applique toutes les méthodes outliers
            use_feature_engineering (bool): Activer feature engineering
            use_target_encoding (bool): Activer target encoding
            use_categorical_encoding (bool): Activer encodage catégoriel
        """
        self.use_outlier_handler = use_outlier_handler
        self.outlier_method = outlier_method
        self.apply_all_outlier_methods = apply_all_outlier_methods
        self.use_feature_engineering = use_feature_engineering
        self.use_target_encoding = use_target_encoding
        self.use_categorical_encoding = use_categorical_encoding
        
        # Handlers (seront initialisés lors du fit)
        self.missing_handler_ = None
        self.outlier_handler_ = None
        self.feature_engineer_ = None
        self.categorical_encoder_ = None
        
        self.fitted_ = False
        
    def fit(self, X, y=None):
        """
        Fit tous les preprocessors sur le train set.
        
        L'ordre est critique:
        1. Missing values
        2. Outliers
        3. Feature engineering
        4. Categorical encoding
        
        Args:
            X (pd.DataFrame): Dataset d'entraînement
            y (pd.Series): Variable cible (optionnel, nécessaire pour target encoding)
            
        Returns:
            self: Instance fitted
        """
        print("\n" + "="*80)
        print("FIT PREPROCESSING PIPELINE")
        print("="*80)
        print(f"Dataset initial: {X.shape}")
        
        # Importer les modules nécessaires
        from src.preprocessors.missing_handler import MissingValuesHandler
        if self.use_outlier_handler:
            from src.preprocessors.outlier_handler import OutlierHandler
        if self.use_feature_engineering:
            from src.preprocessors.feature_engineer import FeatureEngineer
        if self.use_categorical_encoding:
            from src.preprocessors.encoder import CategoricalEncoder
        
        # 1. Missing Values Handler
        print("\n" + "-"*80)
        print("ÉTAPE 1: Missing Values Handler")
        print("-"*80)
        self.missing_handler_ = MissingValuesHandler()
        self.missing_handler_.fit(X, y)
        
        # 2. Outlier Handler (optionnel)
        if self.use_outlier_handler:
            print("\n" + "-"*80)
            print("ÉTAPE 2: Outlier Handler")
            print("-"*80)
            self.outlier_handler_ = OutlierHandler(
                method=self.outlier_method,
                apply_all_methods=self.apply_all_outlier_methods,
                lower_percentile=0.05,
                upper_percentile=0.95
            )
            # Fit sur données après traitement des NaN
            X_temp = self.missing_handler_.transform(X)
            self.outlier_handler_.fit(X_temp, y)
            del X_temp
            gc.collect()
        
        # 3. Feature Engineer (optionnel)
        if self.use_feature_engineering:
            print("\n" + "-"*80)
            print("ÉTAPE 3: Feature Engineer")
            print("-"*80)
            self.feature_engineer_ = FeatureEngineer(
                use_target_encoding=self.use_target_encoding,
                n_folds=5
            )
            # Fit sur données après outliers
            X_temp = self.missing_handler_.transform(X)
            if self.use_outlier_handler:
                X_temp = self.outlier_handler_.transform(X_temp)
            self.feature_engineer_.fit(X_temp, y)
            del X_temp
            gc.collect()
        
        # 4. Categorical Encoder (optionnel)
        if self.use_categorical_encoding:
            print("\n" + "-"*80)
            print("ÉTAPE 4: Categorical Encoder")
            print("-"*80)
            self.categorical_encoder_ = CategoricalEncoder(
                target_encoding_vars=[
                    'NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
                    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                    'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
                    'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE'
                ],
                onehot_vars=[
                    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                    'NAME_CONTRACT_TYPE', 'EMERGENCYSTATE_MODE'
                ],
                n_folds=5
            )
            # Fit sur données après feature engineering
            X_temp = self.missing_handler_.transform(X)
            if self.use_outlier_handler:
                X_temp = self.outlier_handler_.transform(X_temp)
            if self.use_feature_engineering:
                X_temp = self.feature_engineer_.transform(X_temp)
            self.categorical_encoder_.fit(X_temp, y)
            del X_temp
            gc.collect()
        
        self.fitted_ = True
        
        print("\n" + "="*80)
        print("FIT TERMINÉ")
        print("="*80)
        
        return self
    
    def transform(self, X):
        """
        Applique tous les preprocessors dans l'ordre.
        
        Args:
            X (pd.DataFrame): Dataset à transformer (train ou test)
            
        Returns:
            pd.DataFrame: Dataset transformé
        """
        if not self.fitted_:
            raise ValueError("Le pipeline doit être fitted avant transform(). Utilisez fit() ou fit_transform().")
        
        print("\n" + "="*80)
        print("TRANSFORM PREPROCESSING PIPELINE")
        print("="*80)
        print(f"Dataset initial: {X.shape}")
        
        # Copie pour éviter modification inplace
        X = X.copy()
        
        # 1. Missing Values
        print("\n" + "-"*80)
        print("ÉTAPE 1: Traitement des valeurs manquantes")
        print("-"*80)
        X = self.missing_handler_.transform(X)
        print(f"Shape après missing values: {X.shape}")
        
        # 2. Outliers
        if self.use_outlier_handler and self.outlier_handler_ is not None:
            print("\n" + "-"*80)
            print("ÉTAPE 2: Traitement des outliers")
            print("-"*80)
            X = self.outlier_handler_.transform(X)
            print(f"Shape après outliers: {X.shape}")
        
        # 3. Feature Engineering
        if self.use_feature_engineering and self.feature_engineer_ is not None:
            print("\n" + "-"*80)
            print("ÉTAPE 3: Feature engineering")
            print("-"*80)
            X = self.feature_engineer_.transform(X)
            print(f"Shape après feature engineering: {X.shape}")
        
        # 4. Categorical Encoding
        if self.use_categorical_encoding and self.categorical_encoder_ is not None:
            print("\n" + "-"*80)
            print("ÉTAPE 4: Encodage catégoriel")
            print("-"*80)
            X = self.categorical_encoder_.transform(X)
            # Optionnel: supprimer colonnes catégorielles originales
            X = self.categorical_encoder_.drop_original_columns(X, keep_original=False)
            print(f"Shape après encodage: {X.shape}")
        
        print("\n" + "="*80)
        print("TRANSFORM TERMINÉ")
        print("="*80)
        print(f"Dataset final: {X.shape}")
        
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
    # SAUVEGARDE / CHARGEMENT
    # ==================================================================
    
    def save(self, directory):
        """
        Sauvegarde le pipeline complet et tous ses handlers.
        
        Structure:
        directory/
        ├── pipeline_config.pkl         # Configuration du pipeline
        ├── missing_handler.pkl         # Handler des valeurs manquantes
        ├── outlier_handler.pkl         # Handler des outliers
        ├── feature_engineer.pkl        # Feature engineer
        └── categorical_encoder.pkl     # Encodeur catégoriel
        
        Args:
            directory (str): Répertoire de sauvegarde
        """
        if not self.fitted_:
            raise ValueError("Le pipeline doit être fitted avant sauvegarde.")
        
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("SAUVEGARDE DU PIPELINE")
        print("="*80)
        print(f"Répertoire: {save_dir}")
        
        # Sauvegarder la configuration du pipeline
        config = {
            'use_outlier_handler': self.use_outlier_handler,
            'outlier_method': self.outlier_method,
            'apply_all_outlier_methods': self.apply_all_outlier_methods,
            'use_feature_engineering': self.use_feature_engineering,
            'use_target_encoding': self.use_target_encoding,
            'use_categorical_encoding': self.use_categorical_encoding,
            'fitted': self.fitted_,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(config, save_dir / 'pipeline_config.pkl')
        print("   - Configuration sauvegardée")
        
        # Sauvegarder chaque handler
        if self.missing_handler_ is not None:
            self.missing_handler_.save(save_dir / 'missing_handler.pkl')
        
        if self.outlier_handler_ is not None:
            self.outlier_handler_.save(save_dir / 'outlier_handler.pkl')
        
        if self.feature_engineer_ is not None:
            self.feature_engineer_.save(save_dir / 'feature_engineer.pkl')
        
        if self.categorical_encoder_ is not None:
            self.categorical_encoder_.save(save_dir / 'categorical_encoder.pkl')
        
        print("\n" + "="*80)
        print("SAUVEGARDE TERMINÉE")
        print("="*80)
    
    @classmethod
    def load(cls, directory):
        """
        Charge un pipeline depuis un répertoire.
        
        Args:
            directory (str): Répertoire contenant le pipeline sauvegardé
            
        Returns:
            PreprocessingPipeline: Instance chargée
        """
        load_dir = Path(directory)
        
        if not load_dir.exists():
            raise ValueError(f"Répertoire non trouvé: {load_dir}")
        
        print("\n" + "="*80)
        print("CHARGEMENT DU PIPELINE")
        print("="*80)
        print(f"Répertoire: {load_dir}")
        
        # Charger la configuration
        config = joblib.load(load_dir / 'pipeline_config.pkl')
        print(f"   - Configuration chargée (timestamp: {config['timestamp']})")
        
        # Créer instance du pipeline
        pipeline = cls(
            use_outlier_handler=config['use_outlier_handler'],
            outlier_method=config['outlier_method'],
            apply_all_outlier_methods=config.get('apply_all_outlier_methods', False),
            use_feature_engineering=config['use_feature_engineering'],
            use_target_encoding=config['use_target_encoding'],
            use_categorical_encoding=config['use_categorical_encoding']
        )
        
        # Charger les handlers
        from src.preprocessors.missing_handler import MissingValuesHandler
        from src.preprocessors.outlier_handler import OutlierHandler
        from src.preprocessors.feature_engineer import FeatureEngineer
        from src.preprocessors.encoder import CategoricalEncoder
        
        pipeline.missing_handler_ = MissingValuesHandler.load(load_dir / 'missing_handler.pkl')
        
        if config['use_outlier_handler']:
            pipeline.outlier_handler_ = OutlierHandler.load(load_dir / 'outlier_handler.pkl')
        
        if config['use_feature_engineering']:
            pipeline.feature_engineer_ = FeatureEngineer.load(load_dir / 'feature_engineer.pkl')
        
        if config['use_categorical_encoding']:
            pipeline.categorical_encoder_ = CategoricalEncoder.load(load_dir / 'categorical_encoder.pkl')
        
        pipeline.fitted_ = config['fitted']
        
        print("\n" + "="*80)
        print("CHARGEMENT TERMINÉ")
        print("="*80)
        
        return pipeline
    
    # ==================================================================
    # UTILITAIRES
    # ==================================================================
    
    def get_feature_names(self):
        """
        Retourne la liste de toutes les features après transformation.
        
        Returns:
            list: Noms des features finales
        """
        if not self.fitted_:
            raise ValueError("Le pipeline doit être fitted.")
        
        features = []
        
        # Features de base (application_train)
        # (on ne les liste pas toutes ici, elles sont préservées)
        
        # Features créées par missing_handler
        if self.missing_handler_ is not None:
            features.extend(self.missing_handler_.new_features)
        
        # Features créées par feature_engineer
        if self.feature_engineer_ is not None:
            features.extend(self.feature_engineer_.get_feature_names())
        
        # Features créées par categorical_encoder
        if self.categorical_encoder_ is not None:
            features.extend(self.categorical_encoder_.get_feature_names())
        
        return features
    
    def summary(self):
        """
        Affiche un résumé du pipeline.
        """
        print("\n" + "="*80)
        print("RÉSUMÉ DU PIPELINE")
        print("="*80)
        
        print(f"\nStatut: {'Fitted' if self.fitted_ else 'Non fitted'}")
        
        print("\nÉtapes activées:")
        print(f"   1. Missing Values Handler: OUI")
        print(f"   2. Outlier Handler: {'OUI' if self.use_outlier_handler else 'NON'}")
        if self.use_outlier_handler:
            if self.apply_all_outlier_methods:
                print(f"      - Mode: MULTI (toutes les méthodes)")
            else:
                print(f"      - Mode: SINGLE")
                print(f"      - Méthode: {self.outlier_method}")
        print(f"   3. Feature Engineering: {'OUI' if self.use_feature_engineering else 'NON'}")
        if self.use_feature_engineering:
            print(f"      - Target Encoding: {'OUI' if self.use_target_encoding else 'NON'}")
        print(f"   4. Categorical Encoding: {'OUI' if self.use_categorical_encoding else 'NON'}")
        
        if self.fitted_:
            print("\nFeatures créées:")
            new_features = self.get_feature_names()
            print(f"   - Total: {len(new_features)} nouvelles features")
            
            if self.missing_handler_:
                print(f"   - Missing Handler: {len(self.missing_handler_.new_features)}")
            if self.feature_engineer_:
                print(f"   - Feature Engineer: {len(self.feature_engineer_.get_feature_names())}")
            if self.categorical_encoder_:
                print(f"   - Categorical Encoder: {len(self.categorical_encoder_.get_feature_names())}")
        
        print("="*80)


# ======================================================================
# FONCTION WRAPPER POUR USAGE SIMPLE
# ======================================================================

def preprocess_data(train_df, test_df=None, 
                   outlier_method='winsorize',
                   apply_all_outlier_methods=False,
                   save_artifacts=True,
                   artifacts_dir='artifacts/preprocessing'):
    """
    Fonction wrapper pour preprocessing complet train/test.
    
    Usage simple:
        # Mode single (une méthode)
        train_processed, test_processed = preprocess_data(train, test, outlier_method='winsorize')
        
        # Mode multi (toutes les méthodes)
        train_processed, test_processed = preprocess_data(train, test, apply_all_outlier_methods=True)
    
    Args:
        train_df (pd.DataFrame): Dataset d'entraînement
        test_df (pd.DataFrame): Dataset de test (optionnel)
        outlier_method (str): Méthode pour outliers (si apply_all_outlier_methods=False)
        apply_all_outlier_methods (bool): Si True, applique toutes les méthodes outliers
        save_artifacts (bool): Sauvegarder le pipeline
        artifacts_dir (str): Répertoire pour artefacts
        
    Returns:
        tuple: (train_processed, test_processed, pipeline)
    """
    print("\n" + "="*80)
    print("PREPROCESSING COMPLET")
    print("="*80)
    
    # Extraire TARGET du train
    if 'TARGET' in train_df.columns:
        y_train = train_df['TARGET']
    else:
        y_train = None
        print("ATTENTION: TARGET non trouvée dans train_df")
    
    # Créer et fit le pipeline
    pipeline = PreprocessingPipeline(
        use_outlier_handler=True,
        outlier_method=outlier_method,
        apply_all_outlier_methods=apply_all_outlier_methods,
        use_feature_engineering=True,
        use_target_encoding=True,
        use_categorical_encoding=True
    )
    
    # Fit sur train
    train_processed = pipeline.fit_transform(train_df, y_train)
    
    # Transform test si fourni
    test_processed = None
    if test_df is not None:
        test_processed = pipeline.transform(test_df)
    
    # Sauvegarder artefacts
    if save_artifacts:
        pipeline.save(artifacts_dir)
        print(f"\nArtefacts sauvegardés dans {artifacts_dir}")
    
    # Résumé
    pipeline.summary()
    
    return train_processed, test_processed, pipeline


if __name__ == "__main__":
    """
    Test du pipeline avec des données fictives.
    """
    print("Test du PreprocessingPipeline")
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
        'NAME_INCOME_TYPE': np.random.choice(['Working', 'Commercial associate', 'Pensioner'], n_samples),
    })
    
    # Introduire des valeurs manquantes
    train_data.loc[np.random.choice(n_samples, 100), 'EXT_SOURCE_1'] = np.nan
    train_data.loc[np.random.choice(n_samples, 50), 'AMT_GOODS_PRICE'] = np.nan
    
    # Créer test set
    test_data = train_data.copy()
    test_data['SK_ID_CURR'] = range(n_samples, n_samples * 2)
    test_data = test_data.drop('TARGET', axis=1)
    
    print(f"\nTrain shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    # Test du pipeline
    train_processed, test_processed, pipeline = preprocess_data(
        train_data, 
        test_data,
        outlier_method='winsorize',
        save_artifacts=True,
        artifacts_dir='/tmp/test_pipeline'
    )
    
    print(f"\nTrain processed shape: {train_processed.shape}")
    print(f"Test processed shape: {test_processed.shape}")
    
    # Test chargement
    print("\n" + "="*80)
    print("TEST CHARGEMENT")
    print("="*80)
    
    pipeline_loaded = PreprocessingPipeline.load('/tmp/test_pipeline')
    test_processed_2 = pipeline_loaded.transform(test_data)
    
    print(f"\nTest processed (loaded pipeline) shape: {test_processed_2.shape}")
    
    print("\nTests terminés avec succès!")