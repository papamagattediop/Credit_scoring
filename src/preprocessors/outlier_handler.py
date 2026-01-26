"""
Outlier Handler - Preprocessing Pipeline
Projet: Crédit Scoring - Home Credit Default Risk

Méthodes de traitement des outliers:
1. Détection IQR (Interquartile Range)
2. Winsorisation
3. Capping (percentiles)
4. Transformation logarithmique

Stratégie:
- < 1% outliers: Suppression possible
- 1-5% outliers: Winsorisation recommandée
- > 5% outliers: Winsorisation obligatoire
- Distribution asymétrique: Transformation log (si pas de valeurs négatives)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.stats.mstats import winsorize


class OutlierHandler:
    """
    Gestionnaire des outliers pour le pipeline de preprocessing.
    
    Cette classe suit le pattern scikit-learn avec fit() et transform().
    
    Attributes:
        method (str): Méthode de traitement ('winsorize', 'cap', 'log', 'remove')
        lower_percentile (float): Percentile inférieur (défaut: 0.05)
        upper_percentile (float): Percentile supérieur (défaut: 0.95)
        bounds_ (dict): Bornes calculées par variable sur train
        outlier_cols_ (list): Colonnes à traiter
        fitted_ (bool): Indique si fit() a été appelé
    """
    
    def __init__(self, method='winsorize', apply_all_methods=False, 
                 lower_percentile=0.05, upper_percentile=0.95):
        """
        Initialise le handler.
        
        Args:
            method (str): 'winsorize', 'cap', 'log', 'remove' (si apply_all_methods=False)
            apply_all_methods (bool): Si True, ignore method et applique toutes les méthodes
            lower_percentile (float): Percentile inférieur (0-1)
            upper_percentile (float): Percentile supérieur (0-1)
        """
        if not apply_all_methods:
            valid_methods = ['winsorize', 'cap', 'log', 'remove']
            if method not in valid_methods:
                raise ValueError(f"method doit être dans {valid_methods}")
        
        if not (0 < lower_percentile < upper_percentile < 1):
            raise ValueError("0 < lower_percentile < upper_percentile < 1")
        
        self.method = method
        self.apply_all_methods = apply_all_methods
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.bounds_ = {}
        self.outlier_cols_ = []
        self.fitted_ = False
        
    def fit(self, X, y=None):
        """
        Apprend les bornes sur le train set.
        
        Args:
            X (pd.DataFrame): Dataset d'entraînement
            y: Non utilisé (compatibilité scikit-learn)
            
        Returns:
            self: Instance fitted
        """
        print("\n" + "="*60)
        if self.apply_all_methods:
            print("FIT: Calcul des bornes (TOUTES MÉTHODES)")
        else:
            print(f"FIT: Calcul des bornes ({self.method})")
        print("="*60)
        
        # Identifier colonnes numériques (exclure ID et TARGET)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['SK_ID_CURR', 'TARGET']
        self.outlier_cols_ = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"\nNombre de variables à traiter: {len(self.outlier_cols_)}")
        
        # Calculer les bornes pour chaque colonne
        for col in self.outlier_cols_:
            # Pour mode multi ou log: vérifier valeurs négatives
            can_log = not (X[col] <= 0).any()
            
            # Calculer percentiles (utile pour toutes méthodes sauf log seul)
            lower_bound = X[col].quantile(self.lower_percentile)
            upper_bound = X[col].quantile(self.upper_percentile)
            
            self.bounds_[col] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'can_log': can_log
            }
                   
        
        print(f"\nBornes calculées pour {len(self.bounds_)} variables")
        
        # Afficher quelques exemples
        print(f"\nExemples de bornes:")
        for i, (col, bounds) in enumerate(list(self.bounds_.items())[:5]):
            if self.method == 'log':
                print(f"   - {col}: transformation log applicable")
            else:
                print(f"   - {col}: [{bounds['lower']:.2f}, {bounds['upper']:.2f}]")
        
        self.fitted_ = True
        return self
        
    def transform(self, X):
        """
        Applique le traitement des outliers.
        
        Si apply_all_methods=False: applique la méthode choisie
        Si apply_all_methods=True: applique toutes les méthodes et crée colonnes suffixées
        
        Args:
            X (pd.DataFrame): Dataset à transformer
            
        Returns:
            pd.DataFrame: Dataset transformé
        """
        if not self.fitted_:
            raise ValueError("Le handler doit être fitted avant transform(). Utilisez fit() ou fit_transform().")
        
        print("\n" + "="*60)
        if self.apply_all_methods:
            print("TRANSFORM: Application de TOUTES les méthodes")
        else:
            print(f"TRANSFORM: Traitement des outliers ({self.method})")
        print("="*60)
        
        # Copie pour éviter modification inplace
        X = X.copy()
        
        if self.apply_all_methods:
            # Mode MULTI: appliquer toutes les méthodes
            X = self._apply_all_methods(X)
        else:
            # Mode SINGLE: appliquer la méthode choisie
            if self.method == 'winsorize':
                X = self._winsorize(X)
            elif self.method == 'cap':
                X = self._cap(X)
            elif self.method == 'log':
                X = self._log_transform(X)
            elif self.method == 'remove':
                X = self._remove_outliers(X)
        
        return X
        
    def fit_transform(self, X, y=None):
        """
        Fit et transform en une seule étape.
        
        Args:
            X (pd.DataFrame): Dataset d'entraînement
            y: Non utilisé
            
        Returns:
            pd.DataFrame: Dataset transformé
        """
        return self.fit(X, y).transform(X)
    
    def _apply_all_methods(self, X):
        """
        Applique TOUTES les méthodes et crée colonnes suffixées.
        
        Pour chaque variable numérique, crée 4 versions:
        - COL_WINS (winsorisée)
        - COL_CAP (cappée, identique à winsorisée)
        - COL_LOG (log transformée, si possible)
        - Garde aussi la colonne originale
        
        Note: remove n'est PAS appliqué en mode multi car il réduit le dataset
        """
        print("\nApplication de toutes les méthodes en parallèle...")
        print("Note: 'remove' n'est pas appliqué (réduirait le dataset)")
        
        n_created = 0
        
        for col in self.bounds_.keys():
            if col not in X.columns:
                continue
            
            lower = self.bounds_[col]['lower']
            upper = self.bounds_[col]['upper']
            can_log = self.bounds_[col]['can_log']
            
            # 1. Winsorize
            X[f'{col}_WINS'] = X[col].clip(lower=lower, upper=upper)
            n_created += 1
            
            # 2. Cap (identique à winsorize, mais on la crée quand même)
            X[f'{col}_CAP'] = X[col].clip(lower=lower, upper=upper)
            n_created += 1
            
            # 3. Log transform (si possible)
            if can_log:
                X[f'{col}_LOG'] = np.log1p(X[col])
                n_created += 1
        
        print(f"   - {n_created} nouvelles colonnes créées")
        print(f"   - Colonnes originales préservées")
        
        return X
        
    def _winsorize(self, X):
        """
        Applique la winsorisation (remplace outliers par bornes).
        
        Avantages:
        - Préserve toutes les observations
        - Réduit l'influence des valeurs extrêmes
        - Maintient la distribution générale
        """
        print("\nApplication de la winsorisation...")
        
        n_modified = 0
        for col in self.bounds_.keys():
            if col not in X.columns:
                continue
            
            lower = self.bounds_[col]['lower']
            upper = self.bounds_[col]['upper']
            
            # Compter les valeurs modifiées
            n_lower = (X[col] < lower).sum()
            n_upper = (X[col] > upper).sum()
            n_modified += n_lower + n_upper
            
            # Appliquer winsorisation
            X[col] = X[col].clip(lower=lower, upper=upper)
        
        print(f"   - {n_modified:,} valeurs winsorisées sur {len(X) * len(self.bounds_):,}")
        print(f"   - Pourcentage: {n_modified / (len(X) * len(self.bounds_)) * 100:.2f}%")
        
        return X
        
    def _cap(self, X):
        """
        Applique le capping (identique à winsorisation pour notre usage).
        
        Note: Capping et winsorisation sont souvent utilisés de manière interchangeable.
        """
        print("\nApplication du capping...")
        return self._winsorize(X)
        
    def _log_transform(self, X):
        """
        Applique la transformation logarithmique.
        
        Avantages:
        - Réduit l'asymétrie (skewness)
        - Normalise les distributions
        - Utile pour distributions log-normales
        
        Limitations:
        - Ne fonctionne que pour valeurs > 0
        """
        print("\nApplication de la transformation log...")
        
        n_transformed = 0
        for col in self.bounds_.keys():
            if col not in X.columns:
                continue
            
            if self.bounds_[col].get('can_log'):
                # log1p = log(1 + x) pour gérer les valeurs proches de 0
                X[f'{col}_LOG'] = np.log1p(X[col])
                n_transformed += 1
        
        print(f"   - {n_transformed} variables transformées")
        print(f"   - Nouvelles colonnes créées avec suffixe '_LOG'")
        
        return X
        
    def _remove_outliers(self, X):
        """
        Supprime les observations contenant des outliers.
        
        ATTENTION: Réduit la taille du dataset!
        
        Recommandé uniquement si:
        - Très peu d'outliers (< 1%)
        - Outliers clairement erronés
        """
        print("\nSuppression des outliers...")
        
        initial_size = len(X)
        mask = pd.Series(True, index=X.index)
        
        for col in self.bounds_.keys():
            if col not in X.columns:
                continue
            
            lower = self.bounds_[col]['lower']
            upper = self.bounds_[col]['upper']
            
            # Marquer les outliers
            col_mask = (X[col] >= lower) & (X[col] <= upper)
            mask &= col_mask
        
        # Filtrer
        X = X[mask]
        
        n_removed = initial_size - len(X)
        print(f"   - Observations supprimées: {n_removed:,} ({n_removed/initial_size*100:.2f}%)")
        print(f"   - Observations restantes: {len(X):,}")
        
        return X
        
    def detect_outliers_summary(self, X):
        """
        Génère un résumé des outliers détectés (avant traitement).
        
        Args:
            X (pd.DataFrame): Dataset à analyser
            
        Returns:
            pd.DataFrame: Résumé par variable
        """
        if not self.fitted_:
            raise ValueError("Le handler doit être fitted avant detection.")
        
        print("\n" + "="*60)
        print("DETECTION DES OUTLIERS - Méthode IQR")
        print("="*60)
        
        summary = []
        
        for col in self.bounds_.keys():
            if col not in X.columns:
                continue
            
            if self.method == 'log':
                continue  # Pas de détection pour log
            
            lower = self.bounds_[col]['lower']
            upper = self.bounds_[col]['upper']
            
            # Calculer IQR pour contexte
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Bornes IQR classiques (1.5 * IQR)
            iqr_lower = Q1 - 1.5 * IQR
            iqr_upper = Q3 + 1.5 * IQR
            
            # Compter outliers
            n_outliers_lower = (X[col] < iqr_lower).sum()
            n_outliers_upper = (X[col] > iqr_upper).sum()
            n_outliers_total = n_outliers_lower + n_outliers_upper
            
            summary.append({
                'variable': col,
                'n_outliers': n_outliers_total,
                'pct_outliers': n_outliers_total / len(X) * 100,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound_iqr': iqr_lower,
                'upper_bound_iqr': iqr_upper,
                'lower_bound_used': lower,
                'upper_bound_used': upper
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('pct_outliers', ascending=False)
        
        print(f"\nRésumé des outliers:")
        print(f"   - Variables analysées: {len(summary_df)}")
        print(f"   - Variables avec outliers (>0%): {(summary_df['pct_outliers'] > 0).sum()}")
        print(f"\nTop 10 variables avec le plus d'outliers:")
        print(summary_df[['variable', 'n_outliers', 'pct_outliers']].head(10).to_string(index=False))
        
        return summary_df
        
    def save(self, filepath):
        """
        Sauvegarde le handler fitted (bornes).
        
        Args:
            filepath (str): Chemin du fichier de sauvegarde
        """
        if not self.fitted_:
            raise ValueError("Le handler doit être fitted avant sauvegarde.")
        
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        params = {
            'method': self.method,
            'apply_all_methods': self.apply_all_methods,
            'lower_percentile': self.lower_percentile,
            'upper_percentile': self.upper_percentile,
            'bounds': self.bounds_,
            'outlier_cols': self.outlier_cols_,
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
            OutlierHandler: Instance chargée
        """
        params = joblib.load(filepath)
        
        handler = cls(
            method=params['method'],
            apply_all_methods=params.get('apply_all_methods', False),
            lower_percentile=params['lower_percentile'],
            upper_percentile=params['upper_percentile']
        )
        handler.bounds_ = params['bounds']
        handler.outlier_cols_ = params['outlier_cols']
        handler.fitted_ = params['fitted']
        
        print(f"Handler chargé: {filepath}")
        return handler


# Fonction utilitaire pour comparer les méthodes
def compare_methods(df, methods=['winsorize', 'cap', 'log']):
    """
    Compare différentes méthodes de traitement des outliers.
    
    Args:
        df (pd.DataFrame): Dataset à traiter
        methods (list): Liste des méthodes à comparer
        
    Returns:
        dict: Résultats par méthode
    """
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Méthode: {method.upper()}")
        print(f"{'='*60}")
        
        handler = OutlierHandler(method=method)
        df_transformed = handler.fit_transform(df)
        
        results[method] = {
            'handler': handler,
            'data': df_transformed,
            'shape': df_transformed.shape
        }
    
    return results


if __name__ == "__main__":
    """
    Test du handler avec des données fictives.
    """
    print("Test du OutlierHandler")
    print("="*60)
    
    # Créer des données de test avec outliers
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'SK_ID_CURR': range(n_samples),
        'AMT_INCOME_TOTAL': np.concatenate([
            np.random.normal(150000, 50000, 950),
            np.random.uniform(500000, 1000000, 50)  # Outliers
        ]),
        'AMT_CREDIT': np.concatenate([
            np.random.normal(500000, 200000, 970),
            np.random.uniform(2000000, 4000000, 30)  # Outliers
        ]),
        'DAYS_EMPLOYED': np.random.randint(-10000, 0, n_samples),
    })
    
    print(f"\nDonnées test: {test_data.shape}")
    print(f"Statistiques avant traitement:")
    print(test_data.describe())
    
    # Test winsorisation
    print("\n" + "="*60)
    print("TEST WINSORISATION")
    print("="*60)
    
    handler = OutlierHandler(method='winsorize', lower_percentile=0.05, upper_percentile=0.95)
    test_processed = handler.fit_transform(test_data)
    
    print(f"\nStatistiques après traitement:")
    print(test_processed.describe())
    
    # Test sauvegarde/chargement
    handler.save('/tmp/outlier_handler.pkl')
    handler_loaded = OutlierHandler.load('/tmp/outlier_handler.pkl')
    
    print("\nTests terminés avec succès!")