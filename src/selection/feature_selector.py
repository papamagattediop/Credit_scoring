"""
Feature Selector - Pipeline de Selection de Features
Projet: Credit Scoring - Home Credit Default Risk

Methodes de selection implementees:
1. Variance Threshold - Elimine features a variance quasi-nulle
2. Correlation Filter - Elimine features tres correlees entre elles
3. Mutual Information - Score par information mutuelle avec TARGET
4. LightGBM Importance - Feature importance d'un modele de reference

Pattern scikit-learn: fit() / transform() / fit_transform()
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Selecteur de features pour le pipeline de preprocessing.

    Cette classe suit le pattern scikit-learn avec fit() et transform().
    Combine plusieurs methodes de selection pour identifier les features
    les plus pertinentes.

    Attributes:
        selected_features_ (list): Liste des features selectionnees
        feature_scores_ (dict): Scores par methode pour chaque feature
        dropped_features_ (dict): Features eliminees par methode
        fitted_ (bool): Indique si fit() a ete appele

    Example:
        >>> selector = FeatureSelector(methods=['variance', 'correlation', 'mutual_info'])
        >>> X_train_selected = selector.fit_transform(X_train, y_train)
        >>> X_test_selected = selector.transform(X_test)
    """

    def __init__(
        self,
        methods: List[str] = None,
        n_features: int = 100,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        random_state: int = 42
    ):
        """
        Initialise le selecteur de features.

        Args:
            methods: Liste des methodes a utiliser
                     Options: ['variance', 'correlation', 'mutual_info', 'lightgbm']
                     Default: ['variance', 'correlation', 'mutual_info', 'lightgbm']
            n_features: Nombre de features a selectionner (top N)
            variance_threshold: Seuil de variance minimum (default: 0.01)
            correlation_threshold: Seuil de correlation pour elimination (default: 0.95)
            random_state: Seed pour reproductibilite
        """
        self.methods = methods or ['variance', 'correlation', 'mutual_info', 'lightgbm']
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state

        # Attributs learned
        self.selected_features_ = []
        self.feature_scores_ = {}
        self.dropped_features_ = {}
        self.variance_scores_ = {}
        self.correlation_pairs_ = []
        self.mutual_info_scores_ = {}
        self.lightgbm_importance_ = {}
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FeatureSelector':
        """
        Apprend les parametres de selection sur le train set.

        Args:
            X: DataFrame des features (sans TARGET)
            y: Serie TARGET pour mutual_info et lightgbm

        Returns:
            self: Instance fitted
        """
        print("\n" + "="*60)
        print("FIT: Apprentissage des parametres de selection")
        print("="*60)
        print(f"Shape initial: {X.shape}")
        print(f"Methodes: {self.methods}")

        # Copie pour eviter modification
        X = X.copy()

        # Identifier colonnes numeriques uniquement
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Features numeriques: {len(numeric_cols)}")

        # Features candidates (commencent avec toutes les numeriques)
        candidates = set(numeric_cols)

        # Etape 1: Variance Threshold
        if 'variance' in self.methods:
            low_variance = self._fit_variance_threshold(X[list(candidates)])
            self.dropped_features_['variance'] = low_variance
            candidates -= set(low_variance)
            print(f"Apres variance filter: {len(candidates)} features")

        # Etape 2: Correlation Filter
        if 'correlation' in self.methods:
            correlated = self._fit_correlation_filter(X[list(candidates)])
            self.dropped_features_['correlation'] = correlated
            candidates -= set(correlated)
            print(f"Apres correlation filter: {len(candidates)} features")

        # Etape 3: Mutual Information (necessite y)
        if 'mutual_info' in self.methods and y is not None:
            self._fit_mutual_info(X[list(candidates)], y)

        # Etape 4: LightGBM Importance (necessite y)
        if 'lightgbm' in self.methods and y is not None:
            self._fit_lightgbm_importance(X[list(candidates)], y)

        # Selection finale: combiner les scores
        self._select_top_features(list(candidates))

        self.fitted_ = True

        print("\n" + "-"*60)
        print(f"RESULTAT: {len(self.selected_features_)} features selectionnees")
        print("-"*60)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applique la selection de features.

        Args:
            X: DataFrame a transformer

        Returns:
            DataFrame avec uniquement les features selectionnees
        """
        if not self.fitted_:
            raise ValueError("Le selecteur doit etre fitted avant transform(). Utilisez fit() ou fit_transform().")

        print("\n" + "="*60)
        print("TRANSFORM: Application de la selection")
        print("="*60)

        # Verifier que toutes les features sont presentes
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            print(f"ATTENTION: {len(missing_features)} features manquantes")
            print(f"Exemples: {list(missing_features)[:5]}")
            # Utiliser uniquement les features disponibles
            available_features = [f for f in self.selected_features_ if f in X.columns]
        else:
            available_features = self.selected_features_

        X_selected = X[available_features].copy()

        print(f"Shape avant: {X.shape}")
        print(f"Shape apres: {X_selected.shape}")
        print(f"Reduction: {X.shape[1] - X_selected.shape[1]} features eliminees")

        return X_selected

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit et transform en une seule etape.

        Args:
            X: DataFrame d'entrainement
            y: Serie TARGET

        Returns:
            DataFrame transforme
        """
        return self.fit(X, y).transform(X)

    def _fit_variance_threshold(self, X: pd.DataFrame) -> List[str]:
        """
        Identifie les features a variance quasi-nulle.

        Args:
            X: DataFrame des features

        Returns:
            Liste des features a eliminer
        """
        print("\n1. Variance Threshold Analysis...")

        # Calculer variance pour chaque feature
        variances = X.var()
        self.variance_scores_ = variances.to_dict()

        # Identifier features a faible variance
        low_variance_features = variances[variances < self.variance_threshold].index.tolist()

        print(f"   Seuil: {self.variance_threshold}")
        print(f"   Features a faible variance: {len(low_variance_features)}")

        if low_variance_features:
            print(f"   Exemples elimines: {low_variance_features[:5]}")

        return low_variance_features

    def _fit_correlation_filter(self, X: pd.DataFrame) -> List[str]:
        """
        Identifie les features tres correlees entre elles.

        Pour chaque paire correlée, garde celle avec la plus grande
        correlation moyenne avec les autres features (plus informative).

        Args:
            X: DataFrame des features

        Returns:
            Liste des features a eliminer
        """
        print("\n2. Correlation Filter Analysis...")

        # Calculer matrice de correlation
        corr_matrix = X.corr().abs()

        # Masque triangulaire superieur
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Trouver paires correlees
        high_corr_pairs = []
        to_drop = set()

        for col in upper_tri.columns:
            correlated_cols = upper_tri.index[upper_tri[col] > self.correlation_threshold].tolist()
            for corr_col in correlated_cols:
                corr_value = corr_matrix.loc[corr_col, col]
                high_corr_pairs.append((col, corr_col, corr_value))

                # Garder celle avec plus grande variance (plus informative)
                if self.variance_scores_.get(col, 0) >= self.variance_scores_.get(corr_col, 0):
                    to_drop.add(corr_col)
                else:
                    to_drop.add(col)

        self.correlation_pairs_ = high_corr_pairs

        print(f"   Seuil: {self.correlation_threshold}")
        print(f"   Paires tres correlees: {len(high_corr_pairs)}")
        print(f"   Features a eliminer: {len(to_drop)}")

        if high_corr_pairs:
            print(f"   Top correlations:")
            for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:3]:
                print(f"      {col1} <-> {col2}: {corr:.4f}")

        return list(to_drop)

    def _fit_mutual_info(self, X: pd.DataFrame, y: pd.Series):
        """
        Calcule les scores d'information mutuelle avec TARGET.

        Args:
            X: DataFrame des features
            y: Serie TARGET
        """
        print("\n3. Mutual Information Analysis...")

        from sklearn.feature_selection import mutual_info_classif

        # Remplacer NaN par median pour le calcul
        X_filled = X.fillna(X.median())

        # Calculer mutual info
        mi_scores = mutual_info_classif(
            X_filled, y,
            random_state=self.random_state,
            n_neighbors=5
        )

        self.mutual_info_scores_ = dict(zip(X.columns, mi_scores))

        # Top features
        top_mi = sorted(self.mutual_info_scores_.items(), key=lambda x: x[1], reverse=True)[:10]

        print(f"   Top 10 features par MI:")
        for feat, score in top_mi:
            print(f"      {feat}: {score:.4f}")

    def _fit_lightgbm_importance(self, X: pd.DataFrame, y: pd.Series):
        """
        Calcule l'importance des features avec LightGBM.

        Args:
            X: DataFrame des features
            y: Serie TARGET
        """
        print("\n4. LightGBM Feature Importance...")

        try:
            import lightgbm as lgb
        except ImportError:
            print("   LightGBM non installe, methode ignoree")
            return

        # Remplacer NaN par median
        X_filled = X.fillna(X.median())

        # Modele leger pour feature importance
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1
        )

        model.fit(X_filled, y)

        # Extraire importance
        importance = model.feature_importances_
        self.lightgbm_importance_ = dict(zip(X.columns, importance))

        # Top features
        top_lgb = sorted(self.lightgbm_importance_.items(), key=lambda x: x[1], reverse=True)[:10]

        print(f"   Top 10 features par LightGBM:")
        for feat, score in top_lgb:
            print(f"      {feat}: {score:.0f}")

    def _select_top_features(self, candidates: List[str]):
        """
        Selectionne les top N features en combinant les scores.

        Strategie: moyenne des rangs normalises de chaque methode.

        Args:
            candidates: Liste des features candidates
        """
        print("\n5. Selection finale (combinaison des scores)...")

        if not candidates:
            print("   ERREUR: Aucune feature candidate")
            return

        # Creer DataFrame de scores
        scores_df = pd.DataFrame(index=candidates)

        # Ajouter scores de chaque methode (normalises 0-1)
        if self.mutual_info_scores_:
            mi_series = pd.Series(self.mutual_info_scores_)
            mi_series = mi_series[mi_series.index.isin(candidates)]
            if len(mi_series) > 0:
                scores_df['mi_score'] = (mi_series - mi_series.min()) / (mi_series.max() - mi_series.min() + 1e-10)

        if self.lightgbm_importance_:
            lgb_series = pd.Series(self.lightgbm_importance_)
            lgb_series = lgb_series[lgb_series.index.isin(candidates)]
            if len(lgb_series) > 0:
                scores_df['lgb_score'] = (lgb_series - lgb_series.min()) / (lgb_series.max() - lgb_series.min() + 1e-10)

        if self.variance_scores_:
            var_series = pd.Series(self.variance_scores_)
            var_series = var_series[var_series.index.isin(candidates)]
            if len(var_series) > 0:
                scores_df['var_score'] = (var_series - var_series.min()) / (var_series.max() - var_series.min() + 1e-10)

        # Remplir NaN par 0
        scores_df = scores_df.fillna(0)

        # Score combine (moyenne des scores disponibles)
        scores_df['combined_score'] = scores_df.mean(axis=1)

        # Selectionner top N
        n_select = min(self.n_features, len(candidates))
        top_features = scores_df.nlargest(n_select, 'combined_score').index.tolist()

        self.selected_features_ = top_features
        self.feature_scores_ = scores_df['combined_score'].to_dict()

        print(f"   Features candidates: {len(candidates)}")
        print(f"   Features selectionnees: {len(self.selected_features_)}")

        # Afficher top 10
        print(f"\n   Top 10 features selectionnees:")
        for i, feat in enumerate(self.selected_features_[:10], 1):
            score = self.feature_scores_.get(feat, 0)
            print(f"      {i}. {feat}: {score:.4f}")

    def get_selected_features(self) -> List[str]:
        """
        Retourne la liste des features selectionnees.

        Returns:
            Liste des noms de features
        """
        if not self.fitted_:
            raise ValueError("Le selecteur doit etre fitted d'abord.")
        return self.selected_features_.copy()

    def get_feature_scores(self) -> Dict[str, float]:
        """
        Retourne les scores de toutes les features.

        Returns:
            Dictionnaire {feature: score}
        """
        if not self.fitted_:
            raise ValueError("Le selecteur doit etre fitted d'abord.")
        return self.feature_scores_.copy()

    def get_dropped_features(self) -> Dict[str, List[str]]:
        """
        Retourne les features eliminees par methode.

        Returns:
            Dictionnaire {methode: [features]}
        """
        if not self.fitted_:
            raise ValueError("Le selecteur doit etre fitted d'abord.")
        return self.dropped_features_.copy()

    def summary(self) -> pd.DataFrame:
        """
        Retourne un resume des scores de features.

        Returns:
            DataFrame avec scores par methode
        """
        if not self.fitted_:
            raise ValueError("Le selecteur doit etre fitted d'abord.")

        data = {
            'selected': [f in self.selected_features_ for f in self.feature_scores_.keys()],
            'combined_score': list(self.feature_scores_.values())
        }

        if self.mutual_info_scores_:
            data['mutual_info'] = [self.mutual_info_scores_.get(f, 0) for f in self.feature_scores_.keys()]

        if self.lightgbm_importance_:
            data['lightgbm'] = [self.lightgbm_importance_.get(f, 0) for f in self.feature_scores_.keys()]

        df = pd.DataFrame(data, index=self.feature_scores_.keys())
        df = df.sort_values('combined_score', ascending=False)

        return df

    def save(self, filepath: str):
        """
        Sauvegarde le selecteur fitted.

        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        if not self.fitted_:
            raise ValueError("Le selecteur doit etre fitted avant sauvegarde.")

        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        params = {
            'methods': self.methods,
            'n_features': self.n_features,
            'variance_threshold': self.variance_threshold,
            'correlation_threshold': self.correlation_threshold,
            'random_state': self.random_state,
            'selected_features': self.selected_features_,
            'feature_scores': self.feature_scores_,
            'dropped_features': self.dropped_features_,
            'variance_scores': self.variance_scores_,
            'correlation_pairs': self.correlation_pairs_,
            'mutual_info_scores': self.mutual_info_scores_,
            'lightgbm_importance': self.lightgbm_importance_,
            'fitted': self.fitted_
        }

        joblib.dump(params, save_path)
        print(f"Selecteur sauvegarde: {save_path}")

    @classmethod
    def load(cls, filepath: str) -> 'FeatureSelector':
        """
        Charge un selecteur depuis un fichier.

        Args:
            filepath: Chemin du fichier

        Returns:
            FeatureSelector: Instance chargee
        """
        params = joblib.load(filepath)

        selector = cls(
            methods=params['methods'],
            n_features=params['n_features'],
            variance_threshold=params['variance_threshold'],
            correlation_threshold=params['correlation_threshold'],
            random_state=params['random_state']
        )

        selector.selected_features_ = params['selected_features']
        selector.feature_scores_ = params['feature_scores']
        selector.dropped_features_ = params['dropped_features']
        selector.variance_scores_ = params['variance_scores']
        selector.correlation_pairs_ = params['correlation_pairs']
        selector.mutual_info_scores_ = params['mutual_info_scores']
        selector.lightgbm_importance_ = params['lightgbm_importance']
        selector.fitted_ = params['fitted']

        print(f"Selecteur charge: {filepath}")
        print(f"Features selectionnees: {len(selector.selected_features_)}")

        return selector


def select_features(
    train_df: pd.DataFrame,
    target_col: str = 'TARGET',
    test_df: pd.DataFrame = None,
    n_features: int = 100,
    methods: List[str] = None
) -> tuple:
    """
    Fonction utilitaire pour la selection de features.

    Args:
        train_df: DataFrame d'entrainement avec TARGET
        target_col: Nom de la colonne cible
        test_df: DataFrame de test (optionnel)
        n_features: Nombre de features a selectionner
        methods: Methodes de selection

    Returns:
        tuple: (train_selected, test_selected, selector) ou (train_selected, selector)

    Example:
        >>> train_sel, test_sel, selector = select_features(train, test_df=test, n_features=100)
    """
    # Separer X et y
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # Creer et entrainer selecteur
    selector = FeatureSelector(
        methods=methods or ['variance', 'correlation', 'mutual_info', 'lightgbm'],
        n_features=n_features
    )

    # Fit et transform train
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Ajouter TARGET
    train_selected = X_train_selected.copy()
    train_selected[target_col] = y_train.values

    if test_df is not None:
        # Transform test
        X_test = test_df.drop(columns=[target_col], errors='ignore')
        X_test_selected = selector.transform(X_test)

        return train_selected, X_test_selected, selector

    return train_selected, selector


if __name__ == "__main__":
    """
    Test du FeatureSelector avec des donnees fictives.
    """
    print("Test du FeatureSelector")
    print("="*60)

    # Creer donnees de test
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    # Features avec differentes caracteristiques
    X = pd.DataFrame()

    # Features informatives (correlees avec target)
    for i in range(10):
        X[f'INFORMATIVE_{i}'] = np.random.randn(n_samples)

    # Features bruitees
    for i in range(20):
        X[f'NOISE_{i}'] = np.random.randn(n_samples)

    # Features correlees entre elles
    X['CORR_A'] = np.random.randn(n_samples)
    X['CORR_B'] = X['CORR_A'] + np.random.randn(n_samples) * 0.01  # Tres correle
    X['CORR_C'] = X['CORR_A'] + np.random.randn(n_samples) * 0.01  # Tres correle

    # Features a faible variance
    X['LOW_VAR_1'] = np.ones(n_samples) + np.random.randn(n_samples) * 0.001
    X['LOW_VAR_2'] = np.zeros(n_samples)

    # Completer avec des features aleatoires
    for i in range(15):
        X[f'RANDOM_{i}'] = np.random.randn(n_samples)

    # Target (correle avec features informatives)
    y = (X[[f'INFORMATIVE_{i}' for i in range(5)]].mean(axis=1) > 0).astype(int)

    print(f"\nDonnees test: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Test du selecteur
    selector = FeatureSelector(
        methods=['variance', 'correlation', 'mutual_info', 'lightgbm'],
        n_features=20,
        variance_threshold=0.01,
        correlation_threshold=0.95
    )

    X_selected = selector.fit_transform(X, y)

    print(f"\nResultat: {X_selected.shape}")
    print(f"Features selectionnees: {selector.get_selected_features()[:10]}")

    # Test sauvegarde/chargement
    selector.save('artifacts/selected_features/test_selector.pkl')
    selector_loaded = FeatureSelector.load('artifacts/selected_features/test_selector.pkl')

    print("\nTests termines avec succes!")
