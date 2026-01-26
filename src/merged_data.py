import pandas as pd
import numpy as np
import gc
from pathlib import Path


class DataMerger:
    """Merging + agrégations uniquement (sans traitement des valeurs manquantes)"""

    def __init__(self, data_path='./data/raw', output_path='./data/merged'):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------
    def load_data(self):
        print("Chargement des données...")

        self.app_train = pd.read_csv(self.data_path / 'application_train.csv')
        self.app_test = pd.read_csv(self.data_path / 'application_test.csv')
        self.bureau = pd.read_csv(self.data_path / 'bureau.csv')
        self.bureau_balance = pd.read_csv(self.data_path / 'bureau_balance.csv')
        self.prev_app = pd.read_csv(self.data_path / 'previous_application.csv')
        self.pos_cash = pd.read_csv(self.data_path / 'POS_CASH_balance.csv')
        self.credit_card = pd.read_csv(self.data_path / 'credit_card_balance.csv')
        self.installments = pd.read_csv(self.data_path / 'installments_payments.csv')

        print(f"Train: {self.app_train.shape}")
        print(f"Test : {self.app_test.shape}")

    # ------------------------------------------------------------------
    # BUREAU
    # ------------------------------------------------------------------
    def create_bureau_features(self):
        print("\nCréation des features Bureau...")

        # Agrégation bureau_balance (OBLIGATOIRE AVANT MERGE)
        bb_agg = self.bureau_balance.groupby('SK_ID_BUREAU').agg({
            'MONTHS_BALANCE': ['min', 'max']
        })
        bb_agg.columns = ['BB_' + '_'.join(c).upper() for c in bb_agg.columns]
        bb_agg.reset_index(inplace=True)

        bureau = self.bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')

        # Numérique
        bureau_agg = bureau.groupby('SK_ID_CURR').agg({
            'DAYS_CREDIT': ['min', 'max', 'mean'],
            'AMT_CREDIT_SUM': ['mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'BB_MONTHS_BALANCE_MIN': 'min',
            'BB_MONTHS_BALANCE_MAX': 'max',
            'SK_ID_BUREAU': 'count'
        })

        bureau_agg.columns = ['BUREAU_' + '_'.join(c).upper() for c in bureau_agg.columns]
        bureau_agg.reset_index(inplace=True)

        # Catégoriel
        bureau_cat = pd.get_dummies(
            self.bureau[['SK_ID_CURR', 'CREDIT_ACTIVE', 'CREDIT_TYPE']],
            columns=['CREDIT_ACTIVE', 'CREDIT_TYPE']
        )

        bureau_cat = bureau_cat.groupby('SK_ID_CURR').sum().reset_index()
        bureau_cat.columns = ['SK_ID_CURR'] + [
            'BUREAU_' + c for c in bureau_cat.columns if c != 'SK_ID_CURR'
        ]

        bureau_features = bureau_agg.merge(bureau_cat, on='SK_ID_CURR', how='left')

        print(f"Bureau features: {bureau_features.shape}")

        del bureau, bureau_agg, bureau_cat, bb_agg
        gc.collect()

        return bureau_features

    # ------------------------------------------------------------------
    # PREVIOUS APPLICATION
    # ------------------------------------------------------------------
    def create_previous_application_features(self):
        print("\nCréation des features Previous Application...")

        prev_agg = self.prev_app.groupby('SK_ID_CURR').agg({
            'AMT_APPLICATION': ['mean', 'sum'],
            'AMT_CREDIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['mean'],
            'DAYS_DECISION': ['min', 'mean'],
            'CNT_PAYMENT': ['mean'],
            'SK_ID_PREV': 'count'
        })

        prev_agg.columns = ['PREV_' + '_'.join(c).upper() for c in prev_agg.columns]
        prev_agg.reset_index(inplace=True)

        prev_cat = pd.get_dummies(
            self.prev_app[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']],
            columns=['NAME_CONTRACT_STATUS']
        )

        prev_cat = prev_cat.groupby('SK_ID_CURR').sum().reset_index()
        prev_cat.columns = ['SK_ID_CURR'] + [
            'PREV_' + c for c in prev_cat.columns if c != 'SK_ID_CURR'
        ]

        prev_features = prev_agg.merge(prev_cat, on='SK_ID_CURR', how='left')

        print(f"Previous app features: {prev_features.shape}")

        del prev_agg, prev_cat
        gc.collect()

        return prev_features

    # ------------------------------------------------------------------
    # POS CASH
    # ------------------------------------------------------------------
    def create_pos_cash_features(self):
        print("\nCréation des features POS_CASH...")

        pos_agg = self.pos_cash.groupby('SK_ID_CURR').agg({
            'MONTHS_BALANCE': ['max', 'mean'],
            'CNT_INSTALMENT_FUTURE': ['mean'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']
        })

        pos_agg.columns = ['POS_' + '_'.join(c).upper() for c in pos_agg.columns]
        pos_agg.reset_index(inplace=True)

        print(f"POS_CASH features: {pos_agg.shape}")

        return pos_agg

    # ------------------------------------------------------------------
    # CREDIT CARD
    # ------------------------------------------------------------------
    def create_credit_card_features(self):
        print("\nCréation des features Credit Card...")

        cc_agg = self.credit_card.groupby('SK_ID_CURR').agg({
            'AMT_BALANCE': ['mean', 'sum'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['mean'],
            'SK_DPD': ['max'],
            'SK_DPD_DEF': ['max']
        })

        cc_agg.columns = ['CC_' + '_'.join(c).upper() for c in cc_agg.columns]
        cc_agg.reset_index(inplace=True)

        print(f"Credit Card features: {cc_agg.shape}")

        return cc_agg

    # ------------------------------------------------------------------
    # INSTALLMENTS
    # ------------------------------------------------------------------
    def create_installments_features(self):
        print("\nCréation des features Installments...")

        inst = self.installments.copy()
        inst['PAYMENT_DIFF'] = inst['AMT_PAYMENT'] - inst['AMT_INSTALMENT']
        inst['DAYS_DELAY'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
        inst.loc[inst['DAYS_DELAY'] < 0, 'DAYS_DELAY'] = 0

        inst_agg = inst.groupby('SK_ID_CURR').agg({
            'PAYMENT_DIFF': ['mean', 'sum'],
            'DAYS_DELAY': ['max', 'mean'],
            'AMT_PAYMENT': ['sum'],
            'SK_ID_PREV': 'count'
        })

        inst_agg.columns = ['INST_' + '_'.join(c).upper() for c in inst_agg.columns]
        inst_agg.reset_index(inplace=True)

        print(f"Installments features: {inst_agg.shape}")

        del inst
        gc.collect()

        return inst_agg

    # ------------------------------------------------------------------
    # MERGE FINAL
    # ------------------------------------------------------------------
    def run(self):
        print("=" * 60)
        print("DÉBUT DU MERGING (SANS TRAITEMENT NaN)")
        print("=" * 60)

        self.load_data()

        # Features calculées UNE SEULE FOIS
        bureau_feat = self.create_bureau_features()
        prev_feat = self.create_previous_application_features()
        pos_feat = self.create_pos_cash_features()
        cc_feat = self.create_credit_card_features()
        inst_feat = self.create_installments_features()

        # TRAIN
        train = self.app_train.copy()
        for feat in [bureau_feat, prev_feat, pos_feat, cc_feat, inst_feat]:
            train = train.merge(feat, on='SK_ID_CURR', how='left')

        # TEST
        test = self.app_test.copy()
        for feat in [bureau_feat, prev_feat, pos_feat, cc_feat, inst_feat]:
            test = test.merge(feat, on='SK_ID_CURR', how='left')

        # SAVE
        train.to_csv(self.output_path / 'train_merged.csv', index=False)
        test.to_csv(self.output_path / 'test_merged.csv', index=False)

        print("\nMERGING TERMINÉ")
        print(f"Train final: {train.shape}")
        print(f"Test final : {test.shape}")

        return train, test


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    merger = DataMerger()
    train_merged, test_merged = merger.run()

    print("\nRÉSUMÉ FINAL")
    print(f"Train: {train_merged.shape}")
    print(f"Test : {test_merged.shape}")
