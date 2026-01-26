"""
Preprocessors module for Credit Scoring Pipeline.

Contains:
- MissingValuesHandler: Handle missing values
- OutlierHandler: Handle outliers with various methods
- FeatureEngineer: Create new features
- CategoricalEncoder: Encode categorical variables
"""

from .missing_handler import MissingValuesHandler
from .outlier_handler import OutlierHandler
from .feature_engineer import FeatureEngineer
from .encoder import CategoricalEncoder

__all__ = [
    'MissingValuesHandler',
    'OutlierHandler',
    'FeatureEngineer',
    'CategoricalEncoder'
]
